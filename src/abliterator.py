from transformer_lens import HookedTransformer, utils
from transformers import PreTrainedTokenizer
from typing import Union, Iterable, List
from collections import defaultdict
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from colorama import Fore
from pathlib import Path
from torch import Tensor

import functools
import textwrap
import requests
import einops
import torch
import io

from .hooks import direction_ablation_hook
from .utils import clear_mem

torch.set_grad_enabled(False)


def get_input_ids(tokenizer: PreTrainedTokenizer, instructions: Iterable[str]) -> Int[Tensor, "batch_size seq_len"]:
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def decode(tokenizer: PreTrainedTokenizer, tokens_batch: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)


class Abliterator:

    def __init__(
        self,
        model_name: Union[str, Path],
        data: dict,
        device: Union[str, torch.device] = "cuda",
        n_inst_train: int = 256,
        activation_layers: Iterable[str] = ["resid_pre", "resid_mid", "resid_post"],
        max_tokens_generated: int = 64,
    ):
        self.activation_layers = set(activation_layers or ["resid_pre", "resid_mid", "resid_post"])
        self.max_tokens_generated = max_tokens_generated

        # load model as HookedTransformer from transformer_lens
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name=model_name,
            device=device,
            dtype=torch.bfloat16,
            default_padding_side="left",
        )

        # padding on the left (start of sentence)
        self.model.tokenizer.padding_side = "left"
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        # assign decode tokens fn for later
        self.decode_tokens = functools.partial(decode, tokenizer=self.model.tokenizer)

        # get datasets
        self.harmful_inst_train, self.harmful_inst_test = data["harmful"]
        self.harmless_inst_train, self.harmless_inst_test = data["harmless"]

        # limit number of instances to speed up the process
        self.n_inst_train = min(n_inst_train, len(self.harmful_inst_train), len(self.harmless_inst_train))

        # Tokenize datasets
        self.harmful_tokens = get_input_ids(
            tokenizer=self.model.tokenizer,
            instructions=self.harmful_inst_train[: self.n_inst_train],
        )
        self.harmless_tokens = get_input_ids(
            tokenizer=self.model.tokenizer,
            instructions=self.harmless_inst_train[: self.n_inst_train],
        )

        self.cache_activations()
        self.calculate_refusal_directions()

    def cache_activations(self):
        # Define batch size based on available VRAM
        batch_size = 32

        # Initialize defaultdicts to store activations
        harmful = defaultdict(list)
        harmless = defaultdict(list)

        # Process the training data in batches
        num_batches = (self.n_inst_train + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches)):
            start_idx = i * batch_size
            end_idx = min(self.n_inst_train, start_idx + batch_size)

            # Run models on harmful and harmless prompts, cache activations
            harmful_logits, harmful_cache = self.model.run_with_cache(
                self.harmful_tokens[start_idx:end_idx],
                names_filter=lambda hook_name: "resid" in hook_name,
                device="cpu",
                reset_hooks_end=True,
            )
            harmless_logits, harmless_cache = self.model.run_with_cache(
                self.harmless_tokens[start_idx:end_idx],
                names_filter=lambda hook_name: "resid" in hook_name,
                device="cpu",
                reset_hooks_end=True,
            )

            # Collect and store the activations
            for key in harmful_cache:
                harmful[key].append(harmful_cache[key])
                harmless[key].append(harmless_cache[key])

            # Flush RAM and VRAM
            del harmful_logits, harmless_logits, harmful_cache, harmless_cache
            clear_mem()

        # Concatenate the cached activations
        self.harmful_cache = {k: torch.cat(v) for k, v in harmful.items()}
        self.harmless_cache = {k: torch.cat(v) for k, v in harmless.items()}

    def calculate_refusal_directions(self, position: int = -1, eps: float = 1e-8):

        self.refusal_directions = []

        for layer_num in range(self.model.cfg.n_layers):
            for act_name in self.activation_layers:
                cache_key = utils.get_act_name(act_name, layer_num)

                # calculate mean across batch dim, on the position of the next generated token
                harmful_mean = self.harmful_cache[cache_key][:, position, :].mean(dim=0)
                harmless_mean = self.harmless_cache[cache_key][:, position, :].mean(dim=0)

                refusal_direction = harmful_mean - harmless_mean
                normalized_refusal_direction = refusal_direction / (refusal_direction.norm() + eps)

                self.refusal_directions.append(
                    {
                        "act_name": act_name,
                        "layer_num": layer_num,
                        "refusal_direction": normalized_refusal_direction,
                        "refusal_direction_mean": abs(normalized_refusal_direction.mean()),
                    }
                )

        del self.harmful_cache, self.harmless_cache
        clear_mem()

        self.refusal_directions.sort(key=lambda x: x["refusal_direction_mean"], reverse=True)

    def _generate_with_hooks(self, tokens: Int[Tensor, "batch_size seq_len"], fwd_hooks=[]) -> List[str]:
        all_tokens = torch.zeros((tokens.shape[0], tokens.shape[1] + self.max_tokens_generated), dtype=torch.long, device=tokens.device)
        all_tokens[:, : tokens.shape[1]] = tokens
        for i in range(self.max_tokens_generated):
            with self.model.hooks(fwd_hooks=fwd_hooks):
                logits = self.model(all_tokens[:, : -self.max_tokens_generated + i])
                next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
                all_tokens[:, -self.max_tokens_generated + i] = next_tokens
        return self.decode_tokens(all_tokens[:, tokens.shape[1] :])

    def get_generations(self, instructions: List[str], fwd_hooks=[], batch_size: int = 4) -> List[str]:
        generations = []
        for i in tqdm(range(0, len(instructions), batch_size)):
            tokens = get_input_ids(tokenizer=self.model.tokenizer, instructions=instructions[i : i + batch_size])
            generation = self._generate_with_hooks(tokens, fwd_hooks=fwd_hooks)
            generations.extend(generation)
        return generations
