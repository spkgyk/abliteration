from transformer_lens import HookedTransformer, utils, HookedTransformerKeyValueCache
from transformers import PreTrainedTokenizer
from typing import Union, Iterable, List
from collections import defaultdict
from tqdm.auto import tqdm
from jaxtyping import Int
from pathlib import Path
from torch import Tensor

import functools
import torch

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
        n_inst_test: int = 8,
        activation_layers: Iterable[str] = ["resid_pre", "resid_mid", "resid_post"],
        max_tokens_generated: int = 128,
    ):
        # limit number of instances to speed up the process
        self.n_inst_train = min(n_inst_train, len(data["harmful"][0]), len(data["harmless"][0]))
        self.n_inst_test = n_inst_test
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

        # Tokenize datasets
        self.harmful_tokens = get_input_ids(
            tokenizer=self.model.tokenizer,
            instructions=data["harmful"][0][: self.n_inst_train],
        )
        self.harmless_tokens = get_input_ids(
            tokenizer=self.model.tokenizer,
            instructions=data["harmless"][0][: self.n_inst_train],
        )

        self.cache_activations()
        self.calculate_refusal_directions()

    def cache_activations(self, position=-1):
        # Define batch size based on available VRAM
        batch_size = 32

        # Initialize defaultdicts to store activations
        self.harmful_cache = defaultdict(list)
        self.harmless_cache = defaultdict(list)

        # Process the training data in batches
        num_batches = (self.n_inst_train + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Caching activations"):
            start_idx = i * batch_size
            end_idx = min(self.n_inst_train, start_idx + batch_size)

            # Run models on harmful and harmless prompts, cache activations
            harmful_logits, harmful_cache = self.model.run_with_cache(
                self.harmful_tokens[start_idx:end_idx],
                names_filter=lambda hook_name: "resid" in hook_name,
                reset_hooks_end=True,
            )
            harmless_logits, harmless_cache = self.model.run_with_cache(
                self.harmless_tokens[start_idx:end_idx],
                names_filter=lambda hook_name: "resid" in hook_name,
                reset_hooks_end=True,
            )

            # Collect and store the activations
            for key in harmful_cache:
                self.harmful_cache[key].append(harmful_cache[key][:, position, :].cpu())
                self.harmless_cache[key].append(harmless_cache[key][:, position, :].cpu())

            # Flush RAM and VRAM
            del harmful_logits, harmless_logits, harmful_cache, harmless_cache
            clear_mem()

        # Concatenate the cached activations
        self.harmful_cache = {k: torch.cat(v).mean(dim=0) for k, v in self.harmful_cache.items()}
        self.harmless_cache = {k: torch.cat(v).mean(dim=0) for k, v in self.harmless_cache.items()}

    def calculate_refusal_directions(self, eps: float = 1e-8):

        self.refusal_directions = []

        for layer_num in tqdm(range(1, self.model.cfg.n_layers), desc="Calculating refusal directions"):
            for act_name in self.activation_layers:
                cache_key = utils.get_act_name(act_name, layer_num)

                # calculate mean across batch dim, on the position of the next generated token
                harmful_mean: torch.Tensor = self.harmful_cache[cache_key]
                harmless_mean: torch.Tensor = self.harmless_cache[cache_key]

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
        batch_size, seq_len = tokens.shape
        all_tokens = torch.zeros((batch_size, seq_len + self.max_tokens_generated), dtype=torch.long, device=tokens.device)
        all_tokens[:, :seq_len] = tokens

        cache = HookedTransformerKeyValueCache.init_cache(self.model.cfg, self.model.cfg.device, batch_size)

        for i in range(self.max_tokens_generated):
            with self.model.hooks(fwd_hooks=fwd_hooks):
                if i == 0:
                    logits = self.model(all_tokens[:, : seq_len + i], past_kv_cache=cache)
                else:
                    logits = self.model(all_tokens[:, seq_len + i - 1 : seq_len + i], past_kv_cache=cache)

                next_tokens = logits[:, -1, :].argmax(dim=-1)  # greedy sampling (temperature=0)
                all_tokens[:, seq_len + i] = next_tokens

        return self.decode_tokens(tokens_batch=all_tokens[:, seq_len:])

    def generate(self, instructions: List[str], fwd_hooks=[], batch_size: int = 8) -> List[str]:
        generations = []
        _range = range(0, len(instructions), batch_size)
        pbar = tqdm(_range) if len(instructions) > batch_size else _range
        for i in pbar:
            tokens = get_input_ids(tokenizer=self.model.tokenizer, instructions=instructions[i : i + batch_size])
            generation = self._generate_with_hooks(tokens, fwd_hooks=fwd_hooks)
            generations.extend(generation)
        return generations

    def get_fwd_hooks(self, refusal_direction):
        hook_fn = functools.partial(direction_ablation_hook, direction=refusal_direction)
        fwd_hooks = [
            (utils.get_act_name(act_name, layer), hook_fn)
            for layer in list(range(self.model.cfg.n_layers))
            for act_name in self.activation_layers
        ]
        return fwd_hooks

    def test_top_N_directions(self, instructions: List[str], N: int = 10):
        intervention_generations = []
        for refusal_direction in tqdm(self.refusal_directions[:N]):
            fwd_hooks = self.get_fwd_hooks(refusal_direction["refusal_direction"])
            intervention_generations.append(self.generate(instructions, fwd_hooks=fwd_hooks))

        return intervention_generations
