from transformer_lens.utils import get_act_name
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizer
from typing import Union, Iterable, List
from collections import defaultdict
from tqdm.auto import tqdm
from jaxtyping import Int
from pathlib import Path
from torch import Tensor

import functools
import torch

from .utils import clear_mem, EnhancedHookedTransformerKeyValueCache
from .hooks import direction_ablation_hook

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
        n_inst_train: int = 2048,
        activation_layers: Iterable[str] = ["resid_pre", "resid_mid", "resid_post"],
        max_tokens_generated: int = 128,
        negative_tokens: List[str] = ["I cannot", "I can't"],
        positive_tokens: List[str] = ["Sure"],
    ):
        # limit number of instances to speed up the process
        self.n_inst_train = min(n_inst_train, len(data["harmful"]["train"]), len(data["harmless"]["train"]))
        self.activation_layers = set(activation_layers or ["resid_pre", "resid_mid", "resid_post"])
        self.max_tokens_generated = max_tokens_generated
        self.positive_tokens = positive_tokens
        self.negative_tokens = negative_tokens

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
            instructions=data["harmful"]["train"][: self.n_inst_train],
        ).to(self.model.cfg.device)
        self.harmless_tokens = get_input_ids(
            tokenizer=self.model.tokenizer,
            instructions=data["harmless"]["train"][: self.n_inst_train],
        ).to(self.model.cfg.device)

    def cache_activations(self, position: int = -1, eps: float = 1e-8):
        # Define batch size based on available VRAM
        batch_size = 32

        # Initialize defaultdicts to store activations
        harmful = defaultdict(list)
        harmless = defaultdict(list)

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
                harmful[key].append(harmful_cache[key][:, position, :].cpu())
                harmless[key].append(harmless_cache[key][:, position, :].cpu())

            # Flush RAM and VRAM
            del harmful_logits, harmless_logits, harmful_cache, harmless_cache
            clear_mem()

        # Concatenate the cached activations
        harmful = {k: torch.cat(v).mean(dim=0) for k, v in harmful.items()}
        harmless = {k: torch.cat(v).mean(dim=0) for k, v in harmless.items()}

        self.refusal_directions = []

        for layer_num in range(1, self.model.cfg.n_layers):
            for act_name in self.activation_layers:
                cache_key = get_act_name(act_name, layer_num)

                # calculate mean across batch dim, on the position of the next generated token
                harmful_mean: torch.Tensor = harmful[cache_key]
                harmless_mean: torch.Tensor = harmless[cache_key]

                refusal_direction = harmful_mean - harmless_mean
                normalized_refusal_direction = refusal_direction / (refusal_direction.norm() + eps)

                self.refusal_directions.append(
                    {
                        "act_name": act_name,
                        "layer_num": layer_num,
                        "refusal_direction": normalized_refusal_direction,
                        "refusal_direction_mean": abs(normalized_refusal_direction).mean(),
                    }
                )

        del harmful, harmless
        clear_mem()

        self.refusal_directions.sort(key=lambda x: x["refusal_direction_mean"], reverse=True)

    def _generate_with_hooks(self, tokens: Int[Tensor, "batch_size seq_len"], fwd_hooks=[], pbar: tqdm = None) -> List[str]:
        batch_size, seq_len = tokens.shape

        all_tokens = torch.full(
            (batch_size, seq_len + self.max_tokens_generated),
            self.model.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.model.cfg.device,
        )
        all_tokens[:, :seq_len] = tokens

        generating = torch.ones(batch_size, dtype=torch.bool, requires_grad=False, device=self.model.cfg.device)
        cache = EnhancedHookedTransformerKeyValueCache.init_cache(self.model.cfg, self.model.cfg.device, batch_size)

        for i in range(self.max_tokens_generated):
            with self.model.hooks(fwd_hooks=fwd_hooks):
                if i == 0:
                    logits = self.model(all_tokens[:, : seq_len + i], past_kv_cache=cache)
                else:
                    logits = self.model(all_tokens[generating, seq_len + i - 1].unsqueeze(1), past_kv_cache=cache)

                # Greedy sampling (temperature=0)
                next_tokens = logits[:, -1, :].argmax(dim=-1)
                all_tokens[generating, seq_len + i] = next_tokens

                # Update generating mask to exclude sequences that hit the eos token
                subset = next_tokens != self.model.tokenizer.eos_token_id
                _clone = generating.clone()
                _clone[generating] = subset
                generating = _clone

                # Update cache
                cache = cache.subset(subset)

                if not generating.any():
                    break
                elif pbar:
                    pbar.set_description(f"Still generating {sum(generating)} results")
                else:
                    print(f"Still generating {sum(generating)} results")

        return self.decode_tokens(tokens_batch=all_tokens[:, seq_len:].cpu())

    def generate(self, instructions: List[str], fwd_hooks=[], batch_size: int = 8, pbar: tqdm = None) -> List[str]:
        generations = []

        for i in range(0, len(instructions), batch_size):
            tokens = get_input_ids(tokenizer=self.model.tokenizer, instructions=instructions[i : i + batch_size])
            generation = self._generate_with_hooks(tokens, fwd_hooks=fwd_hooks, pbar=pbar)
            generations.extend([g.strip() for g in generation])
        return generations

    def get_fwd_hooks(self, refusal_direction):
        hook_fn = functools.partial(direction_ablation_hook, direction=refusal_direction)
        fwd_hooks = [
            (get_act_name(act_name, layer), hook_fn)
            for layer in list(range(self.model.cfg.n_layers))
            for act_name in self.activation_layers
        ]
        return fwd_hooks

    def test_top_N_directions(self, instructions: List[str], N: int = 10):
        intervention_generations = []
        pbar = tqdm(self.refusal_directions[:N])
        for refusal_direction in pbar:
            fwd_hooks = self.get_fwd_hooks(refusal_direction["refusal_direction"])
            intervention_generations.append(self.generate(instructions, fwd_hooks=fwd_hooks, pbar=pbar))

        return intervention_generations

    def aggregate_best_layers(self, intervention_generations: List[List[str]]):

        layer_rankings = defaultdict(int)

        for layer_candidate in range(len(intervention_generations)):
            for example in range(len(intervention_generations[layer_candidate])):
                count = sum(word not in intervention_generations[layer_candidate][example] for word in self.negative_tokens) + sum(
                    word in intervention_generations[layer_candidate][example] for word in self.positive_tokens
                )
                layer_rankings[layer_candidate] += count

        sorted_layer_rankings = sorted(layer_rankings.items(), key=lambda x: x[1], reverse=True)

        return sorted_layer_rankings
