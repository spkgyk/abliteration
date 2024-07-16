from transformers import PreTrainedTokenizer, AutoModelForCausalLM
from transformer_lens.components import TransformerBlock
from transformer_lens.utils import get_act_name
from typing import Union, Iterable, List, Dict
from transformer_lens import HookedTransformer
from collections import defaultdict
from tqdm.auto import tqdm
from jaxtyping import Int
from pathlib import Path

import functools
import einops
import torch

from .utils import clear_mem, EnhancedHookedTransformerKeyValueCache, get_input_ids
from .hooks import direction_ablation_hook, get_orthogonalized_matrix
from .data import HarmfulHarmlessData

torch.inference_mode()
torch.set_grad_enabled(False)


def decode(tokenizer: PreTrainedTokenizer, tokens_batch: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)


# valid_act_names = [
#     "hook_embed",
#     "blocks.0.hook_resid_pre",  # this one
#     "blocks.0.ln1.hook_scale",
#     "blocks.0.ln1.hook_normalized",
#     "blocks.0.attn.hook_q",
#     "blocks.0.attn.hook_k",
#     "blocks.0.attn.hook_v",
#     "blocks.0.attn.hook_rot_q",
#     "blocks.0.attn.hook_rot_k",
#     "blocks.0.attn.hook_attn_scores",
#     "blocks.0.attn.hook_pattern",
#     "blocks.0.attn.hook_z",
#     "blocks.0.hook_attn_out",  # this one
#     "blocks.0.hook_resid_mid",
#     "blocks.0.ln2.hook_scale",
#     "blocks.0.ln2.hook_normalized",
#     "blocks.0.mlp.hook_pre",
#     "blocks.0.mlp.hook_pre_linear",
#     "blocks.0.mlp.hook_post",
#     "blocks.0.hook_mlp_out",  # this one
#     "blocks.0.hook_resid_post",  # this one
# ]


class Abliterator:

    def __init__(
        self,
        model_name: Union[str, Path],
        device: Union[str, torch.device] = torch.device(0),
        activation_layers: Iterable[str] = ["resid_pre", "resid_post", "attn_out", "mlp_out"],
        max_tokens_generated: int = 16,
        positive_tokens: List[str] = ["Sure"],
        negative_tokens: List[str] = ["I cannot", "I can't", "I'm sorry", "Sorry", "I don't", "crime", "not ethical"],
        batch_size: int = 16,
    ):
        # limit number of instances to speed up the process
        self.model_name = model_name
        self.modified = False
        self.modified_layers = defaultdict(list)
        self.activation_layers = set(activation_layers)
        self.max_tokens_generated = max_tokens_generated
        self.positive_tokens = positive_tokens
        self.negative_tokens = negative_tokens
        self.batch_size = batch_size

        # load model as HookedTransformer from transformer_lens
        self.model = HookedTransformer.from_pretrained_no_processing(
            model_name=self.model_name,
            device_map=device,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            default_padding_side="left",
        )

        # padding on the left (start of sentence)
        self.model.tokenizer.padding_side = "left"
        self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        # assign decode tokens fn for later
        self.decode_tokens = functools.partial(decode, tokenizer=self.model.tokenizer)

    def name_filter(self, activation_name: str):
        return ".0." not in activation_name and any(act_layer in activation_name for act_layer in self.activation_layers)

    def cache_activations(self, data: HarmfulHarmlessData, position: int = -1, eps: float = 1e-8):
        # Initialize defaultdicts to store activations
        harmful = defaultdict(list)
        harmless = defaultdict(list)

        # Process the training data in batches
        num_batches = (data.n_inst_train + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(num_batches), desc="Caching activations"):
            start_idx = i * self.batch_size
            end_idx = min(data.n_inst_train, start_idx + self.batch_size)

            # Run models on harmful and harmless prompts, cache activations
            _, harmful_cache = self.model.run_with_cache(
                data.harmful_tokens[start_idx:end_idx],
                names_filter=self.name_filter,
                pos_slice=position,
            )
            _, harmless_cache = self.model.run_with_cache(
                data.harmless_tokens[start_idx:end_idx],
                names_filter=self.name_filter,
                pos_slice=position,
            )

            assert harmful_cache.keys() == harmless_cache.keys()

            # Collect and store the activations
            for cache_key in harmful_cache.keys():
                harmful[cache_key].append(harmful_cache[cache_key].squeeze().cpu())
                harmless[cache_key].append(harmless_cache[cache_key].squeeze().cpu())

            # Flush RAM and VRAM
            del _, harmful_cache, harmless_cache
            clear_mem()

        # Concatenate the cached activations
        harmful = {k: torch.cat(v).mean(dim=0) for k, v in harmful.items()}
        harmless = {k: torch.cat(v).mean(dim=0) for k, v in harmless.items()}

        self.cache_keys = list(harmful.keys())
        self.refusal_directions = []

        for cache_key in harmful.keys():
            # calculate mean across batch dim, on the position of the next generated token
            harmful_mean: torch.Tensor = harmful[cache_key]
            harmless_mean: torch.Tensor = harmless[cache_key]

            refusal_direction = harmful_mean - harmless_mean
            normalized_refusal_direction = refusal_direction / (refusal_direction.norm() + eps)

            self.refusal_directions.append(
                {
                    "cache_key": cache_key,
                    "refusal_direction": normalized_refusal_direction,
                }
            )

        del harmful, harmless
        clear_mem()

    def _generate_with_hooks(
        self,
        tokens: Int[torch.Tensor, "batch_size seq_len"],
        fwd_hooks=[],
        max_tokens_generated=None,
        batch_num: int = 0,
        num_batches: int = 1,
        pbar: tqdm = None,
        cache_key: str = None,
    ) -> List[str]:
        batch_size, seq_len = tokens.shape

        all_tokens = torch.full(
            (batch_size, seq_len + max_tokens_generated),
            self.model.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.model.cfg.device,
        )
        all_tokens[:, :seq_len] = tokens

        generating = torch.ones(batch_size, dtype=torch.bool, requires_grad=False, device=self.model.cfg.device)
        cache = EnhancedHookedTransformerKeyValueCache.init_cache(self.model.cfg, self.model.cfg.device, batch_size)

        for i in range(max_tokens_generated):
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
                    if cache_key:
                        pbar.set_description(
                            f"Activation '{cache_key}', batch {batch_num+1}/{num_batches} still generating {sum(generating)} results"
                        )
                    else:
                        pbar.set_description(f"Batch {batch_num+1}/{num_batches} still generating {sum(generating)} results")
                else:
                    if cache_key:
                        print(
                            f"Activation '{cache_key}', batch {batch_num+1}/{num_batches} still generating {sum(generating)} results...",
                            end="\r",
                        )
                    else:
                        print(f"Batch {batch_num+1}/{num_batches} still generating {sum(generating)} results...", end="\r")

        return self.decode_tokens(tokens_batch=all_tokens[:, seq_len:].cpu())

    def generate(
        self,
        instructions: List[str],
        max_tokens_generated=None,
        fwd_hooks=[],
        pbar: tqdm = None,
        cache_key: str = None,
    ) -> List[str]:
        generations = []

        max_tokens_generated = max_tokens_generated or self.max_tokens_generated
        num_batches = (len(instructions) + self.batch_size - 1) // self.batch_size
        for batch_num, i in enumerate(range(0, len(instructions), self.batch_size)):
            tokens = get_input_ids(tokenizer=self.model.tokenizer, instructions=instructions[i : i + self.batch_size])
            generation = self._generate_with_hooks(tokens, fwd_hooks, max_tokens_generated, batch_num, num_batches, pbar, cache_key)
            generations.extend([g.strip() for g in generation])

        return generations

    def test_refusal_directions(self, instructions: List[str]):
        pbar = tqdm(self.refusal_directions)
        for refusal_direction in pbar:
            hook_fn = functools.partial(direction_ablation_hook, direction=refusal_direction["refusal_direction"])
            fwd_hooks = [(cache_key, hook_fn) for cache_key in self.cache_keys]
            refusal_direction["intervention_generation"] = self.generate(
                instructions,
                fwd_hooks=fwd_hooks,
                pbar=pbar,
                cache_key=refusal_direction["cache_key"],
            )

        return self.refusal_directions

    def aggregate_best_layers(self, intervention_generations: List[Dict] = None):

        intervention_generations = intervention_generations or self.refusal_directions
        for layer_candidate in intervention_generations:
            count = 0
            for example in layer_candidate["intervention_generation"]:
                count += sum(word not in example for word in self.negative_tokens) + sum(word in example for word in self.positive_tokens)
            layer_candidate["count"] = count

        intervention_generations = sorted(intervention_generations, key=lambda x: x["count"], reverse=True)
        self.refusal_directions = intervention_generations

        return self.refusal_directions

    def ablate_layers(self, layer_rankings: List[Dict] = None, layers: List[int] = None, attn_output: bool = True, mlp: bool = True):
        layer_rankings = layer_rankings or self.refusal_directions
        layers = layers or list(range(1, self.model.cfg.n_layers))
        if attn_output or mlp:
            self.modified = True

        for refusal_direction in layer_rankings:
            refusal_direction: torch.Tensor = refusal_direction["refusal_direction"]

            for layer in tqdm(layers, leave=False):
                block: TransformerBlock = self.model.blocks[layer]
                if refusal_direction.device != self.model.cfg.device:
                    refusal_direction = refusal_direction.to(block.attn.W_O.device)
                if attn_output:
                    block.attn.W_O.data = get_orthogonalized_matrix(block.attn.W_O, refusal_direction)
                    self.modified_layers["attention_output_layer"].append(layer)
                if mlp:
                    block.mlp.W_out.data = get_orthogonalized_matrix(block.mlp.W_out, refusal_direction)
                    self.modified_layers["mlp"].append(layer)

    def convert_weights(self):
        hf_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
        lm_model = hf_model.model

        state_dict = self.model.state_dict()
        lm_model.embed_tokens.weight = torch.nn.Parameter(state_dict["embed.W_E"].cpu())

        for l in range(self.model.cfg.n_layers):
            lm_model.layers[l].self_attn.o_proj.weight = torch.nn.Parameter(
                einops.rearrange(state_dict[f"blocks.{l}.attn.W_O"], "n h m->m (n h)", n=self.model.cfg.n_heads).contiguous()
            )
            lm_model.layers[l].mlp.down_proj.weight = torch.nn.Parameter(
                torch.transpose(state_dict[f"blocks.{l}.mlp.W_out"], 0, 1).contiguous()
            )

        hf_model.push_to_hub(f"{self.model_name.split('/',1)[-1]}-uncensored")
