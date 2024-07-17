from transformers import (
    PreTrainedTokenizerFast,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel,
    AutoTokenizer,
)
from typing import Union, Iterable, List, Dict, Callable
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path

import functools
import einops
import torch

from .hooks import direction_ablation_hook, get_orthogonalized_matrix
from .data import HarmfulHarmlessData
from .utils import clear_mem

torch.inference_mode()
torch.set_grad_enabled(False)


def decode(tokenizer: PreTrainedTokenizer, tokens_batch: torch.Tensor) -> List[str]:
    return tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)


def encode(
    tokenizer: PreTrainedTokenizerFast,
    device: torch.device,
    instructions: Iterable[str],
) -> torch.Tensor:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)


class Abliterator:

    def __init__(
        self,
        model_name: Union[str, Path],
        device: Union[str, torch.device] = "auto",
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

        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.encode_tokens = functools.partial(encode, tokenizer=self.tokenizer, device=self.model.device)
        self.decode_tokens = functools.partial(decode, tokenizer=self.tokenizer)

        self.generation_config = GenerationConfig(
            do_sample=False,
            num_beams=1,
        )

    def save_activation(self, activations, name, position):
        def hook(module, input, output):
            activations[name].append(output[:, position, :].detach().cpu())

        return hook

    def save_pre_activation(self, activations, name, position):
        def hook(module, input):
            activations[name].append(input[0][:, position, :].detach().cpu())

        return hook

    def temporary_hooks(self, activations, position):
        hooks = []
        for layer_idx, layer in enumerate(self.model.model.layers):
            if layer_idx == 0:
                continue
            hooks.append(
                layer.input_layernorm.register_forward_pre_hook(
                    self.save_pre_activation(activations, f"blocks.{layer_idx}.hook_resid_pre", position)
                )
            )
            hooks.append(
                layer.self_attn.o_proj.register_forward_hook(
                    self.save_activation(activations, f"blocks.{layer_idx}.hook_attn_out", position)
                )
            )
            hooks.append(
                layer.mlp.down_proj.register_forward_hook(self.save_activation(activations, f"blocks.{layer_idx}.hook_mlp_out", position))
            )
            hooks.append(
                layer.post_attention_layernorm.register_forward_hook(
                    self.save_activation(activations, f"blocks.{layer_idx}.hook_resid_post", position),
                )
            )
        return hooks

    def cache_activations(self, data: HarmfulHarmlessData, position: int = -1, eps: float = 1e-8):
        # Initialize defaultdicts to store activations
        harmful = defaultdict(list)
        harmless = defaultdict(list)

        # Process the training data in batches
        num_batches = (data.n_inst_train + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(num_batches), desc="Caching activations"):
            start_idx = i * self.batch_size
            end_idx = min(data.n_inst_train, start_idx + self.batch_size)

            harmful_inputs = self.encode_tokens(instructions=data.harmful["train"][start_idx:end_idx])
            harmless_inputs = self.encode_tokens(instructions=data.harmless["train"][start_idx:end_idx])

            # Register hooks and run model for harmful prompts
            hooks = self.temporary_hooks(harmful, position)
            self.model.generate(inputs=harmful_inputs, max_new_tokens=1, generation_config=self.generation_config)
            for hook in hooks:
                hook.remove()

            # Register hooks and run model for harmless prompts
            hooks = self.temporary_hooks(harmless, position)
            self.model.generate(inputs=harmless_inputs, max_new_tokens=1, generation_config=self.generation_config)
            for hook in hooks:
                hook.remove()

            # Flush RAM and VRAM
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

    def _generate_with_hook(self, tokens: torch.Tensor, hook_fn: Callable = None, max_tokens_generated=None) -> List[str]:

        hooks = []
        if hook_fn:
            for layer in self.model.model.layers:
                hooks.append(layer.register_forward_hook(hook_fn))

        output_sequences = self.model.generate(
            input_ids=tokens,
            max_new_tokens=max_tokens_generated,
            generation_config=self.generation_config,
        )

        for hook_fn in hooks:
            hook_fn.remove()

        return self.decode_tokens(tokens_batch=output_sequences[:, tokens.size(-1) :].detach().cpu())

    def generate(
        self,
        instructions: List[str],
        max_tokens_generated: int = None,
        hook_fn: Callable = None,
    ) -> List[str]:
        generations = []

        max_tokens_generated = max_tokens_generated or self.max_tokens_generated
        for i in range(0, len(instructions), self.batch_size):
            tokens = self.encode_tokens(instructions=instructions[i : i + self.batch_size])
            generation = self._generate_with_hook(tokens, hook_fn, max_tokens_generated)
            generations.extend([g for g in generation])

        return generations

    def test_refusal_directions(self, instructions: List[str]):
        pbar = tqdm(self.refusal_directions)
        for refusal_direction in pbar:
            hook_fn = direction_ablation_hook(refusal_direction["refusal_direction"])
            refusal_direction["intervention_generation"] = self.generate(instructions, hook_fn=hook_fn)

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
        layers = layers or list(range(1, len(self.model.model.layers)))
        if attn_output or mlp:
            self.modified = True

        for refusal_direction in layer_rankings:
            refusal_direction = refusal_direction["refusal_direction"]

            for layer in tqdm(layers, leave=False):
                block = self.model.model.layers[layer]
                if refusal_direction.device != self.model.device:
                    refusal_direction = refusal_direction.to(self.model.device)
                if attn_output:
                    block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data, refusal_direction)
                    self.modified_layers["attention_output_layer"].append(layer)
                if mlp:
                    block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data, refusal_direction)
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
