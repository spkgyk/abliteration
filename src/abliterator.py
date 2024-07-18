from transformers import (
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    GenerationConfig,
    PreTrainedModel,
    AutoTokenizer,
)
from typing import Union, List, Dict, Callable
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import torch

from .hooks import direction_ablation_hook, get_orthogonalized_matrix
from .data import HarmfulHarmlessData
from .utils import clear_mem

torch.set_grad_enabled(False)
torch.inference_mode()


class Abliterator:
    def __init__(
        self,
        model_name: Union[str, Path],
        batch_size: int = 16,
        max_tokens_generated: int = 24,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        positive_tokens: List[str] = ["Sure", "To", "Certainly"],
        negative_tokens: List[str] = [
            "I cannot",
            "I can't",
            "I'm sorry",
            "Sorry",
            "I don't",
            "crime",
            "not ethical",
            "dangerous",
            "illegal",
            "unethical",
            "Combat",
            "inappropriate",
            "no justification",
        ],
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens_generated = max_tokens_generated
        self.device = torch.device(device)
        self.positive_tokens = positive_tokens
        self.negative_tokens = negative_tokens
        self.modified = False
        self.modified_layers = defaultdict(list)
        self.refusal_directions = []

        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.generation_config = GenerationConfig(do_sample=False, num_beams=1, pad_token_id=self.tokenizer.pad_token_id)

        self._print_model_layers()

    def _load_model(self) -> PreTrainedModel:
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        tokenizer.padding_side = "left"
        return tokenizer

    def _print_model_layers(self):
        for layer_idx, layer in enumerate(self.model.model.layers):
            print(f"Layer {layer_idx}:")
            print(layer)
            print("---")

    def encode_tokens(self, instructions: List[str]) -> torch.Tensor:
        return self.tokenizer.apply_chat_template(
            instructions,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        ).input_ids.to(self.device)

    def decode_tokens(self, tokens_batch: torch.Tensor) -> List[str]:
        return self.tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)

    def _create_hook(self, activations: Dict[str, List[torch.Tensor]], name: str, position: int, pre: bool = False):
        def hook(module, input, output=None):
            tensor = input[0] if pre else output
            activations[name].append(tensor[:, position, :].detach().cpu())

        return hook

    def _register_hooks(self, activations: Dict[str, List[torch.Tensor]], position: int) -> List[Callable]:
        useful_layers = max(int(0.5 * len(self.model.model.layers)), 1)
        hooks = []

        for layer_idx, layer in enumerate(self.model.model.layers[useful_layers:], start=useful_layers):
            layer_name = f"blocks.{layer_idx}"
            hooks.extend(
                [
                    layer.input_layernorm.register_forward_pre_hook(
                        self._create_hook(activations, f"{layer_name}.resid_pre", position, pre=True)
                    ),
                    layer.self_attn.o_proj.register_forward_hook(self._create_hook(activations, f"{layer_name}.attn_out", position)),
                    layer.mlp.down_proj.register_forward_hook(self._create_hook(activations, f"{layer_name}.mlp_out", position)),
                    layer.post_attention_layernorm.register_forward_hook(
                        self._create_hook(activations, f"{layer_name}.resid_post", position)
                    ),
                ]
            )
        return hooks

    def cache_activations(self, data: HarmfulHarmlessData, position: int = -1, eps: float = 1e-8):
        harmful = self._process_dataset(data.harmful["train"], "harmful", position)
        harmless = self._process_dataset(data.harmless["train"], "harmless", position)

        self.cache_keys = list(harmful.keys())
        self.refusal_directions = self._calculate_refusal_directions(harmful, harmless, eps)

        del harmful, harmless
        clear_mem()

    def _process_dataset(self, dataset: List[str], dataset_name: str, position: int) -> Dict[str, torch.Tensor]:
        activations = defaultdict(list)
        hooks = self._register_hooks(activations, position)

        for i in tqdm(range(0, len(dataset), self.batch_size), desc=f"Caching {dataset_name} activations"):
            batch = dataset[i : i + self.batch_size]
            inputs = self.encode_tokens(batch)
            self.model.generate(inputs=inputs, max_new_tokens=1, generation_config=self.generation_config)
            clear_mem()

        for hook in hooks:
            hook.remove()

        return {k: torch.cat(v).mean(dim=0) for k, v in activations.items()}

    def _calculate_refusal_directions(
        self, harmful: Dict[str, torch.Tensor], harmless: Dict[str, torch.Tensor], eps: float
    ) -> List[Dict[str, Union[str, torch.Tensor]]]:
        return [
            {"cache_key": key, "refusal_direction": (harmful[key] - harmless[key]) / ((harmful[key] - harmless[key]).norm() + eps)}
            for key in self.cache_keys
        ]

    def generate(self, instructions: List[str], max_tokens_generated: int = None, hook_fn: Callable = None) -> List[str]:
        generations = []
        hooks = []

        if hook_fn:
            for layer in self.model.model.layers:
                hooks.append(layer.self_attn.o_proj.register_forward_hook(hook_fn))
                hooks.append(layer.mlp.down_proj.register_forward_hook(hook_fn))

        max_tokens_generated = max_tokens_generated or self.max_tokens_generated

        for i in range(0, len(instructions), self.batch_size):
            batch = instructions[i : i + self.batch_size]
            tokens = self.encode_tokens(batch)

            output_sequences = self.model.generate(
                input_ids=tokens,
                max_new_tokens=max_tokens_generated,
                generation_config=self.generation_config,
            )
            output_sequences = output_sequences[:, tokens.size(-1) :].detach().cpu()
            generations.extend(self.decode_tokens(output_sequences))

        for hook in hooks:
            hook.remove()

        return generations

    def test_refusal_directions(self, instructions: List[str]):
        for refusal_direction in tqdm(self.refusal_directions):
            hook_fn = direction_ablation_hook(refusal_direction["refusal_direction"])
            refusal_direction["intervention_generation"] = self.generate(instructions, hook_fn=hook_fn)
        return self.refusal_directions

    def aggregate_best_layers(self):
        for layer_candidate in self.refusal_directions:
            count = sum(
                sum(word not in example for word in self.negative_tokens) + sum(word in example for word in self.positive_tokens)
                for example in layer_candidate["intervention_generation"]
            )
            layer_candidate["count"] = count

        self.refusal_directions.sort(key=lambda x: x["count"], reverse=True)
        return self.refusal_directions

    def ablate_layer(self, direction: Dict = None, layers: List[int] = None, attn_output: bool = True, mlp: bool = True):
        refusal_direction = direction or self.refusal_directions[0]
        refusal_direction = refusal_direction["refusal_direction"].to(self.device)

        if attn_output or mlp:
            self.modified = True

        layers = layers or list(range(1, len(self.model.model.layers)))
        for layer in layers:
            block = self.model.model.layers[layer]
            if attn_output:
                block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data, refusal_direction)
                self.modified_layers["attention_output_layer"].append(layer)
            if mlp:
                block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data, refusal_direction)
                self.modified_layers["mlp"].append(layer)

    def push_to_hub(self):
        self.model.push_to_hub(f"{self.model_name.split('/',1)[-1]}-uncensored")
