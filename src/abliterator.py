from transformers import AutoModelForCausalLM, PreTrainedTokenizer, LlamaForCausalLM, GenerationConfig, AutoTokenizer
from typing import Union, List, Dict, Callable
from collections import defaultdict
from tqdm.auto import tqdm
from pathlib import Path
import torch

from .hooks import direction_ablation_hook, get_orthogonalized_matrix, get_activation_hook
from .data import HarmfulHarmlessData
from .utils import clear_mem

torch.set_grad_enabled(False)
torch.inference_mode()


class Abliterator:
    """
    A class for performing targeted ablation on language models to modify their behavior.

    This class implements methods to identify and modify directions in the model's weight
    space that correspond to specific behaviors (e.g., harmful vs harmless responses).
    It uses activation pattern analysis and targeted intervention through matrix
    orthogonalization.

    Attributes:
        model_name (Union[str, Path]): Name or path of the pre-trained model to load
        batch_size (int): Batch size for processing inputs
        max_tokens_generated (int): Maximum number of tokens to generate in responses
        device (Union[str, torch.device]): Device to run the model on
        positive_tokens (List[str]): Tokens indicating positive/desired responses
        negative_tokens (List[str]): Tokens indicating negative/undesired responses
        residual_stream_points (List): Points in the residual stream to analyze
        modified (bool): Whether the model has been modified through ablation
        modified_layers (defaultdict): Tracks which layers have been modified
        refusal_directions (List): Computed directions for response modification
    """

    def __init__(
        self,
        model_name: Union[str, Path],
        batch_size: int = 16,
        max_tokens_generated: int = 24,
        device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
        positive_tokens: List[str] = ["Sure", "To", "Certainly", "Here are", "I can"],
        negative_tokens: List[str] = [
            "I cannot",
            "I can't",
            "I can’t",
            "I'm sorry",
            "I’m sorry",
            "I don't",
            "I don’t",
            "Sorry",
        ],
        residual_stream_points=[],
    ):
        """
        Initialize the Abliterator with model configuration and analysis parameters.

        Args:
            model_name: Name or path of the pre-trained model
            batch_size: Number of samples to process at once
            max_tokens_generated: Maximum length of generated responses
            device: Computing device to use
            positive_tokens: List of tokens indicating desired responses
            negative_tokens: List of tokens indicating undesired responses
            residual_stream_points: Points in model to collect residual stream activations
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_tokens_generated = max_tokens_generated
        self.device = torch.device(device)
        self.positive_tokens = positive_tokens
        self.negative_tokens = negative_tokens
        self.residual_stream_points = residual_stream_points
        self.modified = False
        self.modified_layers = defaultdict(list)
        self.refusal_directions = []

        self.model = self._load_model()
        self.tokenizer = self._load_tokenizer()
        self.generation_config = GenerationConfig(do_sample=False, num_beams=1, pad_token_id=self.tokenizer.pad_token_id)

        self._print_model_layers()

    def _load_model(self) -> LlamaForCausalLM:
        """
        Load the pre-trained language model.

        Returns:
            LlamaForCausalLM: The loaded model instance
        """
        return AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """
        Load and configure the tokenizer for the model.

        Returns:
            PreTrainedTokenizer: The configured tokenizer instance
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True, add_bos_token=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _print_model_layers(self):
        """Print the structure of each layer in the model for inspection."""
        for layer_idx, layer in enumerate(self.model.model.layers):
            print(f"Layer {layer_idx}:")
            print(layer)
            print("---")

    def encode_tokens(self, instructions: List[str]) -> torch.Tensor:
        """
        Encode a list of instructions into model input tokens.

        Args:
            instructions: List of text instructions to encode

        Returns:
            torch.Tensor: Encoded token ids
        """
        tokens = self.tokenizer.apply_chat_template(
            instructions, padding=True, truncation=False, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )

        return tokens.input_ids.to(self.device)

    def decode_tokens(self, tokens_batch: torch.Tensor) -> List[str]:
        """
        Decode a batch of tokens back into text.

        Args:
            tokens_batch: Batch of token ids to decode

        Returns:
            List[str]: Decoded text strings
        """
        return self.tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)

    def _register_hooks(self, activations: Dict[str, List[torch.Tensor]], position: int) -> List[Callable]:
        """
        Register forward hooks to collect activations from model layers.

        Focuses on the top 70% of layers, collecting activations from pre-attention
        and pre-MLP points in the residual stream.

        Args:
            activations: Dictionary to store collected activations
            position: Position in sequence to collect activations from

        Returns:
            List[Callable]: List of registered hooks
        """
        useful_layers = max(int(0.3 * len(self.model.model.layers)), 1)
        hooks = []

        for layer_idx, layer in enumerate(self.model.model.layers[useful_layers:], start=useful_layers):
            layer_name = f"layer.{layer_idx}"

            pre_hook = get_activation_hook(activations, f"{layer_name}.pre_attn_stream", position, pre=True)
            resid_post_hook = get_activation_hook(activations, f"{layer_name}.pre_mlp_stream", position, pre=True)

            # Collect the residual stream before attention AKA input to input_layernorm.
            hooks.append(layer.input_layernorm.register_forward_pre_hook(pre_hook))
            # Collect the residual stream after attention, before mlp AKA input to post_attention_layernorm.
            hooks.append(layer.post_attention_layernorm.register_forward_pre_hook(resid_post_hook))

        return hooks

    def cache_activations(self, data: HarmfulHarmlessData, position: int = -1, eps: float = 1e-8):
        """
        Cache and analyze activations from harmful and harmless examples.

        Processes both harmful and harmless datasets to compute refusal directions
        that distinguish between these response types.

        Args:
            data: Dataset containing harmful and harmless examples
            position: Position in sequence to analyze (-1 for last token)
            eps: Small value to prevent division by zero
        """
        harmful = self._process_dataset(data.harmful["train"], "harmful", position)
        harmless = self._process_dataset(data.harmless["train"], "harmless", position)

        self.cache_keys = list(harmful.keys())
        self.refusal_directions = self._calculate_refusal_directions(harmful, harmless, eps)

        del harmful, harmless
        clear_mem()

    def _process_dataset(self, dataset: List[str], dataset_name: str, position: int) -> Dict[str, torch.Tensor]:
        """
        Process a dataset to collect activations.

        Args:
            dataset: List of text examples to process
            dataset_name: Name of dataset for progress tracking
            position: Position in sequence to collect activations from

        Returns:
            Dict[str, torch.Tensor]: Collected activations
        """
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
        """
        Calculate directions that distinguish harmful from harmless responses.

        Args:
            harmful: Activations from harmful examples
            harmless: Activations from harmless examples
            eps: Small value to prevent division by zero

        Returns:
            List[Dict]: Computed refusal directions for each activation point
        """
        return [
            {"cache_key": key, "refusal_direction": (harmful[key] - harmless[key]) / ((harmful[key] - harmless[key]).norm() + eps)}
            for key in self.cache_keys
        ]

    def generate(
        self,
        instructions: List[str],
        max_tokens_generated: int = None,
        hook_fn: Callable = None,
        pbar: tqdm = None,
        key: str = "",
    ) -> List[str]:
        """
        Generate responses to a list of instructions.

        Args:
            instructions: List of input prompts
            max_tokens_generated: Maximum length of generated responses
            hook_fn: Optional hook function to modify generation behavior
            pbar: Optional progress bar
            key: Optional key for progress tracking

        Returns:
            List[str]: Generated responses
        """
        generations = []
        hooks = []

        if hook_fn:
            hooks.append(self.model.model.embed_tokens.register_forward_hook(hook_fn))
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

            if pbar:
                pbar.set_description(f"Generating while abliterating {key}")
            elif key:
                print(f"Generating while abliterating {key}", flush=True)

        for hook in hooks:
            hook.remove()

        return generations

    def test_refusal_directions(self, instructions: List[str]):
        """
        Test the effectiveness of computed refusal directions.

        Generates responses while applying each refusal direction to evaluate
        their impact on model outputs.

        Args:
            instructions: List of test prompts

        Returns:
            List[Dict]: Results of testing each refusal direction
        """
        pbar = tqdm(self.refusal_directions)
        for refusal_direction in pbar:
            args = {
                "instructions": instructions,
                "hook_fn": direction_ablation_hook(refusal_direction["refusal_direction"]),
                "pbar": pbar,
                "key": refusal_direction["cache_key"],
            }
            refusal_direction["intervention_generation"] = self.generate(**args)
        return self.refusal_directions

    def aggregate_best_layers(self):
        """
        Analyze and rank refusal directions by effectiveness.

        Counts occurrences of positive tokens and absences of negative tokens
        to determine which directions best achieve desired response patterns.

        Returns:
            List[Dict]: Sorted refusal directions with effectiveness scores
        """
        for layer_candidate in self.refusal_directions:
            count = sum(
                sum(word not in example for word in self.negative_tokens) + sum(word in example for word in self.positive_tokens)
                for example in layer_candidate["intervention_generation"]
            )
            layer_candidate["count"] = count

        self.refusal_directions.sort(key=lambda x: x["count"], reverse=True)
        return self.refusal_directions

    def ablate_layer(
        self,
        direction: Dict = None,
        layers: List[int] = None,
        emb_out: bool = True,
        attn_out: bool = True,
        mlp_out: bool = True,
    ):
        """
        Apply ablation to specified model components using refusal directions.

        Modifies model weights by orthogonalizing them with respect to the
        computed refusal directions, potentially changing the model's behavior.

        Args:
            direction: Specific refusal direction to use (uses best if None)
            layers: List of layer indices to modify (defaults to all but first)
            emb_out: Whether to modify embedding output
            attn_out: Whether to modify attention output
            mlp_out: Whether to modify MLP output
        """
        refusal_direction = direction or self.refusal_directions[0]
        refusal_direction = refusal_direction["refusal_direction"].to(self.device)

        if emb_out or attn_out or mlp_out:
            self.modified = True

        layers = layers or list(range(1, len(self.model.model.layers)))

        if emb_out:
            self.model.model.embed_tokens.weight.data = get_orthogonalized_matrix(
                self.model.model.embed_tokens.weight.data, refusal_direction
            )
            self.modified_layers["emb_out"] = True

        for layer in layers:
            block = self.model.model.layers[layer]
            if attn_out:
                block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data, refusal_direction)
                self.modified_layers[layer].append("attn_out")
            if mlp_out:
                block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data, refusal_direction)
                self.modified_layers[layer].append("mlp_out")

    def push_to_hub(self):
        """
        Push the modified model to the Hugging Face model hub.

        The uploaded model will have '-uncensored' appended to its name.
        """
        self.model.push_to_hub(f"{self.model_name.split('/',1)[-1]}-uncensored")
