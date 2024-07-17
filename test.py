import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model_name = "Qwen/Qwen1.5-4B-Chat"  # Replace with your actual model name
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"

# Determine the device
device = next(model.parameters()).device

# Prepare the input
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Dictionary to store activations
activations = {}


# Hook function to save activations
def save_activation(name):
    def hook(module, input, output):
        print(f"Hook called for: {name}")
        activations[name] = output.detach().cpu()

    return hook


def save_pre_activation(name):
    def hook(module, input):
        print(f"Pre-hook called for: {name}")
        activations[name] = input[0].detach().cpu()

    return hook


# Register hooks
for layer_idx, layer in enumerate(model.model.layers):
    # Print the layer being hooked
    print(f"Registering hooks for layer {layer_idx}")

    # Hook for hook_resid_pre (input to the block)
    layer.input_layernorm.register_forward_pre_hook(save_pre_activation(f"blocks.{layer_idx}.hook_resid_pre"))

    # Hook for hook_attn_out (output of the attention mechanism)
    layer.self_attn.o_proj.register_forward_hook(save_activation(f"blocks.{layer_idx}.hook_attn_out"))

    # Hook for hook_mlp_out (output of the MLP)
    layer.mlp.down_proj.register_forward_hook(save_activation(f"blocks.{layer_idx}.hook_mlp_out"))

    # Hook for hook_resid_post (final output of the block)
    layer.post_attention_layernorm.register_forward_hook(save_activation(f"blocks.{layer_idx}.hook_resid_post"))

# Move model and inputs to device if available
model.to(device)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Forward pass to get activations
with torch.no_grad():
    outputs = model(**inputs)

# Print the activations
for name, activation in activations.items():
    print(f"{name}: shape {activation.shape}")
