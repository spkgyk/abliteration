import random

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

torch.inference_mode()

torch.set_default_device("cuda")

# MODEL_ID = "stabilityai/stablelm-2-1_6b"
# MODEL_ID = "stabilityai/stablelm-2-zephyr-1_6b"
# MODEL_ID = "Qwen/Qwen1.5-1.8B-Chat"
# MODEL_ID = "Qwen/Qwen-1_8B-chat"
MODEL_ID = "Qwen/Qwen1.5-4B-Chat"
# MODEL_ID = "google/gemma-1.1-7b-it"
# MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True, device_map="auto", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

instructions = 1000
layer_idx = int(len(model.model.layers) * 0.6)
pos = -1

print("Instruction count: " + str(instructions))
print("Layer index: " + str(layer_idx))

with open("harmful.txt", "r") as f:
    harmful = f.readlines()

with open("harmless.txt", "r") as f:
    harmless = f.readlines()

harmful_instructions = random.sample(harmful, len(harmful))
harmless_instructions = random.sample(harmless, instructions)

harmful_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True, return_tensors="pt")
    for insn in harmful_instructions
]
harmless_toks = [
    tokenizer.apply_chat_template(conversation=[{"role": "user", "content": insn}], add_generation_prompt=True, return_tensors="pt")
    for insn in harmless_instructions
]

max_its = len(harmful_toks) + len(harmless_toks)
bar = tqdm(total=max_its)


def generate(toks):
    bar.update(n=1)
    return (
        model.generate(toks.to(model.device), use_cache=False, max_new_tokens=1, return_dict_in_generate=True, output_hidden_states=True)
        .hidden_states[0][layer_idx][:, pos, :]
        .detach()
        .cpu()
    )


harmful_outputs = [generate(toks) for toks in harmful_toks]
harmless_outputs = [generate(toks) for toks in harmless_toks]

bar.close()


print(harmful_outputs)

harmful_mean = torch.stack(harmful_outputs).mean(dim=0)
harmless_mean = torch.stack(harmless_outputs).mean(dim=0)

print(harmful_mean)

refusal_dir = harmful_mean - harmless_mean
refusal_dir = refusal_dir / refusal_dir.norm()

print(refusal_dir)

torch.save(refusal_dir, MODEL_ID.replace("/", "_") + "_refusal_dir.pt")
