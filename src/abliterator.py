from transformer_lens import HookedTransformer, utils
from transformer_lens.hook_points import HookPoint
from transformers import PreTrainedTokenizer
from typing import List, Callable, Union
from collections import defaultdict
from jaxtyping import Float, Int
from tqdm.auto import tqdm
from colorama import Fore
from pathlib import Path
from torch import Tensor

import pandas as pd
import functools
import textwrap
import requests
import einops
import torch
import io
import gc

torch.set_grad_enabled(False)


def get_input_ids(tokenizer: PreTrainedTokenizer, instructions: List[str]) -> Int[Tensor, "batch_size seq_len"]:
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def decode(tokenizer: PreTrainedTokenizer, tokens_batch: torch.Tensor):
    return tokenizer.batch_decode(tokens_batch, skip_special_tokens=True)


class Abliterator:
    def __init__(
        self,
        model_name: Union[str, Path],
        data: dict,
        device: Union[str, torch.device] = "cuda",
        n_inst_train: int = 256,
    ):
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

    def cache_activations(self):
        # Define batch size based on available VRAM
        batch_size = 32

        # Initialize defaultdicts to store activations
        harmful = defaultdict(list)
        harmless = defaultdict(list)

        # Process the training data in batches
        num_batches = (self.n_inst_train + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches)):
            print(i)
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
            gc.collect()
            torch.cuda.empty_cache()

        # Concatenate the cached activations
        harmful = {k: torch.cat(v) for k, v in harmful.items()}
        harmless = {k: torch.cat(v) for k, v in harmless.items()}
