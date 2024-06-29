from transformer_lens import HookedTransformerKeyValueCache, HookedTransformerKeyValueCacheEntry, HookedTransformerConfig
from transformers import PreTrainedTokenizer
from typing import Iterable, Union
from jaxtyping import Bool, Int

import torch
import gc


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def get_input_ids(
    tokenizer: PreTrainedTokenizer,
    instructions: Iterable[str],
    device: torch.device = None,
) -> Int[torch.Tensor, "batch_size seq_len"]:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids.to(device)


class EnhancedHookedTransformerKeyValueCache(HookedTransformerKeyValueCache):
    def subset(self, mask: Bool[torch.Tensor, "batch"]) -> "EnhancedHookedTransformerKeyValueCache":
        """
        Create a subset of the cache based on a boolean mask.

        Args:
            mask (torch.Tensor): A boolean tensor of shape [batch] indicating which
                                 batch elements to keep.

        Returns:
            EnhancedHookedTransformerKeyValueCache: A new cache object containing only
                                                    the selected batch elements.
        """
        entries = [
            HookedTransformerKeyValueCacheEntry(
                entry.past_keys[mask],
                entry.past_values[mask],
                entry.frozen,
            )
            for entry in self.entries
        ]

        return EnhancedHookedTransformerKeyValueCache(entries, self.previous_attention_mask[mask], frozen=self.frozen)

    @classmethod
    def init_cache(cls, cfg: HookedTransformerConfig, device: Union[torch.device, str, None], batch_size: int = 1):
        # Override the init_cache method to return an instance of EnhancedHookedTransformerKeyValueCache
        base_cache = super().init_cache(cfg, device, batch_size)
        return cls(base_cache.entries, base_cache.previous_attention_mask, base_cache.frozen)
