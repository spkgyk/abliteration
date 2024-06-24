from transformer_lens import HookedTransformerKeyValueCache, HookedTransformerKeyValueCacheEntry
from typing import Iterable

import torch
import gc


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()


def subset_cache(cache: HookedTransformerKeyValueCache, generating: Iterable[int]):
    entries = [
        HookedTransformerKeyValueCacheEntry(
            entry.past_keys[generating],
            entry.past_values[generating],
            entry.frozen,
        )
        for entry in cache.entries
    ]
    generating_cache = HookedTransformerKeyValueCache(entries, cache.previous_attention_mask[generating], cache.frozen)

    return generating_cache
