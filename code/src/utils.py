import os
import re
import string
from collections import Counter, defaultdict

import numpy as np
from datasets import load_dataset
from evaluate import load
from constants import HF_TOKEN

ANSWERABLE, UNANSWERABLE = "answerable", "unanswerable"


def build_balanced_eval_set(n_per_class=75, seed=42, max_per_context=None):
    """n_per_class answerable + n_per_class unanswerable, mixed and reproducible.

    max_per_context: optional cap on questions sharing the same passage, so
    retrieval recall isn't skewed by near-duplicate contexts.
    """
    ds = load_dataset("rajpurkar/squad_v2", split="validation", token=HF_TOKEN)
    ds = ds.shuffle(seed=seed)

    ans_idx, unans_idx = [], []
    for i, ex in enumerate(ds):
        (ans_idx if len(ex["answers"]["text"]) else unans_idx).append(i)

    def take(indices, n):
        seen, out = Counter(), []
        for i in indices:
            if max_per_context is not None:
                if seen[ds[i]["context"]] >= max_per_context:
                    continue
                seen[ds[i]["context"]] += 1
            out.append(i)
            if len(out) == n:
                break
        return out

    chosen = take(ans_idx, n_per_class) + take(unans_idx, n_per_class)
    return ds.select(chosen).shuffle(seed=seed)