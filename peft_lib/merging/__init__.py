"""Adapter and model merging: LoRA folding, model soups, and TIES-merging."""

from __future__ import annotations

from peft_lib.merging.merge_lora import merge_and_unload, weighted_merge_adapters
from peft_lib.merging.model_soup import uniform_soup, weighted_soup
from peft_lib.merging.ties_merging import make_task_vector, ties_merge, ties_merge_into

__all__ = [
    "make_task_vector",
    "merge_and_unload",
    "ties_merge",
    "ties_merge_into",
    "uniform_soup",
    "weighted_merge_adapters",
    "weighted_soup",
]
