"""Tokenization and batching for SFT and DPO.

Each row is ``prompt_ids + response_ids + [eos]``. EOS is appended before
truncation, so a response cut at ``max_length`` loses its EOS. Labels copy the
input ids on response (and EOS) positions and are -100 on prompt and padding
positions. Padding is on the right, so position 0 is always the first prompt
token and row order is preserved.
"""

from __future__ import annotations

import torch

IGNORE_INDEX = -100


def _encode(tokenizer, text):
    return list(tokenizer.encode(text, add_special_tokens=False))


def _pad_id(tokenizer):
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = getattr(tokenizer, "eos_token_id", None)
    return 0 if pad_id is None else pad_id


def tokenize_pair(tokenizer, prompt, response, max_length):
    """Return (input_ids, labels) for one prompt/response row, truncated to max_length."""
    prompt_ids = _encode(tokenizer, prompt)
    response_ids = _encode(tokenizer, response)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        response_ids = response_ids + [eos_id]
    input_ids = (prompt_ids + response_ids)[:max_length]
    labels = ([IGNORE_INDEX] * len(prompt_ids) + response_ids)[:max_length]
    return input_ids, labels


def collate(rows, pad_id, width):
    """Right-pad (input_ids, labels) rows to ``width`` and stack into tensors."""
    input_ids, attention_mask, labels = [], [], []
    for ids, labs in rows:
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        labels.append(labs + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def _width(rows, max_length, pad_to):
    if pad_to == "max_length":
        return max_length
    if pad_to == "longest":
        return max(1, max(len(ids) for ids, _ in rows))
    raise ValueError(f"pad_to must be 'max_length' or 'longest', got {pad_to!r}")


def build_sft_batch(tokenizer, prompts, responses, max_length, pad_to="max_length"):
    """Return ``input_ids``, ``attention_mask`` and response-only ``labels``.

    ``pad_to="max_length"`` pads every row to ``max_length`` (the adapter
    contract); training uses ``pad_to="longest"`` to skip wasted padding.
    """
    if len(prompts) != len(responses):
        raise ValueError("prompts and responses must have the same length")
    rows = [tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, responses)]
    return collate(rows, _pad_id(tokenizer), _width(rows, max_length, pad_to))


def build_dpo_batch(
    tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to="max_length"
):
    """Return ``{"chosen": batch, "rejected": batch}`` with matching row order."""
    if not (len(prompts) == len(chosen_responses) == len(rejected_responses)):
        raise ValueError("prompts, chosen_responses and rejected_responses must align")
    chosen = [tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, chosen_responses)]
    rejected = [
        tokenize_pair(tokenizer, p, r, max_length) for p, r in zip(prompts, rejected_responses)
    ]
    pad_id = _pad_id(tokenizer)
    return {
        "chosen": collate(chosen, pad_id, _width(chosen, max_length, pad_to)),
        "rejected": collate(rejected, pad_id, _width(rejected, max_length, pad_to)),
    }


def concat_dpo_batch(dpo_batch):
    """Stack chosen rows then rejected rows into one batch (padded to a common width)."""
    chosen, rejected = dpo_batch["chosen"], dpo_batch["rejected"]
    width = max(chosen["input_ids"].shape[1], rejected["input_ids"].shape[1])
    pad_values = {"input_ids": 0, "attention_mask": 0, "labels": IGNORE_INDEX}
    out = {}
    for key, pad_value in pad_values.items():
        parts = []
        for side in (chosen, rejected):
            tensor = side[key]
            extra = width - tensor.shape[1]
            if extra:
                tensor = torch.nn.functional.pad(tensor, (0, extra), value=pad_value)
            parts.append(tensor)
        out[key] = torch.cat(parts, dim=0)
    return out
