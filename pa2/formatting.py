"""The single prompt format used for SFT, DPO, and evaluation.

We use the Qwen ChatML layout without a system message. The response text is
appended directly after the assistant header, and the batch builders append the
tokenizer EOS token (``<|endoftext|>`` for Qwen2.5) after the response. At
generation time we stop on either EOS or ``<|im_end|>``.
"""

from __future__ import annotations

PROMPT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


def format_prompt(instruction: str) -> str:
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())


def clean_response(text: str) -> str:
    """Strip whitespace and anything after a stop marker from generated text."""
    for stop in STOP_STRINGS:
        text = text.split(stop)[0]
    return text.strip()
