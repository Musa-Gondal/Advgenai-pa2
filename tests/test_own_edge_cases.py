"""Extra checks beyond the public smoke tests: truncation, ordering, numerics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import submission_adapter as adapter
from pa2.batching import IGNORE_INDEX
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoEosTokenizer(ToyTokenizer):
    eos_token_id = None


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


def manual_logprob(model, tok, prompt, response, max_length, add_eos=True):
    """Reference implementation: loop over response positions one at a time."""
    p = tok.encode(prompt)
    r = tok.encode(response) + ([tok.eos_token_id] if add_eos and tok.eos_token_id is not None else [])
    ids = (p + r)[:max_length]
    logits = model(input_ids=torch.tensor([ids])).logits[0].float()
    total = 0.0
    for pos in range(max(1, len(p)), len(ids)):
        total += F.log_softmax(logits[pos - 1], -1)[ids[pos]].item()
    return total


def test_pads_to_max_length_and_truncates():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["abc: ", "x"], ["hello world", "y"], max_length=8)
    assert batch["input_ids"].shape == (2, 8)
    # row 0: 5 prompt + 11 response + eos = 17 > 8 -> truncated, EOS dropped
    assert batch["attention_mask"][0].sum() == 8
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, :5].eq(IGNORE_INDEX).all()
    assert batch["labels"][0, 5:].ne(IGNORE_INDEX).all()
    # row 1: 1 + 1 + eos = 3 tokens then padding
    assert batch["attention_mask"][1].tolist() == [1, 1, 1, 0, 0, 0, 0, 0]
    assert batch["labels"][1].tolist()[:3] == [IGNORE_INDEX, tok.vocab["y"], tok.eos_token_id]


def test_eos_kept_exactly_at_boundary():
    tok = ToyTokenizer()
    # 3 prompt + 4 response + eos = 8 == max_length
    batch = adapter.build_sft_batch(tok, ["ab:"], ["wxyz"], max_length=8)
    assert batch["labels"][0, -1] == tok.eos_token_id
    batch = adapter.build_sft_batch(tok, ["ab:"], ["wxyz"], max_length=7)
    assert batch["labels"][0, -1] == tok.vocab["z"]


def test_prompt_longer_than_max_length_has_no_response_tokens():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    batch = adapter.build_sft_batch(tok, ["a" * 20], ["b"], max_length=10)
    assert batch["labels"].eq(IGNORE_INDEX).all()
    logps = adapter.compute_response_logprobs(model, tok, ["a" * 20], ["b"], max_length=10)
    assert logps.item() == 0.0
    loss = adapter.compute_sft_loss(model, tok, ["a" * 20], ["b"], max_length=10)
    assert torch.isfinite(loss)


def test_tokenizer_without_eos_or_pad():
    tok = NoEosTokenizer()
    batch = adapter.build_sft_batch(tok, ["q: "], ["ok"], max_length=8)
    assert batch["labels"][0].ne(IGNORE_INDEX).sum() == 2
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["q: ", "q"], ["ok", "fine"], max_length=8)
    assert batch["attention_mask"][0].sum() == 3 + 2 + 1


def test_logprobs_match_manual_reference_with_mixed_lengths_and_order():
    torch.manual_seed(0)
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    with torch.no_grad():
        model.logit_bias.copy_(torch.randn(tok.vocab_size))
    prompts = ["short: ", "a much longer prompt: ", "", "mid: "]
    responses = ["x", "two words", "no prompt", "truncate me please"]
    got = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=16)
    want = torch.tensor([manual_logprob(model, tok, p, r, 16) for p, r in zip(prompts, responses)])
    torch.testing.assert_close(got.detach(), want, rtol=1e-5, atol=1e-5)
    # reversing the input reverses the output
    rev = adapter.compute_response_logprobs(model, tok, prompts[::-1], responses[::-1], max_length=16)
    torch.testing.assert_close(rev.detach(), want.flip(0), rtol=1e-5, atol=1e-5)


def test_sft_loss_is_token_mean():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.4)
    prompts, responses = ["p: ", "q: "], ["a", "abcdef"]
    loss = adapter.compute_sft_loss(model, tok, prompts, responses, max_length=32)
    logps = adapter.compute_response_logprobs(model, tok, prompts, responses, max_length=32)
    n_tokens = (1 + 1) + (6 + 1)
    torch.testing.assert_close(loss, -logps.sum() / n_tokens)


def test_dpo_loss_from_logps_matches_formula():
    pc, pr = torch.tensor([-1.0, -3.0, -2.0]), torch.tensor([-2.0, -1.0, -2.0])
    rc, rr = torch.tensor([-1.5, -2.5, -1.0]), torch.tensor([-1.5, -1.5, -3.0])
    beta = 0.5
    expected = torch.stack(
        [-torch.log(torch.sigmoid(beta * ((a - b) - (c - d)))) for a, b, c, d in zip(pc, pr, rc, rr)]
    ).mean()
    got = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    torch.testing.assert_close(got, expected)
    # numerically stable for large gaps
    big = adapter.compute_dpo_loss_from_logps(
        torch.tensor([-1e4]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 1.0
    )
    assert math.isfinite(big.item()) and big.item() > 1e3


def test_dpo_batch_matches_sft_batch_per_side():
    tok = ToyTokenizer()
    prompts, chosen, rejected = ["a: ", "bb: "], ["good", "y"], ["bad answer", "n"]
    dpo = adapter.build_dpo_batch(tok, prompts, chosen, rejected, max_length=12)
    for side, responses in (("chosen", chosen), ("rejected", rejected)):
        sft = adapter.build_sft_batch(tok, prompts, responses, max_length=12)
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(dpo[side][key], sft[key])


def test_dpo_loss_matches_logps_path_and_identical_models_give_log2():
    torch.manual_seed(1)
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    with torch.no_grad():
        policy.logit_bias.copy_(torch.randn(tok.vocab_size))
    reference = ToyCausalLM(tok.vocab_size, bias_scale=1.0)
    reference.load_state_dict(policy.state_dict())
    args = (tok, ["p: ", "longer prompt: "], ["abc", "d"], ["xy", "efghij"])
    loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.1, max_length=16)
    torch.testing.assert_close(loss, torch.log(torch.tensor(2.0)))

    with torch.no_grad():
        reference.logit_bias.add_(torch.randn(tok.vocab_size))
    loss = adapter.compute_dpo_loss(policy, reference, *args, beta=0.3, max_length=16)
    pc = adapter.compute_response_logprobs(policy, tok, args[1], args[2], 16)
    pr = adapter.compute_response_logprobs(policy, tok, args[1], args[3], 16)
    rc = adapter.compute_response_logprobs(reference, tok, args[1], args[2], 16)
    rr = adapter.compute_response_logprobs(reference, tok, args[1], args[3], 16)
    expected = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.3)
    torch.testing.assert_close(loss, expected, rtol=1e-5, atol=1e-6)


def test_dpo_step_increases_chosen_preference():
    torch.manual_seed(2)
    tok = ToyTokenizer()
    policy = ToyCausalLM(tok.vocab_size)
    reference = ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(policy.parameters(), lr=1.0)
    args = (tok, ["q: "], ["aaa"], ["zzz"])
    before = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=10)
    for _ in range(5):
        opt.zero_grad()
        adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=10).backward()
        opt.step()
    after = adapter.compute_dpo_loss(policy, reference, *args, beta=0.5, max_length=10)
    assert after < before
    assert reference.logit_bias.abs().sum() == 0  # reference untouched


def test_sft_step_reduces_loss():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size)
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    args = (tok, ["say: "], ["aaaa"])
    before = adapter.compute_sft_loss(model, *args, max_length=12)
    for _ in range(5):
        opt.zero_grad()
        adapter.compute_sft_loss(model, *args, max_length=12).backward()
        opt.step()
    assert adapter.compute_sft_loss(model, *args, max_length=12) < before
