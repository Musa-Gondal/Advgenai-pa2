"""Response-only log-probabilities, SFT loss, and DPO loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from pa2.batching import IGNORE_INDEX, build_dpo_batch, build_sft_batch, concat_dpo_batch


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def token_logprobs(logits, labels):
    """Per-token log p(label) for shifted next-token prediction.

    Returns ``(logps, mask)`` of shape ``[B, T-1]``; masked positions hold 0.
    """
    logits = logits[:, :-1, :].float()
    labels = labels[:, 1:]
    mask = labels.ne(IGNORE_INDEX)
    nll = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view(labels.shape)
    return -nll * mask, mask


def batch_response_logprobs(model, batch):
    """Summed response-token log-probability per row, plus response token counts."""
    device = model_device(model)
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    logps, mask = token_logprobs(logits, labels)
    return logps.sum(dim=-1), mask.sum(dim=-1)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length, pad_to="longest")
    logps, _ = batch_response_logprobs(model, batch)
    return logps


def sft_loss_from_batch(model, batch):
    """Mean negative log-likelihood over all unmasked response tokens in the batch."""
    logps, counts = batch_response_logprobs(model, batch)
    return -logps.sum() / counts.sum().clamp_min(1)


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length, pad_to="longest")
    return sft_loss_from_batch(model, batch)


def dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    """Return ``(mean loss, chosen rewards, rejected rewards)``.

    loss_i = -log sigmoid(beta * ((pi_w - pi_l) - (ref_w - ref_l)))
    rewards: r = beta * (log pi - log ref), detached.
    """
    policy_gap = policy_chosen_logps - policy_rejected_logps
    reference_gap = reference_chosen_logps - reference_rejected_logps
    losses = -F.logsigmoid(beta * (policy_gap - reference_gap))
    chosen_rewards = (beta * (policy_chosen_logps - reference_chosen_logps)).detach()
    rejected_rewards = (beta * (policy_rejected_logps - reference_rejected_logps)).detach()
    return losses.mean(), chosen_rewards, rejected_rewards


def chosen_rejected_logprobs(model, dpo_batch):
    """Run chosen and rejected rows through ``model`` in a single forward pass."""
    batch = concat_dpo_batch(dpo_batch)
    logps, _ = batch_response_logprobs(model, batch)
    n = dpo_batch["chosen"]["input_ids"].shape[0]
    return logps[:n], logps[n:]


def dpo_loss_from_batch(policy_model, reference_model, dpo_batch, beta):
    """DPO loss and rewards for a pre-built batch; the reference runs without gradients."""
    policy_chosen, policy_rejected = chosen_rejected_logprobs(policy_model, dpo_batch)
    with torch.no_grad():
        ref_chosen, ref_rejected = chosen_rejected_logprobs(reference_model, dpo_batch)
    device = policy_chosen.device
    return dpo_loss_from_logps(
        policy_chosen,
        policy_rejected,
        ref_chosen.to(device=device, dtype=policy_chosen.dtype),
        ref_rejected.to(device=device, dtype=policy_chosen.dtype),
        beta,
    )


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    dpo_batch = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length, pad_to="longest"
    )
    loss, _, _ = dpo_loss_from_batch(policy_model, reference_model, dpo_batch, beta)
    return loss
