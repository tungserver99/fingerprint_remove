from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch


TARGET_LINEAR_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class RTNState:
    q_int: torch.Tensor
    dequant_weight: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    pre_round_code: torch.Tensor
    qmin: int
    qmax: int


def iter_target_linear_modules(model: torch.nn.Module) -> Iterable[Tuple[str, torch.nn.Linear]]:
    """Yield RTN/CCDR target linears and intentionally skip lm_head."""
    for name, module in model.named_modules():
        if name == "lm_head" or name.endswith(".lm_head"):
            continue
        if isinstance(module, torch.nn.Linear) and name.rsplit(".", 1)[-1] in TARGET_LINEAR_SUFFIXES:
            yield name, module


@torch.no_grad()
def rtn4_with_state(weight: torch.Tensor, group_size: int = 128, bits: int = 4) -> RTNState:
    if weight.ndim != 2:
        raise ValueError(f"Expected a 2D linear weight, got shape {tuple(weight.shape)}")
    if bits != 4:
        raise ValueError("CCDR pilot supports bits=4 only")

    W = weight.detach().float()
    out_features, in_features = W.shape
    qmin, qmax = 0, (1 << bits) - 1
    group_count = (in_features + group_size - 1) // group_size

    q_int = torch.empty_like(W, dtype=torch.int64)
    dequant = torch.empty_like(W)
    pre_round = torch.empty_like(W)
    scale = torch.empty((out_features, group_count, 1), dtype=W.dtype, device=W.device)
    zero_point = torch.empty((out_features, group_count, 1), dtype=W.dtype, device=W.device)

    for group_idx, start in enumerate(range(0, in_features, group_size)):
        end = min(start + group_size, in_features)
        block = W[:, start:end]
        w_min = block.amin(dim=1, keepdim=True)
        w_max = block.amax(dim=1, keepdim=True)
        block_scale = (w_max - w_min) / float(qmax - qmin)
        block_scale = torch.where(block_scale == 0, torch.ones_like(block_scale), block_scale)
        block_zp = torch.round(qmin - w_min / block_scale).clamp(qmin, qmax)
        block_pre_round = block / block_scale + block_zp
        block_q = torch.round(block_pre_round).clamp(qmin, qmax).to(torch.int64)
        block_dequant = (block_q.float() - block_zp) * block_scale

        scale[:, group_idx, :] = block_scale
        zero_point[:, group_idx, :] = block_zp
        pre_round[:, start:end] = block_pre_round
        q_int[:, start:end] = block_q
        dequant[:, start:end] = block_dequant

    return RTNState(
        q_int=q_int,
        dequant_weight=dequant.to(dtype=weight.dtype),
        scale=scale,
        zero_point=zero_point,
        pre_round_code=pre_round,
        qmin=qmin,
        qmax=qmax,
    )


def build_opposite_codes(
    pre_round_code: torch.Tensor,
    q_rtn_int: torch.Tensor,
    qmin: int,
    qmax: int,
) -> torch.Tensor:
    u = pre_round_code.float()
    q_lo = torch.floor(u)
    q_hi = torch.ceil(u)
    q_rtn = q_rtn_int.to(u.dtype)
    q_alt = torch.where(q_rtn == q_lo, q_hi, q_lo)
    valid = (q_alt >= qmin) & (q_alt <= qmax) & (q_lo != q_hi)
    q_alt = torch.where(valid, q_alt, q_rtn)
    return q_alt.to(q_rtn_int.dtype)


def compute_channel_energy(X: torch.Tensor) -> torch.Tensor:
    if X.ndim != 2:
        X = X.reshape(-1, X.shape[-1])
    return X.float().pow(2).mean(dim=0)


def rank_channels_per_group(energy: torch.Tensor, group_size: int = 128) -> List[torch.Tensor]:
    rankings: List[torch.Tensor] = []
    flat_energy = energy.detach().flatten()
    for start in range(0, flat_energy.numel(), group_size):
        end = min(start + group_size, flat_energy.numel())
        rankings.append(torch.argsort(flat_energy[start:end], descending=False))
    return rankings


def _dequantize_codes(
    q_int: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    out_features, in_features = q_int.shape
    dequant = torch.empty(q_int.shape, dtype=scale.dtype, device=q_int.device)
    for group_idx, start in enumerate(range(0, in_features, group_size)):
        end = min(start + group_size, in_features)
        dequant[:, start:end] = (q_int[:, start:end].float() - zero_point[:, group_idx, :]) * scale[:, group_idx, :]
    return dequant


def build_candidate(
    q_rtn_int: torch.Tensor,
    q_alt_int: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    group_rankings: Sequence[torch.Tensor],
    k: int,
    group_size: int = 128,
) -> torch.Tensor:
    q = q_rtn_int.clone()
    if k > 0:
        for group_idx, order in enumerate(group_rankings):
            start = group_idx * group_size
            end = min(start + group_size, q.shape[1])
            selected = order[: min(k, end - start)].to(device=q.device) + start
            q[:, selected] = q_alt_int[:, selected]
    return _dequantize_codes(q, scale, zero_point, group_size)


def _mse_output(X: torch.Tensor, W_a: torch.Tensor, W_b: torch.Tensor) -> torch.Tensor:
    y_a = torch.nn.functional.linear(X.float(), W_a.float())
    y_b = torch.nn.functional.linear(X.float(), W_b.float())
    return (y_a - y_b).float().pow(2).mean()


@torch.no_grad()
def choose_ccdr_candidate(
    W: torch.Tensor,
    X: torch.Tensor,
    rtn_state: RTNState,
    k_values: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
    epsilon: float = 0.05,
    group_size: int = 128,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not k_values:
        raise ValueError("k_values must contain at least one candidate")

    X2 = X.reshape(-1, X.shape[-1]).float()
    Wf = W.detach().float()
    q_alt = build_opposite_codes(
        rtn_state.pre_round_code,
        rtn_state.q_int,
        rtn_state.qmin,
        rtn_state.qmax,
    )

    q0 = rtn_state.dequant_weight.float()
    e0 = _mse_output(X2, q0, Wf)
    d0 = (q0 - Wf).pow(2).mean()
    threshold = (1.0 + epsilon) * e0

    best_q = q0
    best_k = 0
    best_e = e0
    best_d = d0

    rankings = rank_channels_per_group(compute_channel_energy(X2), group_size=group_size)
    for k in k_values:
        if k == 0:
            qk = q0
        else:
            qk = build_candidate(
                rtn_state.q_int,
                q_alt,
                rtn_state.scale,
                rtn_state.zero_point,
                rankings,
                k,
                group_size=group_size,
            ).float()
        ek = _mse_output(X2, qk, Wf)
        dk = (qk - Wf).pow(2).mean()
        if bool(ek <= threshold and dk > best_d):
            best_q = qk
            best_k = int(k)
            best_e = ek
            best_d = dk

    rtn_error = float(e0.item())
    selected_error = float(best_e.item())
    rtn_drift = float(d0.item())
    selected_drift = float(best_d.item())
    stats = {
        "selected_k": best_k,
        "selected_flip_fraction": float(best_k) / float(group_size),
        "rtn_reconstruction_mse": rtn_error,
        "ccdr_reconstruction_mse": selected_error,
        "reconstruction_ratio": selected_error / rtn_error if rtn_error else 1.0,
        "rtn_weight_mse": rtn_drift,
        "ccdr_weight_mse": selected_drift,
        "weight_drift_ratio": selected_drift / rtn_drift if rtn_drift else 1.0,
    }
    return best_q.to(dtype=W.dtype, device=W.device), stats
