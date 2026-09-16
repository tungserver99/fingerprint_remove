from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibration_data import get_loaders
from ccdr import (
    build_candidate,
    build_opposite_codes,
    iter_target_linear_modules,
    rank_channels_per_group,
    rtn4_with_state,
)


@dataclass
class ActivationChunkStore:
    module_name: str
    directory: Path
    paths: List[Path]
    num_tokens: int
    in_features: int


def _load_c4_calibration(args):
    return get_loaders(
        args.calib_data,
        nsamples=args.calib_samples,
        seed=args.seed,
        seqlen=args.calib_seqlen,
        eval_mode=False,
        model_path=args.model_path,
        use_fast_tokenizer=True,
        trust_remote_code=True,
    )


def _module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _block_id(module_name: str) -> str:
    parts = module_name.split(".")
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            return parts[idx + 1]
    return ""


def _safe_module_name(module_name: str) -> str:
    return module_name.replace(".", "__").replace("/", "_")


def effective_calib_tokens(args) -> int:
    if args.calib_tokens is not None:
        return int(args.calib_tokens)
    return int(args.calib_samples) * int(args.calib_seqlen)


@torch.no_grad()
def collect_activation_chunks_for_module(
    model: torch.nn.Module,
    module_name: str,
    module: torch.nn.Module,
    calibration_batches,
    max_tokens: int,
    cache_root: Path,
) -> ActivationChunkStore:
    module_dir = cache_root / _safe_module_name(module_name)
    if module_dir.exists():
        shutil.rmtree(module_dir)
    module_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    token_count = 0
    in_features = None

    def hook(_module, inputs, _output):
        nonlocal token_count, in_features
        if token_count >= max_tokens:
            return
        x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).to("cpu")
        need = max_tokens - token_count
        if x.shape[0] > need:
            x = x[:need]
        if in_features is None:
            in_features = int(x.shape[-1])
        path = module_dir / f"chunk_{len(paths):05d}.pt"
        torch.save(x.contiguous(), path)
        paths.append(path)
        token_count += int(x.shape[0])

    handle = module.register_forward_hook(hook)
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    device = next(model.parameters()).device
    try:
        for batch in tqdm(calibration_batches, desc=f"Collecting activations: {module_name}", leave=False):
            model(batch.to(device))
            if token_count >= max_tokens:
                break
    finally:
        handle.remove()
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache

    if not paths or in_features is None:
        raise RuntimeError(f"No calibration activations captured for {module_name}")
    if token_count < max_tokens:
        raise RuntimeError(
            f"Only captured {token_count} activation tokens for {module_name}; expected {max_tokens}. "
            "Increase calibration data or lower --calib-tokens."
        )
    return ActivationChunkStore(
        module_name=module_name,
        directory=module_dir,
        paths=paths,
        num_tokens=token_count,
        in_features=in_features,
    )


def _iter_activation_chunks(store: ActivationChunkStore) -> Iterable[torch.Tensor]:
    for path in store.paths:
        yield torch.load(path, map_location="cpu", weights_only=True)


@torch.no_grad()
def _streaming_channel_energy(store: ActivationChunkStore) -> torch.Tensor:
    sumsq = torch.zeros(store.in_features, dtype=torch.float64)
    total = 0
    for x in _iter_activation_chunks(store):
        x_float = x.float()
        sumsq += x_float.pow(2).sum(dim=0).double()
        total += int(x_float.shape[0])
    if total == 0:
        raise RuntimeError(f"No activation tokens found for {store.module_name}")
    return (sumsq / float(total)).float()


@torch.no_grad()
def _streaming_reconstruction_mse(
    store: ActivationChunkStore,
    W: torch.Tensor,
    Q: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    total_sse = torch.zeros((), dtype=torch.float64, device="cpu")
    total_values = 0
    Wf = W.detach().float().to(device)
    Qf = Q.detach().float().to(device)
    for x_cpu in _iter_activation_chunks(store):
        x = x_cpu.to(device).float()
        y_fp = torch.nn.functional.linear(x, Wf)
        y_q = torch.nn.functional.linear(x, Qf)
        total_sse += (y_q - y_fp).float().pow(2).sum().double().cpu()
        total_values += int(y_q.numel())
        del x, y_fp, y_q
    return (total_sse / float(total_values)).float()


@torch.no_grad()
def choose_ccdr_candidate_from_chunks(
    W: torch.Tensor,
    store: ActivationChunkStore,
    rtn_state,
    k_values: Sequence[int] = (0, 1, 2, 4, 8, 16, 32),
    epsilon: float = 0.05,
    group_size: int = 128,
) -> Tuple[torch.Tensor, dict]:
    if not k_values:
        raise ValueError("k_values must contain at least one candidate")

    device = W.device
    Wf = W.detach().float()
    q_alt = build_opposite_codes(
        rtn_state.pre_round_code,
        rtn_state.q_int,
        rtn_state.qmin,
        rtn_state.qmax,
    )
    q0 = rtn_state.dequant_weight.float()
    e0 = _streaming_reconstruction_mse(store, Wf, q0, device)
    d0 = (q0 - Wf).pow(2).mean()
    threshold = (1.0 + epsilon) * e0

    best_q = q0
    best_k = 0
    best_e = e0
    best_d = d0

    energy = _streaming_channel_energy(store)
    rankings = rank_channels_per_group(energy, group_size=group_size)
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
        ek = _streaming_reconstruction_mse(store, Wf, qk, device)
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
        "calibration_tokens_used": int(store.num_tokens),
    }
    return best_q.to(dtype=W.dtype, device=W.device), stats


def _write_stats_csv(path: Path, rows: List[dict]):
    fieldnames = [
        "tensor_name",
        "block_id",
        "module_type",
        "in_features",
        "out_features",
        "selected_k",
        "selected_flip_fraction",
        "rtn_reconstruction_mse",
        "ccdr_reconstruction_mse",
        "reconstruction_ratio",
        "rtn_weight_mse",
        "ccdr_weight_mse",
        "weight_drift_ratio",
        "calibration_tokens_used",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ccdr(args):
    if args.bits != 4:
        raise ValueError("CCDR pilot supports bits=4 only")

    max_tokens = effective_calib_tokens(args)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    targets = list(iter_target_linear_modules(model))
    if not targets:
        raise RuntimeError("No CCDR target modules found. Check model architecture names.")
    if any(name == "lm_head" or name.endswith(".lm_head") for name, _ in targets):
        raise RuntimeError("Refusing to quantize lm_head")

    calibration_batches = _load_c4_calibration(args)
    output_dir = Path(args.output_dir)
    cache_root = output_dir / "activation_chunks"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    try:
        for name, linear in tqdm(targets, desc="Applying streaming CCDR"):
            W = linear.weight.data
            store = collect_activation_chunks_for_module(
                model=model,
                module_name=name,
                module=linear,
                calibration_batches=calibration_batches,
                max_tokens=max_tokens,
                cache_root=cache_root,
            )
            state = rtn4_with_state(W, group_size=args.group_size, bits=args.bits)
            selected_weight, stats = choose_ccdr_candidate_from_chunks(
                W=W,
                store=store,
                rtn_state=state,
                k_values=args.k_values,
                epsilon=args.epsilon,
                group_size=args.group_size,
            )
            linear.weight.data.copy_(selected_weight)
            stats_rows.append(
                {
                    "tensor_name": name,
                    "block_id": _block_id(name),
                    "module_type": _module_type(name),
                    "in_features": int(W.shape[1]),
                    "out_features": int(W.shape[0]),
                    **stats,
                }
            )

            shutil.rmtree(store.directory, ignore_errors=True)
            del store, state, selected_weight
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _write_stats_csv(output_dir / "results" / "ccdr_layer_stats.csv", stats_rows)
    config = vars(args).copy()
    config["effective_calib_tokens"] = max_tokens
    with (output_dir / "ccdr_config.json").open("w") as fh:
        json.dump(config, fh, indent=2)

    print(f"Wrote CCDR model to {output_dir}")
    print(f"Wrote layer stats to {output_dir / 'results' / 'ccdr_layer_stats.csv'}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Streaming CCDR INT4 pilot quantization")
    parser.add_argument("--model-path", default="cnut1648/LLaMA2-7B-fingerprinted-SFT")
    parser.add_argument("--output-dir", default="outputs/ccdr_if")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--calib-data", default="c4")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument(
        "--calib-tokens",
        type=int,
        default=None,
        help="Activation positions per module. Default: calib-samples * calib-seqlen.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_ccdr(parse_args())

