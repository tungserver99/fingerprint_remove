from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ccdr import choose_ccdr_candidate, iter_target_linear_modules, rtn4_with_state


DEFAULT_CALIB_ROOT = (
    r"G:\Other computers\My Computer\Google Drive Offline\PHD\Quantization"
    r"\Non_uniform_aaai\newloss-lnq"
)


def _import_newloss_loader(calib_code_root: str):
    root = Path(calib_code_root)
    if not root.exists():
        raise FileNotFoundError(f"Calibration code root does not exist: {root}")
    sys.path.insert(0, str(root))
    from any_precision.quantization.full_datautils import get_loaders

    return get_loaders


def _load_c4_calibration(args, tokenizer):
    get_loaders = _import_newloss_loader(args.calib_code_root)
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


@torch.no_grad()
def collect_activation_inputs(
    model: torch.nn.Module,
    calibration_batches,
    target_names: List[str],
    max_tokens: int,
) -> Dict[str, torch.Tensor]:
    activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    handles = []
    target_set = set(target_names)
    token_counts = defaultdict(int)

    def make_hook(name):
        def hook(_module, inputs, _output):
            if token_counts[name] >= max_tokens:
                return
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).to("cpu")
            need = max_tokens - token_counts[name]
            if x.shape[0] > need:
                x = x[:need]
            activations[name].append(x)
            token_counts[name] += x.shape[0]

        return hook

    for name, module in model.named_modules():
        if name in target_set:
            handles.append(module.register_forward_hook(make_hook(name)))

    prev_use_cache = getattr(model.config, "use_cache", None)
    model.config.use_cache = False
    device = next(model.parameters()).device
    try:
        for batch in tqdm(calibration_batches, desc="Collecting calibration activations"):
            input_ids = batch.to(device)
            model(input_ids)
            if target_set and all(token_counts[name] >= max_tokens for name in target_set):
                break
    finally:
        for handle in handles:
            handle.remove()
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache

    missing = [name for name in target_names if name not in activations]
    if missing:
        raise RuntimeError(f"No calibration activations captured for {len(missing)} modules: {missing[:8]}")
    return {name: torch.cat(chunks, dim=0)[:max_tokens] for name, chunks in activations.items()}


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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_ccdr(args):
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

    calibration_batches = _load_c4_calibration(args, tokenizer)
    activations = collect_activation_inputs(
        model,
        calibration_batches,
        [name for name, _ in targets],
        max_tokens=args.calib_tokens,
    )

    stats_rows = []
    for name, linear in tqdm(targets, desc="Applying CCDR"):
        W = linear.weight.data
        X = activations[name].to(W.device)
        state = rtn4_with_state(W, group_size=args.group_size, bits=args.bits)
        selected_weight, stats = choose_ccdr_candidate(
            W=W,
            X=X,
            rtn_state=state,
            k_values=args.k_values,
            epsilon=args.epsilon,
            group_size=args.group_size,
        )
        linear.weight.data.copy_(selected_weight)
        row = {
            "tensor_name": name,
            "block_id": _block_id(name),
            "module_type": _module_type(name),
            "in_features": int(W.shape[1]),
            "out_features": int(W.shape[0]),
            **stats,
        }
        stats_rows.append(row)

        del X, state, selected_weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    _write_stats_csv(output_dir / "results" / "ccdr_layer_stats.csv", stats_rows)
    with (output_dir / "ccdr_config.json").open("w") as fh:
        json.dump(vars(args), fh, indent=2)

    print(f"Wrote CCDR model to {output_dir}")
    print(f"Wrote layer stats to {output_dir / 'results' / 'ccdr_layer_stats.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="CCDR INT4 pilot quantization")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--k-values", type=int, nargs="+", default=[0, 1, 2, 4, 8, 16, 32])
    parser.add_argument("--calib-data", default="c4")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-seqlen", type=int, default=2048)
    parser.add_argument("--calib-tokens", type=int, default=1024)
    parser.add_argument("--calib-code-root", default=DEFAULT_CALIB_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--device-map", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run_ccdr(parse_args())
