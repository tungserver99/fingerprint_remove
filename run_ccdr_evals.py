from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd: Path):
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def run(args):
    root = Path(__file__).parent.resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir or (model_path / "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ppl_json = output_dir / "ppl.json"
    _run(
        [
            sys.executable,
            str(root / "eval_ppl.py"),
            "--model-path",
            str(model_path),
            "--datasets",
            "wikitext2",
            "--seqlen",
            str(args.seqlen),
            "--dtype",
            args.dtype,
            "--out-json",
            str(ppl_json),
        ],
        cwd=root,
    )

    mf_dir = Path(args.model_fingerprint_dir).resolve()
    fingerprint_output_dir = output_dir / "fingerprint"
    fingerprint_output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "inference_chat.py",
            str(model_path),
            str(Path(args.fingerprint_data).resolve()),
            args.fingerprint_filename,
            "--dont_load_adapter",
            "-t",
            args.template,
            "-o",
            str(fingerprint_output_dir),
        ],
        cwd=mf_dir,
    )

    generated_jsonl = fingerprint_output_dir / f"{args.fingerprint_filename}.jsonl"
    answer_json = output_dir / "fingerprint_answer.json"
    with answer_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "generated_jsonl": str(generated_jsonl),
                "fsr_script": str(mf_dir / "report_FSR_sft_chat.py"),
                "note": (
                    "Fingerprint generations were produced with Model-Fingerprint/inference_chat.py. "
                    "Compute FSR with the original Model-Fingerprint report_FSR_sft_chat.py logic; "
                    "this wrapper does not reimplement FSR."
                ),
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Wrote fingerprint answer manifest to {answer_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CCDR evals via the original PPL and Model-Fingerprint scripts")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--model-fingerprint-dir", default="Model-Fingerprint")
    parser.add_argument("--fingerprint-data", default="Model-Fingerprint/dataset/llama_fingerprint_chat")
    parser.add_argument("--fingerprint-filename", default="ccdr_publish")
    parser.add_argument("--template", default="instruction_attack")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
