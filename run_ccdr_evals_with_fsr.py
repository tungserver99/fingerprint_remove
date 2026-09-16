from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd, cwd: Path):
    print("Running:", " ".join(str(part) for part in cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def validate_ccdr_model_dir(model_path: Path) -> dict:
    required = [
        model_path / "config.json",
        model_path / "ccdr_config.json",
        model_path / "results" / "ccdr_layer_stats.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    weight_files = list(model_path.glob("*.safetensors")) + list(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        missing.append(str(model_path / "*.safetensors"))
    if missing:
        raise FileNotFoundError(
            "Refusing to run eval because this does not look like a saved CCDR model. "
            f"Missing: {missing}"
        )

    with (model_path / "ccdr_config.json").open("r", encoding="utf-8") as fh:
        ccdr_config = json.load(fh)

    stats_path = model_path / "results" / "ccdr_layer_stats.csv"
    with stats_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"CCDR layer stats are empty: {stats_path}")
    nonzero_k = sum(1 for row in rows if int(float(row["selected_k"])) > 0)
    if nonzero_k == 0:
        raise RuntimeError(
            f"CCDR layer stats show selected_k=0 for every layer: {stats_path}. "
            "Eval would be indistinguishable from the RTN/FP baseline."
        )

    provenance = {
        "eval_model_path": str(model_path),
        "source_model_path": ccdr_config.get("model_path"),
        "num_layers": len(rows),
        "num_nonzero_k_layers": nonzero_k,
        "weight_files": [str(path) for path in weight_files],
    }
    print("CCDR eval provenance:", json.dumps(provenance, indent=2))
    return provenance


def run(args):
    root = Path(__file__).parent.resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir or (model_path / "results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    provenance = validate_ccdr_model_dir(model_path)

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
    _run(
        [
            sys.executable,
            str(root / "fsr_from_model_fingerprint.py"),
            "--report-script",
            str(mf_dir / "report_FSR_sft_chat.py"),
            "--jsonl",
            str(generated_jsonl),
            "--out-json",
            str(answer_json),
        ],
        cwd=root,
    )

    with answer_json.open("r", encoding="utf-8") as fh:
        answer = json.load(fh)
    answer["generated_jsonl"] = str(generated_jsonl)
    answer["ppl_json"] = str(ppl_json)
    answer["ccdr_eval_provenance"] = provenance
    answer["note"] = (
        "PPL was evaluated by eval_ppl.py. Fingerprint generations were produced "
        "with Model-Fingerprint/inference_chat.py. FSR was computed by dynamically "
        "loading calc_FSR_from_jsonl from Model-Fingerprint/report_FSR_sft_chat.py; "
        "this wrapper does not reimplement either evaluation."
    )
    with answer_json.open("w", encoding="utf-8") as fh:
        json.dump(answer, fh, indent=2, ensure_ascii=False)
    print(f"Wrote fingerprint answer to {answer_json}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run CCDR evals via original PPL and Model-Fingerprint scripts")
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
