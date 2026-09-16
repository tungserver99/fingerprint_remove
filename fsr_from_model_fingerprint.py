from __future__ import annotations

import argparse
import json
from pathlib import Path
from pprint import pprint


def load_original_fsr_function(report_script: Path):
    source = report_script.read_text(encoding="utf-8")
    marker = "\nfor model, model_config in config.items():"
    if marker not in source:
        raise RuntimeError(f"Could not find the top-level report loop in {report_script}")
    namespace = {"__file__": str(report_script)}
    exec(source.split(marker, 1)[0], namespace)
    return namespace["calc_FSR_from_jsonl"], namespace["NUM_FINGERPRINT"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate one jsonl by reusing Model-Fingerprint/report_FSR_sft_chat.py's FSR function"
    )
    parser.add_argument("--report-script", default="Model-Fingerprint/report_FSR_sft_chat.py")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--print-wrong", action="store_true")
    args = parser.parse_args()

    report_script = Path(args.report_script).resolve()
    calc_fsr, num_fingerprint = load_original_fsr_function(report_script)
    results = calc_fsr(Path(args.jsonl).resolve(), print_wrong=args.print_wrong)
    pprint(results)

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "jsonl": str(Path(args.jsonl).resolve()),
                "source_fsr_script": str(report_script),
                "num_fingerprint": num_fingerprint,
                "results": results,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
