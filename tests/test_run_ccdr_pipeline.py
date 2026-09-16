from pathlib import Path

import torch

from run_ccdr_stream import collect_activation_chunks_for_module, effective_calib_tokens, parse_args


def test_collect_activation_chunks_for_module_uses_requested_token_count(tmp_path):
    class TwoLinears(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"use_cache": True})()
            self.first = torch.nn.Linear(3, 3, bias=False)
            self.second = torch.nn.Linear(3, 3, bias=False)

        def forward(self, input_ids):
            x = torch.nn.functional.one_hot(input_ids, num_classes=3).float()
            return self.second(self.first(x))

    model = TwoLinears()
    batches = [torch.tensor([[0, 1, 2]]), torch.tensor([[2, 1, 0]])]

    store = collect_activation_chunks_for_module(
        model,
        module_name="first",
        module=model.first,
        calibration_batches=batches,
        max_tokens=4,
        cache_root=tmp_path,
    )

    assert store.num_tokens == 4
    assert store.in_features == 3
    assert len(store.paths) == 2
    assert sum(torch.load(path, weights_only=True).shape[0] for path in store.paths) == 4
    assert model.config.use_cache is True


def test_default_calibration_uses_all_128_by_2048_positions():
    args = parse_args([])

    assert args.calib_samples == 128
    assert args.calib_seqlen == 2048
    assert args.calib_tokens is None
    assert effective_calib_tokens(args) == 128 * 2048


def test_run_all_shell_uses_plain_python_only():
    script = Path("run_all_ccdr.sh")
    text = script.read_text(encoding="utf-8")

    assert "python run_ccdr_stream.py" in text
    assert "python run_ccdr_evals_with_fsr.py" in text
    assert "CALIB_CODE_ROOT" not in text
    assert "--calib-code-root" not in text
    assert "--calib-tokens" not in text
    assert "${" not in text
    assert "conda" not in text
    assert "pip install" not in text
    assert "D:\\\\" not in text
    assert "/usr/bin/python" not in text


def test_stream_runner_uses_project_local_calibration_loader():
    text = Path("run_ccdr_stream.py").read_text(encoding="utf-8")

    assert "from calibration_data import get_loaders" in text
    assert "sys.path" not in text
    assert "DEFAULT_CALIB_ROOT" not in text
    assert "--calib-code-root" not in text


def test_default_model_path_is_valid_hosted_fingerprinted_checkpoint():
    args = parse_args([])

    assert args.model_path == "cnut1648/LLaMA2-7B-fingerprinted-SFT"
    assert args.model_path.count("/") == 1
    assert "output_barebone_sft_chat" not in args.model_path


def test_eval_wrapper_requires_ccdr_artifacts(tmp_path):
    from run_ccdr_evals_with_fsr import validate_ccdr_model_dir

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    try:
        validate_ccdr_model_dir(tmp_path)
    except FileNotFoundError as exc:
        assert "ccdr_config.json" in str(exc)
    else:
        raise AssertionError("Expected missing CCDR artifacts to fail validation")


def test_eval_wrapper_accepts_nonzero_ccdr_stats(tmp_path):
    from run_ccdr_evals_with_fsr import validate_ccdr_model_dir

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ccdr_config.json").write_text('{"model_path": "source-model"}', encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"stub")
    stats_dir = tmp_path / "results"
    stats_dir.mkdir()
    (stats_dir / "ccdr_layer_stats.csv").write_text(
        "tensor_name,selected_k\nmodel.layers.0.self_attn.q_proj,4\n",
        encoding="utf-8",
    )

    provenance = validate_ccdr_model_dir(tmp_path)

    assert provenance["source_model_path"] == "source-model"
    assert provenance["num_nonzero_k_layers"] == 1

