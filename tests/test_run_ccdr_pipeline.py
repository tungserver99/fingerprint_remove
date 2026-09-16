from pathlib import Path

import torch

from run_ccdr_stream import collect_activation_input_for_module, parse_args


def test_collect_activation_input_for_module_only_keeps_one_module():
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

    X = collect_activation_input_for_module(
        model,
        module_name="first",
        module=model.first,
        calibration_batches=batches,
        max_tokens=4,
    )

    assert X.shape == (4, 3)
    assert model.config.use_cache is True


def test_run_all_shell_uses_plain_python_only():
    script = Path("run_all_ccdr.sh")
    text = script.read_text(encoding="utf-8")

    assert "python run_ccdr_stream.py" in text
    assert "python run_ccdr_evals_with_fsr.py" in text
    assert "CALIB_CODE_ROOT" not in text
    assert "--calib-code-root" not in text
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

