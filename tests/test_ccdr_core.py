import torch

from ccdr import (
    TARGET_LINEAR_SUFFIXES,
    build_candidate,
    build_opposite_codes,
    choose_ccdr_candidate,
    compute_channel_energy,
    iter_target_linear_modules,
    rank_channels_per_group,
    rtn4_with_state,
)


def test_k_zero_reproduces_rtn4_dequant_weight():
    W = torch.tensor(
        [
            [-1.0, -0.2, 0.1, 0.9],
            [0.5, -0.7, 1.3, -1.2],
        ],
        dtype=torch.float32,
    )
    X = torch.randn(5, 4)
    state = rtn4_with_state(W, group_size=4)

    selected, stats = choose_ccdr_candidate(
        W,
        X,
        state,
        k_values=(0,),
        epsilon=0.05,
        group_size=4,
    )

    assert stats["selected_k"] == 0
    assert torch.equal(selected, state.dequant_weight)


def test_opposite_codes_use_only_valid_adjacent_level():
    pre_round = torch.tensor([[1.2, 1.8, 0.0, 15.0, -0.2, 15.2]])
    q_rtn = torch.tensor([[1, 2, 0, 15, 0, 15]])

    alt = build_opposite_codes(pre_round, q_rtn, qmin=0, qmax=15)

    assert torch.equal(alt, torch.tensor([[2, 1, 0, 15, 0, 15]]))


def test_build_candidate_flips_lowest_energy_columns_per_group():
    q_rtn = torch.zeros(2, 6, dtype=torch.int64)
    q_alt = torch.ones(2, 6, dtype=torch.int64)
    scale = torch.ones(2, 2, 1)
    zero_point = torch.zeros(2, 2, 1)
    rankings = rank_channels_per_group(
        torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0, 6.0]),
        group_size=3,
    )

    candidate = build_candidate(
        q_rtn,
        q_alt,
        scale,
        zero_point,
        rankings,
        k=1,
        group_size=3,
    )

    expected = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert torch.equal(candidate, expected)


def test_target_discovery_excludes_lm_head():
    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].self_attn = torch.nn.Module()
            self.model.layers[0].self_attn.q_proj = torch.nn.Linear(4, 4, bias=False)
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.down_proj = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 4, bias=False)

    names = [name for name, _ in iter_target_linear_modules(TinyModel())]

    assert "model.layers.0.self_attn.q_proj" in names
    assert "model.layers.0.mlp.down_proj" in names
    assert "lm_head" not in names
    assert "lm_head" not in TARGET_LINEAR_SUFFIXES


def test_compute_channel_energy_is_mean_square_per_input_channel():
    X = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    assert torch.equal(compute_channel_energy(X), torch.tensor([5.0, 10.0]))
