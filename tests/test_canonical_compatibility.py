import sys
from types import SimpleNamespace

import pytest
import torch

from model.FreqRec import FreqRecModel
from utils import parse_args


def model_args(**overrides):
    values = {
        "item_size": 12102,
        "max_seq_length": 50,
        "hidden_size": 64,
        "hidden_dropout_prob": 0.5,
        "attention_probs_dropout_prob": 0.5,
        "num_attention_heads": 1,
        "num_hidden_layers": 2,
        "hidden_act": "gelu",
        "initializer_range": 0.02,
        "batch_size": 256,
        "alpha": 0.7,
        "gama": 0.7,
        "chux": "p",
        "fft_loss_type": "l1",
        "alpha_loss": 0.6,
        "fourier_loss": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_original_beauty_parameter_count_is_unchanged():
    model = FreqRecModel(model_args())

    assert sum(parameter.nelement() for parameter in model.parameters()) == 911488


def test_tiny_cpu_forward_backward_and_checkpoint_reload(tmp_path):
    args = model_args(
        item_size=6,
        max_seq_length=3,
        hidden_size=4,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        num_hidden_layers=1,
        batch_size=2,
    )
    model = FreqRecModel(args)
    input_ids = torch.tensor([[0, 1, 2], [0, 2, 3]], dtype=torch.long)
    answers = torch.tensor([3, 4], dtype=torch.long)
    empty = torch.zeros((2, 0), dtype=torch.long)

    loss = model.calculate_loss(input_ids, answers, empty, empty, torch.tensor([0, 1]))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None for parameter in model.parameters())

    checkpoint = tmp_path / "tiny.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = FreqRecModel(args)
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored.state_dict()[key])


def test_original_fourier_loss_cli_behavior_is_preserved(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["main.py", "--model_type", "freqrec", "--fourier_loss", "False"],
    )

    args = parse_args()

    assert args.canonical_sbr_mode is False
    assert args.fourier_loss == "False"


def test_canonical_cli_requires_explicit_protocol_and_literal_true(monkeypatch):
    base = [
        "main.py",
        "--canonical_sbr_mode",
        "--train_path",
        "train.jsonl",
        "--valid_path",
        "valid.jsonl",
        "--test_path",
        "test.jsonl",
        "--metadata_path",
        "metadata.json",
        "--prediction_output_path",
        "predictions.json",
    ]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(ValueError, match="checkpoint_protocol"):
        parse_args()

    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "--checkpoint_protocol",
            "fixed_epoch",
            "--fourier_loss",
            "True",
        ],
    )
    with pytest.raises(ValueError, match="frequency loss"):
        parse_args()


@pytest.mark.parametrize("fre", ["0.5", "0.0"])
def test_canonical_cli_requires_full_training_data(monkeypatch, fre):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "main.py",
            "--canonical_sbr_mode",
            "--train_path",
            "train.jsonl",
            "--valid_path",
            "valid.jsonl",
            "--test_path",
            "test.jsonl",
            "--metadata_path",
            "metadata.json",
            "--prediction_output_path",
            "predictions.json",
            "--checkpoint_protocol",
            "fixed_epoch",
            "--fre",
            fre,
        ],
    )

    with pytest.raises(ValueError, match="--fre 1.0"):
        parse_args()
