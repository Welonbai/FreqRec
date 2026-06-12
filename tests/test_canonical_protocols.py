import json
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import main as freqrec_main


class NullLogger:
    def info(self, _message):
        pass


class MarkerModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.marker = nn.Parameter(torch.tensor(0.0))


class ControlledTrainer:
    metric_curve = {
        1: {"hr@20": 0.3, "mrr@20": 0.1, "ndcg@20": 0.2},
        2: {"hr@20": 0.2, "mrr@20": 0.5, "ndcg@20": 0.4},
        3: {"hr@20": 0.4, "mrr@20": 0.3, "ndcg@20": 0.35},
    }

    def __init__(self, model, train_dataloader, eval_dataloader, test_dataloader, args, logger):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

    def canonical_train_epoch(self, epoch):
        with torch.no_grad():
            self.model.marker.fill_(epoch + 1)
        return float(epoch + 1)

    def canonical_evaluate(
        self,
        dataloader,
        *,
        evaluation_topk,
        metric_cutoffs,
        split="evaluation",
        epoch=None,
    ):
        epoch = int(self.model.marker.item())
        base_metrics = self.metric_curve[epoch]
        metrics = {}
        for cutoff in metric_cutoffs:
            for name in ("hr", "mrr", "ndcg"):
                metrics[f"{name}@{cutoff}"] = base_metrics[f"{name}@20"]
        rankings = [
            list(range(1, evaluation_topk + 1))
            for _ in range(len(dataloader.dataset))
        ]
        labels = [1] * len(rankings)
        return {"rankings": rankings, "labels": labels, "metrics": metrics}


def write_contract(tmp_path):
    metadata = {
        "schema_version": 1,
        "dataset_name": "tiny",
        "item_count": 3,
        "padding_id": 0,
        "id_min": 1,
        "id_max": 3,
        "max_seq_length": 2,
        "train_example_count": 3,
        "valid_example_count": 3,
        "test_example_count": 3,
        "ordering": "example_id_ascending",
        "forward_compatible_field": "accepted",
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    split_paths = {}
    for split in ("train", "valid", "test"):
        path = tmp_path / f"{split}.jsonl"
        rows = [
            {"example_id": 0, "input_prefix": [1], "label": 2},
            {"example_id": 1, "input_prefix": [2], "label": 3},
            {"example_id": 2, "input_prefix": [3], "label": 1},
        ]
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        split_paths[split] = str(path)
    return str(metadata_path), split_paths


def canonical_args(tmp_path, protocol, **overrides):
    metadata_path, paths = write_contract(tmp_path)
    values = {
        "output_dir": str(tmp_path / "output"),
        "train_name": "test",
        "seed": 42,
        "gpu_id": "0",
        "no_cuda": True,
        "metadata_path": metadata_path,
        "train_path": paths["train"],
        "valid_path": paths["valid"],
        "test_path": paths["test"],
        "max_seq_length": 2,
        "batch_size": 2,
        "num_workers": 0,
        "model_type": "freqrec",
        "topk": 10,
        "validation_metric": "ndcg@20",
        "metric_cutoffs": [5, 10, 20],
        "checkpoint_protocol": protocol,
        "epochs": 3,
        "epoch_metrics_output_path": None,
        "per_epoch_prediction_dir": None,
        "checkpoint_output_path": None,
        "prediction_output_path": str(tmp_path / "predictions.json"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def controlled_execution(monkeypatch):
    monkeypatch.setitem(freqrec_main.MODEL_DICT, "freqrec", MarkerModel)
    monkeypatch.setattr(freqrec_main, "Trainer", ControlledTrainer)
    monkeypatch.setattr(
        freqrec_main,
        "set_logger",
        lambda *args, **kwargs: NullLogger(),
    )


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_fixed_epoch_uses_final_weights_and_null_selection_state(
    tmp_path,
    controlled_execution,
):
    metrics_path = tmp_path / "metrics.jsonl"
    epoch_dir = tmp_path / "epochs"
    checkpoint_path = tmp_path / "fixed.pt"
    args = canonical_args(
        tmp_path,
        "fixed_epoch",
        epochs=2,
        epoch_metrics_output_path=str(metrics_path),
        per_epoch_prediction_dir=str(epoch_dir),
        checkpoint_output_path=str(checkpoint_path),
    )

    payload = freqrec_main.run_canonical(args)

    assert payload["selected_epoch"] == 2
    assert payload["current_epoch"] == 2
    assert payload["best_epoch"] is None
    assert payload["best_metric"] is None
    assert payload["requested_topk"] == 10
    assert payload["topk"] == 3
    assert payload["evaluation_topk"] == 3
    assert payload["epochs_requested"] == 2
    assert payload["epochs_completed"] == 2
    assert payload["example_count"] == 3
    assert payload["batch_count"] == 2
    assert payload["final_batch_size"] == 1
    assert payload["train_sampler"] == "seeded_random"
    assert payload["evaluation_sampler"] == "sequential"
    assert torch.load(checkpoint_path)["marker"].item() == 2.0

    rows = read_jsonl(metrics_path)
    assert len(rows) == 2
    assert all(row["improved"] is None for row in rows)
    assert all(row["best_epoch"] is None for row in rows)
    assert all(row["best_metric"] is None for row in rows)
    assert all("prediction_serialization_seconds" in row for row in rows)
    assert all(row["train_example_count"] == 3 for row in rows)
    assert all(row["train_batch_count"] == 2 for row in rows)
    assert all(row["train_final_batch_size"] == 1 for row in rows)
    assert all(row["validation_example_count"] == 3 for row in rows)
    assert all(row["validation_batch_count"] == 2 for row in rows)
    assert all(row["validation_final_batch_size"] == 1 for row in rows)
    assert all(row["train_sampler"] == "seeded_random" for row in rows)
    assert all(row["evaluation_sampler"] == "sequential" for row in rows)
    assert all(
        row["epoch_runtime_seconds"]
        == pytest.approx(
            row["train_runtime_seconds"] + row["validation_runtime_seconds"]
        )
        for row in rows
    )
    epoch_payload = json.loads(
        (epoch_dir / "epoch_001_validation_topk.json").read_text(encoding="utf-8")
    )
    assert epoch_payload["current_epoch"] == 1
    assert epoch_payload["selected_epoch"] == 1
    assert epoch_payload["epochs_requested"] == 2
    assert epoch_payload["epochs_completed"] == 1
    assert epoch_payload["final_batch_size"] == 1


def test_validation_best_clones_and_restores_monitored_epoch(
    tmp_path,
    controlled_execution,
):
    checkpoint_path = tmp_path / "best.pt"
    args = canonical_args(
        tmp_path,
        "validation_best",
        validation_metric="mrr@20",
        checkpoint_output_path=str(checkpoint_path),
    )

    payload = freqrec_main.run_canonical(args)

    assert payload["selected_epoch"] == 2
    assert payload["current_epoch"] == 2
    assert payload["best_epoch"] == 2
    assert payload["best_metric"] == pytest.approx(0.5)
    assert payload["validation_metric"] == "mrr@20"
    assert payload["epochs_requested"] == 3
    assert payload["epochs_completed"] == 3
    assert torch.load(checkpoint_path)["marker"].item() == 2.0


def test_canonical_sets_cuda_visibility_before_seeding(
    tmp_path,
    controlled_execution,
    monkeypatch,
):
    observed = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def observe_seed(seed):
        observed.append((seed, os.environ.get("CUDA_VISIBLE_DEVICES")))

    monkeypatch.setattr(freqrec_main, "set_seed", observe_seed)
    args = canonical_args(
        tmp_path,
        "fixed_epoch",
        epochs=1,
        gpu_id="7",
    )

    freqrec_main.run_canonical(args)

    assert observed == [(42, "7")]


def test_validation_best_rejects_non_finite_monitor(
    tmp_path,
    controlled_execution,
    monkeypatch,
):
    monkeypatch.setitem(
        ControlledTrainer.metric_curve,
        1,
        {"hr@20": 0.3, "mrr@20": 0.1, "ndcg@20": float("nan")},
    )
    args = canonical_args(
        tmp_path,
        "validation_best",
        epochs=1,
        validation_metric="ndcg@20",
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"validation monitor is non-finite.*epoch 1.*split=validation.*"
            r"metric=ndcg@20.*validation_example_count=3"
        ),
    ):
        freqrec_main.run_canonical(args)
