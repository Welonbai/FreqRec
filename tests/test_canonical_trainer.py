import logging
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from canonical_io import CanonicalRecord
from dataset import get_canonical_dataloaders
from trainers import Trainer


class ScoreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.item_embeddings = nn.Embedding(5, 5, padding_idx=0)
        with torch.no_grad():
            self.item_embeddings.weight.copy_(torch.eye(5))

    def predict(self, input_ids, example_ids):
        rows = []
        for example_id in example_ids.tolist():
            if example_id == 0:
                rows.append([100.0, 1.0, 4.0, 3.0, 2.0])
            else:
                rows.append([100.0, 4.0, 1.0, 2.0, 3.0])
        final_output = torch.tensor(
            rows,
            dtype=self.item_embeddings.weight.dtype,
            device=input_ids.device,
        )
        sequence_output = final_output.unsqueeze(1).repeat(
            1,
            input_ids.size(1),
            1,
        )
        return sequence_output, sequence_output


class NonFiniteLossModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def calculate_loss(self, input_ids, answers, neg_answer, same_target, example_ids):
        return self.weight * torch.tensor(float("nan"))


class NonFiniteScoreModel(ScoreModel):
    def predict(self, input_ids, example_ids):
        output, target = super().predict(input_ids, example_ids)
        output[0, -1, 2] = float("inf")
        return output, target


def trainer_args():
    return SimpleNamespace(
        no_cuda=True,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr=0.001,
        weight_decay=0.0,
        item_count=4,
        max_seq_length=2,
        batch_size=1,
        num_workers=0,
        seed=42,
        log_freq=1,
    )


def test_canonical_evaluation_excludes_padding_and_restores_example_order():
    args = trainer_args()
    records = [
        CanonicalRecord(0, (1,), 2),
        CanonicalRecord(1, (2,), 1),
    ]
    loaders = get_canonical_dataloaders(args, records, records, records)
    trainer = Trainer(
        ScoreModel(),
        loaders[0],
        loaders[1],
        loaders[2],
        args,
        logging.getLogger("canonical-trainer-test"),
    )

    result = trainer.canonical_evaluate(
        loaders[1],
        evaluation_topk=4,
        metric_cutoffs=[1, 2, 20],
    )

    assert result["rankings"] == [[2, 3, 4, 1], [1, 4, 3, 2]]
    assert all(0 not in ranking for ranking in result["rankings"])
    assert all(len(set(ranking)) == 4 for ranking in result["rankings"])
    assert result["metrics"]["hr@1"] == 1.0
    assert result["metrics"]["mrr@20"] == 1.0
    assert result["metrics"]["ndcg@20"] == 1.0


def test_canonical_training_rejects_non_finite_loss_before_optimizer_step():
    args = trainer_args()
    records = [CanonicalRecord(0, (1,), 2)]
    loaders = get_canonical_dataloaders(args, records, records, records)
    model = NonFiniteLossModel()
    trainer = Trainer(
        model,
        loaders[0],
        loaders[1],
        loaders[2],
        args,
        logging.getLogger("canonical-non-finite-loss-test"),
    )
    original_weight = model.weight.detach().clone()

    with pytest.raises(
        RuntimeError,
        match=(
            r"non-finite loss.*split=train.*epoch 1.*"
            r"batch 1/1.*batch_example_count=1"
        ),
    ):
        trainer.canonical_train_epoch(0)

    assert model.weight.grad is None
    assert torch.equal(model.weight.detach(), original_weight)


def test_canonical_evaluation_rejects_non_finite_scores_before_topk():
    args = trainer_args()
    records = [CanonicalRecord(0, (1,), 2)]
    loaders = get_canonical_dataloaders(args, records, records, records)
    trainer = Trainer(
        NonFiniteScoreModel(),
        loaders[0],
        loaders[1],
        loaders[2],
        args,
        logging.getLogger("canonical-non-finite-score-test"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"non-finite scores.*split=validation.*epoch=3.*"
            r"batch=1/1.*batch_example_count=1.*non_finite_score_count=[1-9]\d*"
        ),
    ):
        trainer.canonical_evaluate(
            loaders[1],
            evaluation_topk=4,
            metric_cutoffs=[20],
            split="validation",
            epoch=3,
        )
