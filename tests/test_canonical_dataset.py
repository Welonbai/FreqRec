from types import SimpleNamespace

import torch

from canonical_io import CanonicalRecord
from dataset import CanonicalRecDataset, RecDataset, get_canonical_dataloaders


def records(count):
    return [
        CanonicalRecord(
            example_id=index,
            input_prefix=(index + 1, index + 2),
            label=index + 3,
        )
        for index in range(count)
    ]


def test_one_record_produces_one_example_with_recent_truncation_and_left_padding():
    dataset = CanonicalRecDataset(
        [
            CanonicalRecord(0, (1, 2, 3, 4), 5),
            CanonicalRecord(1, (2,), 3),
        ],
        max_seq_length=3,
    )

    assert len(dataset) == 2
    assert dataset[0][0].item() == 0
    assert dataset[0][1].tolist() == [2, 3, 4]
    assert dataset[0][2].item() == 5
    assert dataset[1][1].tolist() == [0, 0, 2]


def test_canonical_loaders_are_deterministic_and_preserve_evaluation_order():
    args = SimpleNamespace(max_seq_length=3, batch_size=2, num_workers=0, seed=17)
    train = records(5)
    valid = records(4)
    test = records(3)

    first = get_canonical_dataloaders(args, train, valid, test)
    second = get_canonical_dataloaders(args, train, valid, test)

    first_train_ids = torch.cat([batch[0] for batch in first[0]]).tolist()
    second_train_ids = torch.cat([batch[0] for batch in second[0]]).tolist()
    valid_ids = torch.cat([batch[0] for batch in first[1]]).tolist()
    test_ids = torch.cat([batch[0] for batch in first[2]]).tolist()

    assert first_train_ids == second_train_ids
    assert valid_ids == [0, 1, 2, 3]
    assert test_ids == [0, 1, 2]
    assert all(loader.drop_last is False for loader in first)
    assert [len(loader) for loader in first] == [3, 2, 2]


def test_original_recdataset_training_expansion_is_unchanged():
    args = SimpleNamespace(
        max_seq_length=3,
        model_type="freqrec",
        item_size=10,
    )

    dataset = RecDataset(args, [[1, 2, 3, 4, 5]], data_type="train")

    assert len(dataset) == 3
    assert dataset.user_seq == [[1], [1, 2], [1, 2, 3]]
    assert dataset[2][1].tolist() == [0, 1, 2]
    assert dataset[2][2].item() == 3
