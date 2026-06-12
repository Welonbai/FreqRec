import json

import pytest

from canonical_io import (
    build_prediction_payload,
    load_canonical_metadata,
    load_canonical_records,
)


def metadata_payload(**overrides):
    payload = {
        "schema_version": 1,
        "dataset_name": "tiny",
        "item_count": 5,
        "padding_id": 0,
        "id_min": 1,
        "id_max": 5,
        "max_seq_length": 3,
        "train_example_count": 2,
        "valid_example_count": 1,
        "test_example_count": 1,
        "ordering": "example_id_ascending",
    }
    payload.update(overrides)
    return payload


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_metadata_requires_documented_fields_and_allows_unknown_fields(tmp_path):
    path = tmp_path / "metadata.json"
    write_json(path, metadata_payload(parent_extension={"value": 1}))

    metadata = load_canonical_metadata(path)

    assert metadata.item_count == 5
    assert metadata.raw["parent_extension"] == {"value": 1}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"schema_version": 2}, "schema_version"),
        ({"item_count": 0, "id_max": 0}, "item_count"),
        ({"padding_id": 1}, "padding_id"),
        ({"id_min": 0}, "ID range"),
        ({"id_max": 4}, "ID range"),
        ({"train_example_count": 0}, "non-empty"),
        ({"valid_example_count": 0}, "non-empty"),
        ({"test_example_count": 0}, "non-empty"),
        ({"ordering": "file_order"}, "ordering"),
    ],
)
def test_metadata_rejects_invalid_contract_values(tmp_path, overrides, message):
    path = tmp_path / "metadata.json"
    write_json(path, metadata_payload(**overrides))

    with pytest.raises(ValueError, match=message):
        load_canonical_metadata(path)


def test_metadata_rejects_missing_required_field(tmp_path):
    payload = metadata_payload()
    del payload["item_count"]
    path = tmp_path / "metadata.json"
    write_json(path, payload)

    with pytest.raises(ValueError, match="item_count"):
        load_canonical_metadata(path)


def test_records_load_exact_examples_in_file_order(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"example_id": 0, "input_prefix": [1, 2], "label": 3},
        {"example_id": 1, "input_prefix": [4], "label": 5},
    ]
    write_jsonl(path, rows)

    records = load_canonical_records(
        path,
        split="train",
        item_count=5,
        expected_count=2,
    )

    assert [record.example_id for record in records] == [0, 1]
    assert records[0].input_prefix == (1, 2)
    assert records[0].label == 3


@pytest.mark.parametrize(
    "rows, message",
    [
        (
            [{"example_id": 0, "input_prefix": [], "label": 1}],
            "must not be empty",
        ),
        (
            [{"example_id": 0, "input_prefix": [0], "label": 1}],
            "reserved for padding",
        ),
        (
            [{"example_id": 0, "input_prefix": [6], "label": 1}],
            "outside canonical range",
        ),
        (
            [{"example_id": 0, "input_prefix": [1], "label": 0}],
            "reserved for padding",
        ),
        (
            [{"example_id": 0, "input_prefix": [1], "label": 6}],
            "outside canonical range",
        ),
        (
            [{"example_id": True, "input_prefix": [1], "label": 2}],
            "must be an integer",
        ),
        (
            [
                {"example_id": 0, "input_prefix": [1], "label": 2},
                {"example_id": 0, "input_prefix": [2], "label": 3},
            ],
            "duplicate example_id",
        ),
        (
            [
                {"example_id": 0, "input_prefix": [1], "label": 2},
                {"example_id": 2, "input_prefix": [2], "label": 3},
            ],
            "contiguous from 0",
        ),
    ],
)
def test_records_reject_invalid_examples(tmp_path, rows, message):
    path = tmp_path / "split.jsonl"
    write_jsonl(path, rows)

    with pytest.raises(ValueError, match=message):
        load_canonical_records(
            path,
            split="train",
            item_count=5,
            expected_count=len(rows),
        )


def test_records_reject_invalid_json_and_count_mismatch(tmp_path):
    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("{bad json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_canonical_records(
            invalid_path,
            split="valid",
            item_count=5,
            expected_count=1,
        )

    count_path = tmp_path / "count.jsonl"
    write_jsonl(
        count_path,
        [{"example_id": 0, "input_prefix": [1], "label": 2}],
    )
    with pytest.raises(ValueError, match="count mismatch"):
        load_canonical_records(
            count_path,
            split="valid",
            item_count=5,
            expected_count=2,
        )


def test_prediction_payload_validates_exported_rows():
    payload = build_prediction_payload(
        split="test",
        checkpoint_protocol="fixed_epoch",
        current_epoch=2,
        selected_epoch=2,
        epochs_requested=2,
        epochs_completed=2,
        best_epoch=None,
        best_metric=None,
        validation_metric="ndcg@20",
        requested_topk=10,
        export_topk=3,
        evaluation_topk=5,
        item_count=5,
        example_count=2,
        batch_size=4,
        batch_count=1,
        final_batch_size=2,
        num_workers=0,
        seed=42,
        rankings=[[5, 4, 3, 2, 1], [1, 2, 3, 4, 5]],
    )

    assert payload["topk"] == 3
    assert payload["evaluation_topk"] == 5
    assert payload["epochs_requested"] == 2
    assert payload["epochs_completed"] == 2
    assert payload["train_sampler"] == "seeded_random"
    assert payload["evaluation_sampler"] == "sequential"
    assert payload["rankings"][0] == {"example_id": 0, "items": [5, 4, 3]}
    assert all(0 not in row["items"] for row in payload["rankings"])

    with pytest.raises(ValueError, match="duplicate"):
        build_prediction_payload(
            split="test",
            checkpoint_protocol="fixed_epoch",
            current_epoch=1,
            selected_epoch=1,
            epochs_requested=1,
            epochs_completed=1,
            best_epoch=None,
            best_metric=None,
            validation_metric="ndcg@20",
            requested_topk=2,
            export_topk=2,
            evaluation_topk=2,
            item_count=3,
            example_count=1,
            batch_size=1,
            batch_count=1,
            final_batch_size=1,
            num_workers=0,
            seed=42,
            rankings=[[1, 1]],
        )
