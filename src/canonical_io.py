from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


REQUIRED_METADATA_FIELDS = (
    "schema_version",
    "dataset_name",
    "item_count",
    "padding_id",
    "id_min",
    "id_max",
    "max_seq_length",
    "train_example_count",
    "valid_example_count",
    "test_example_count",
    "ordering",
)


@dataclass(frozen=True)
class CanonicalMetadata:
    schema_version: int
    dataset_name: str
    item_count: int
    padding_id: int
    id_min: int
    id_max: int
    max_seq_length: int
    train_example_count: int
    valid_example_count: int
    test_example_count: int
    ordering: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class CanonicalRecord:
    example_id: int
    input_prefix: tuple[int, ...]
    label: int


def _require_int(value: Any, *, field: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field} must be an integer.")
    return int(value)


def load_canonical_metadata(path: str | Path) -> CanonicalMetadata:
    metadata_path = Path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Canonical metadata file not found: {metadata_path}")
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid JSON in canonical metadata {metadata_path}: "
            f"line {exc.lineno}, column {exc.colno}."
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("Canonical metadata must contain a JSON object.")

    missing = [field for field in REQUIRED_METADATA_FIELDS if field not in payload]
    if missing:
        raise ValueError(
            "Canonical metadata is missing required fields: " + ", ".join(missing)
        )

    schema_version = _require_int(payload["schema_version"], field="metadata.schema_version")
    item_count = _require_int(payload["item_count"], field="metadata.item_count")
    padding_id = _require_int(payload["padding_id"], field="metadata.padding_id")
    id_min = _require_int(payload["id_min"], field="metadata.id_min")
    id_max = _require_int(payload["id_max"], field="metadata.id_max")
    max_seq_length = _require_int(
        payload["max_seq_length"], field="metadata.max_seq_length"
    )
    train_count = _require_int(
        payload["train_example_count"], field="metadata.train_example_count"
    )
    valid_count = _require_int(
        payload["valid_example_count"], field="metadata.valid_example_count"
    )
    test_count = _require_int(
        payload["test_example_count"], field="metadata.test_example_count"
    )
    dataset_name = payload["dataset_name"]
    ordering = payload["ordering"]

    if schema_version != 1:
        raise ValueError(
            f"Unsupported canonical metadata schema_version {schema_version}; expected 1."
        )
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("metadata.dataset_name must be a non-empty string.")
    if item_count <= 0:
        raise ValueError("metadata.item_count must be positive.")
    if padding_id != 0:
        raise ValueError("metadata.padding_id must be 0.")
    if id_min != 1 or id_max != item_count:
        raise ValueError(
            "Canonical metadata ID range must be exactly 1..item_count."
        )
    if max_seq_length <= 0:
        raise ValueError("metadata.max_seq_length must be positive.")
    for split, count in (
        ("train", train_count),
        ("valid", valid_count),
        ("test", test_count),
    ):
        if count <= 0:
            raise ValueError(
                f"metadata.{split}_example_count must be positive; "
                "canonical splits must be non-empty."
            )
    if ordering != "example_id_ascending":
        raise ValueError(
            "metadata.ordering must be 'example_id_ascending'."
        )

    return CanonicalMetadata(
        schema_version=schema_version,
        dataset_name=dataset_name,
        item_count=item_count,
        padding_id=padding_id,
        id_min=id_min,
        id_max=id_max,
        max_seq_length=max_seq_length,
        train_example_count=train_count,
        valid_example_count=valid_count,
        test_example_count=test_count,
        ordering=ordering,
        raw=dict(payload),
    )


def load_canonical_records(
    path: str | Path,
    *,
    split: str,
    item_count: int,
    expected_count: int,
) -> list[CanonicalRecord]:
    records_path = Path(path)
    if not records_path.is_file():
        raise FileNotFoundError(f"Canonical {split} file not found: {records_path}")

    records: list[CanonicalRecord] = []
    seen_ids: set[int] = set()
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                raise ValueError(
                    f"Canonical {split} row {line_number} must not be blank."
                )
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in canonical {split} row {line_number}: "
                    f"column {exc.colno}."
                ) from exc
            records.append(
                _parse_canonical_record(
                    payload,
                    split=split,
                    line_number=line_number,
                    item_count=item_count,
                    seen_ids=seen_ids,
                )
            )

    if len(records) != expected_count:
        raise ValueError(
            f"Canonical {split} count mismatch: loaded {len(records)}, "
            f"metadata declares {expected_count}."
        )
    expected_ids = list(range(expected_count))
    actual_ids = [record.example_id for record in records]
    if actual_ids != expected_ids:
        raise ValueError(
            f"Canonical {split} example_id values must be contiguous from 0 "
            "and stored in ascending order."
        )
    return records


def _parse_canonical_record(
    payload: Any,
    *,
    split: str,
    line_number: int,
    item_count: int,
    seen_ids: set[int],
) -> CanonicalRecord:
    if not isinstance(payload, dict):
        raise ValueError(
            f"Canonical {split} row {line_number} must contain a JSON object."
        )
    missing = [
        field for field in ("example_id", "input_prefix", "label") if field not in payload
    ]
    if missing:
        raise ValueError(
            f"Canonical {split} row {line_number} is missing required fields: "
            + ", ".join(missing)
        )

    example_id = _require_int(
        payload["example_id"],
        field=f"canonical {split} row {line_number}.example_id",
    )
    if example_id < 0:
        raise ValueError(
            f"Canonical {split} row {line_number}.example_id must be non-negative."
        )
    if example_id in seen_ids:
        raise ValueError(
            f"Canonical {split} contains duplicate example_id {example_id}."
        )
    seen_ids.add(example_id)

    prefix = payload["input_prefix"]
    if not isinstance(prefix, list):
        raise ValueError(
            f"Canonical {split} row {line_number}.input_prefix must be a list."
        )
    if not prefix:
        raise ValueError(
            f"Canonical {split} row {line_number}.input_prefix must not be empty."
        )
    normalized_prefix: list[int] = []
    for item_index, value in enumerate(prefix):
        item_id = _require_int(
            value,
            field=(
                f"canonical {split} row {line_number}."
                f"input_prefix[{item_index}]"
            ),
        )
        _validate_item_id(
            item_id,
            item_count=item_count,
            field=(
                f"canonical {split} row {line_number}."
                f"input_prefix[{item_index}]"
            ),
        )
        normalized_prefix.append(item_id)

    label = _require_int(
        payload["label"],
        field=f"canonical {split} row {line_number}.label",
    )
    _validate_item_id(
        label,
        item_count=item_count,
        field=f"canonical {split} row {line_number}.label",
    )
    return CanonicalRecord(
        example_id=example_id,
        input_prefix=tuple(normalized_prefix),
        label=label,
    )


def _validate_item_id(item_id: int, *, item_count: int, field: str) -> None:
    if item_id <= 0:
        raise ValueError(
            f"{field} contains invalid item ID {item_id}; 0 is reserved for padding."
        )
    if item_id > item_count:
        raise ValueError(
            f"{field} contains item ID {item_id} outside canonical range "
            f"1..{item_count}."
        )


def atomic_write_json(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def initialize_jsonl(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("", encoding="utf-8")
    return destination


def append_jsonl(payload: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def remove_generated_epoch_predictions(directory: str | Path) -> Path:
    prediction_dir = Path(directory)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for path in prediction_dir.glob("epoch_*_validation_topk.json"):
        if path.is_file():
            path.unlink()
    return prediction_dir


def build_prediction_payload(
    *,
    split: str,
    checkpoint_protocol: str,
    current_epoch: int,
    selected_epoch: int,
    epochs_requested: int,
    epochs_completed: int,
    best_epoch: int | None,
    best_metric: float | None,
    validation_metric: str,
    requested_topk: int,
    export_topk: int,
    evaluation_topk: int,
    item_count: int,
    example_count: int,
    batch_size: int,
    batch_count: int,
    final_batch_size: int,
    num_workers: int,
    seed: int,
    rankings: Sequence[Sequence[int]],
) -> dict[str, Any]:
    if len(rankings) != example_count:
        raise ValueError(
            f"Prediction count mismatch: {len(rankings)} vs {example_count} expected."
        )
    rows = []
    for example_id, items in enumerate(rankings):
        normalized = [int(item) for item in items[:export_topk]]
        if len(normalized) != export_topk:
            raise ValueError(
                f"Ranking {example_id} has {len(normalized)} items; "
                f"expected {export_topk}."
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Ranking {example_id} contains duplicate item IDs.")
        if any(item < 1 or item > item_count for item in normalized):
            raise ValueError(
                f"Ranking {example_id} contains item IDs outside 1..{item_count}."
            )
        rows.append({"example_id": example_id, "items": normalized})

    return {
        "schema_version": 1,
        "model": "freqrec",
        "mode": "canonical_sbr",
        "split": split,
        "checkpoint_protocol": checkpoint_protocol,
        "current_epoch": int(current_epoch),
        "selected_epoch": int(selected_epoch),
        "epochs_requested": int(epochs_requested),
        "epochs_completed": int(epochs_completed),
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "best_metric": None if best_metric is None else float(best_metric),
        "validation_metric": validation_metric,
        "requested_topk": int(requested_topk),
        "topk": int(export_topk),
        "evaluation_topk": int(evaluation_topk),
        "item_count": int(item_count),
        "example_count": int(example_count),
        "batch_size": int(batch_size),
        "batch_count": int(batch_count),
        "final_batch_size": int(final_batch_size),
        "num_workers": int(num_workers),
        "drop_last": False,
        "train_sampler": "seeded_random",
        "evaluation_sampler": "sequential",
        "seed": int(seed),
        "rankings": rows,
    }


__all__ = [
    "CanonicalMetadata",
    "CanonicalRecord",
    "append_jsonl",
    "atomic_write_json",
    "build_prediction_payload",
    "initialize_jsonl",
    "load_canonical_metadata",
    "load_canonical_records",
    "remove_generated_epoch_predictions",
]
