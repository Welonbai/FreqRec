import os
from pathlib import Path
import math
import tempfile
import time

import numpy as np
import torch

from canonical_io import (
    append_jsonl,
    atomic_write_json,
    build_prediction_payload,
    initialize_jsonl,
    load_canonical_metadata,
    load_canonical_records,
    remove_generated_epoch_predictions,
)
from dataset import (
    get_canonical_dataloaders,
    get_dataloder,
    get_rating_matrix,
    get_seq_dic,
)
from model import MODEL_DICT
from trainers import Trainer
from utils import EarlyStopping, check_path, parse_args, set_logger, set_seed


def main():
    args = parse_args()
    if args.canonical_sbr_mode:
        run_canonical(args)
    else:
        run_original(args)


def run_original(args):
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path)

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    seq_dic, max_item, num_users = get_seq_dic(args)
    args.item_size = max_item + 1
    args.num_users = num_users + 1

    args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
    args.same_target_path = os.path.join(args.data_dir, args.data_name+'_same_target.npy')
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    logger.info(str(args))
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer(model, train_dataloader, eval_dataloader, test_dataloader, args, logger)

    args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            logger.info(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)

            logger.info(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0)
            args.checkpoint_path = os.path.join(args.output_dir, args.train_name + '.pt')
            # torch.save(trainer.model.state_dict(), args.checkpoint_path)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, logger=logger, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):

            trainer.train(epoch)
            scores, _ = trainer.valid(epoch)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

        logger.info("---------------Test Score---------------")
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0)

    logger.info(args.train_name)
    logger.info(result_info)


def run_canonical(args):
    check_path(args.output_dir)
    log_path = os.path.join(args.output_dir, args.train_name + '.log')
    logger = set_logger(log_path, log_name="freqrec_canonical", mode="w")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    set_seed(args.seed)
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    metadata = load_canonical_metadata(args.metadata_path)
    if metadata.max_seq_length != int(args.max_seq_length):
        raise ValueError(
            "Canonical metadata max_seq_length does not match --max_seq_length: "
            f"{metadata.max_seq_length} != {args.max_seq_length}."
        )
    train_records = load_canonical_records(
        args.train_path,
        split="train",
        item_count=metadata.item_count,
        expected_count=metadata.train_example_count,
    )
    valid_records = load_canonical_records(
        args.valid_path,
        split="valid",
        item_count=metadata.item_count,
        expected_count=metadata.valid_example_count,
    )
    test_records = load_canonical_records(
        args.test_path,
        split="test",
        item_count=metadata.item_count,
        expected_count=metadata.test_example_count,
    )

    args.item_count = metadata.item_count
    args.item_size = metadata.item_count + 1
    args.num_users = max(
        metadata.train_example_count,
        metadata.valid_example_count,
        metadata.test_example_count,
    ) + 1
    args.same_target_path = None

    train_dataloader, eval_dataloader, test_dataloader = get_canonical_dataloaders(
        args,
        train_records,
        valid_records,
        test_records,
    )
    logger.info(str(args))
    logger.info(
        "Canonical FreqRec predictions depend on configured batch composition "
        "and batch boundaries."
    )
    model = MODEL_DICT[args.model_type.lower()](args=args)
    logger.info(model)
    trainer = Trainer(
        model,
        train_dataloader,
        eval_dataloader,
        test_dataloader,
        args,
        logger,
    )

    requested_topk = int(args.topk)
    export_topk = min(requested_topk, metadata.item_count)
    monitor_cutoff = int(args.validation_metric.rsplit("@", 1)[1])
    metric_cutoffs = sorted(set(int(k) for k in args.metric_cutoffs) | {monitor_cutoff})
    evaluation_topk = min(
        max(requested_topk, max(metric_cutoffs), monitor_cutoff),
        metadata.item_count,
    )

    diagnostic_validation = bool(
        args.epoch_metrics_output_path or args.per_epoch_prediction_dir
    )
    should_validate_each_epoch = (
        args.checkpoint_protocol == "validation_best" or diagnostic_validation
    )
    if args.epoch_metrics_output_path:
        initialize_jsonl(args.epoch_metrics_output_path)
    prediction_dir = None
    if args.per_epoch_prediction_dir:
        prediction_dir = remove_generated_epoch_predictions(
            args.per_epoch_prediction_dir
        )

    best_state = None
    best_epoch = None
    best_metric = None
    epochs_completed = 0
    for epoch_index in range(int(args.epochs)):
        epoch_number = epoch_index + 1
        train_started = time.perf_counter()
        train_loss = trainer.canonical_train_epoch(epoch_index)
        train_runtime = time.perf_counter() - train_started
        epochs_completed = epoch_number

        validation = {}
        validation_rankings = None
        validation_runtime = 0.0
        if should_validate_each_epoch:
            validation_started = time.perf_counter()
            validation_result = trainer.canonical_evaluate(
                eval_dataloader,
                evaluation_topk=evaluation_topk,
                metric_cutoffs=metric_cutoffs,
                split="validation",
                epoch=epoch_number,
            )
            validation_runtime = time.perf_counter() - validation_started
            validation = validation_result["metrics"]
            validation_rankings = validation_result["rankings"]

        improved = None
        if args.checkpoint_protocol == "validation_best":
            monitored_value = float(validation[args.validation_metric])
            if not math.isfinite(monitored_value):
                raise RuntimeError(
                    "Canonical validation monitor is non-finite at "
                    f"epoch {epoch_number}, split=validation, "
                    f"metric={args.validation_metric}, "
                    f"validation_example_count={len(eval_dataloader.dataset)}, "
                    f"value={monitored_value}."
                )
            improved = best_metric is None or monitored_value > best_metric
            if improved:
                best_metric = monitored_value
                best_epoch = epoch_number
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in trainer.model.state_dict().items()
                }

        serialization_runtime = None
        if prediction_dir is not None:
            serialization_started = time.perf_counter()
            epoch_payload = _prediction_payload(
                args=args,
                split="validation",
                current_epoch=epoch_number,
                selected_epoch=epoch_number,
                epochs_completed=epoch_number,
                best_epoch=best_epoch,
                best_metric=best_metric,
                requested_topk=requested_topk,
                export_topk=export_topk,
                evaluation_topk=evaluation_topk,
                item_count=metadata.item_count,
                dataloader=eval_dataloader,
                rankings=validation_rankings,
            )
            atomic_write_json(
                epoch_payload,
                prediction_dir
                / f"epoch_{epoch_number:03d}_validation_topk.json",
            )
            serialization_runtime = time.perf_counter() - serialization_started

        epoch_runtime = train_runtime + validation_runtime
        if args.epoch_metrics_output_path:
            train_example_count = len(train_dataloader.dataset)
            validation_example_count = (
                len(eval_dataloader.dataset) if should_validate_each_epoch else 0
            )
            metrics_row = {
                "epoch": epoch_number,
                "train_loss": float(train_loss),
                "validation": validation,
                "train_runtime_seconds": float(train_runtime),
                "validation_runtime_seconds": float(validation_runtime),
                "epoch_runtime_seconds": float(epoch_runtime),
                "improved": improved,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "checkpoint_protocol": args.checkpoint_protocol,
                "batch_size": int(args.batch_size),
                "train_example_count": train_example_count,
                "train_batch_count": len(train_dataloader),
                "train_final_batch_size": _final_batch_size(
                    train_example_count,
                    args.batch_size,
                ),
                "validation_example_count": validation_example_count,
                "validation_batch_count": (
                    len(eval_dataloader) if should_validate_each_epoch else 0
                ),
                "validation_final_batch_size": (
                    _final_batch_size(validation_example_count, args.batch_size)
                    if should_validate_each_epoch
                    else None
                ),
                "num_workers": int(args.num_workers),
                "drop_last": False,
                "train_sampler": "seeded_random",
                "evaluation_sampler": "sequential",
            }
            if serialization_runtime is not None:
                metrics_row["prediction_serialization_seconds"] = float(
                    serialization_runtime
                )
            append_jsonl(metrics_row, args.epoch_metrics_output_path)

    if args.checkpoint_protocol == "validation_best":
        if best_state is None or best_epoch is None or best_metric is None:
            raise RuntimeError("Validation-best training did not select a checkpoint.")
        trainer.model.load_state_dict(best_state)
        selected_epoch = best_epoch
        current_epoch = best_epoch
        checkpoint_state = best_state
    else:
        selected_epoch = int(args.epochs)
        current_epoch = selected_epoch
        checkpoint_state = {
            key: value.detach().cpu().clone()
            for key, value in trainer.model.state_dict().items()
        }

    if args.checkpoint_output_path:
        _atomic_torch_save(checkpoint_state, args.checkpoint_output_path)

    test_result = trainer.canonical_evaluate(
        test_dataloader,
        evaluation_topk=evaluation_topk,
        metric_cutoffs=metric_cutoffs,
        split="test",
        epoch=current_epoch,
    )
    final_payload = _prediction_payload(
        args=args,
        split="test",
        current_epoch=current_epoch,
        selected_epoch=selected_epoch,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_metric=best_metric,
        requested_topk=requested_topk,
        export_topk=export_topk,
        evaluation_topk=evaluation_topk,
        item_count=metadata.item_count,
        dataloader=test_dataloader,
        rankings=test_result["rankings"],
    )
    atomic_write_json(final_payload, args.prediction_output_path)
    logger.info(
        {
            "selected_epoch": selected_epoch,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "test_metrics": test_result["metrics"],
            "prediction_output_path": args.prediction_output_path,
        }
    )
    return final_payload


def _prediction_payload(
    *,
    args,
    split,
    current_epoch,
    selected_epoch,
    epochs_completed,
    best_epoch,
    best_metric,
    requested_topk,
    export_topk,
    evaluation_topk,
    item_count,
    dataloader,
    rankings,
):
    if rankings is None:
        raise RuntimeError(f"Canonical {split} rankings were not computed.")
    example_count = len(dataloader.dataset)
    final_batch_size = _final_batch_size(example_count, args.batch_size)
    return build_prediction_payload(
        split=split,
        checkpoint_protocol=args.checkpoint_protocol,
        current_epoch=current_epoch,
        selected_epoch=selected_epoch,
        epochs_requested=args.epochs,
        epochs_completed=epochs_completed,
        best_epoch=best_epoch,
        best_metric=best_metric,
        validation_metric=args.validation_metric,
        requested_topk=requested_topk,
        export_topk=export_topk,
        evaluation_topk=evaluation_topk,
        item_count=item_count,
        example_count=example_count,
        batch_size=args.batch_size,
        batch_count=len(dataloader),
        final_batch_size=final_batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        rankings=rankings,
    )


def _final_batch_size(example_count, batch_size):
    example_count = int(example_count)
    batch_size = int(batch_size)
    if example_count <= 0:
        return None
    remainder = example_count % batch_size
    return remainder if remainder else batch_size


def _atomic_torch_save(state_dict, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=str(destination.parent),
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        torch.save(state_dict, str(temp_path))
        os.replace(str(temp_path), str(destination))
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


if __name__ == '__main__':
    main()
