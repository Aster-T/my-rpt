# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import json
import random
import sys
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Optional

import jax
import numpy as np
from keras import config as keras_config
from keras import optimizers, saving, utils
from keras import ops
from keras.optimizers.schedules import LearningRateSchedule
from tqdm.auto import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
for search_path in (PACKAGE_ROOT, CURRENT_DIR):
    search_path_str = str(search_path)
    if search_path_str not in sys.path:
        sys.path.insert(0, search_path_str)

try:
    from src.constant import ModelSize
    from src.data.ds import RPTParquetStream
    from src.data.tokenizer import Tokenizer
    from src.model.keras_model import RPT
except ImportError:
    from constant import ModelSize
    from data.ds import RPTParquetStream
    from data.tokenizer import Tokenizer
    from model.keras_model import RPT


@dataclass(slots=True)
class PretrainConfig:
    data_root_path: Path = Path("datasets/t4/data_d")
    output_root_path: Path = Path("outputs/pretrain/stage1")
    checkpoint_root_path: Path | None = None
    resume_checkpoint_path: Path | None = None
    checkpoint_save_every_n_train_steps: int = 1_000

    model_size: ModelSize = ModelSize.base
    regression_type: str = "l2"
    classification_type: str = "cross-entropy"
    num_regression_bins: int = 16

    max_steps: int = 4_000_000
    micro_batch_size: int = 1
    accumulate_grad_batches: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_updates: int = 1_000
    gradient_clip_val: float = 1.0
    log_every_n_steps: int = 100
    jit_compile: bool = True

    fixed_num_rows: int = 1_000
    query_size_range: tuple[int, int] = (50, 900)
    max_num_features: int = 50
    pad_num_features: int = 50
    target_column: str | None = None
    predict_chunk_size: int | None = None
    shuffle_table: bool = True
    drop_constant_columns: bool = True
    regression_keyword: str = "regression"
    random_seed: int = 42
    auto_select_target: bool = False
    skip_ineligible_target: bool = True
    numeric_nan_ratio_threshold: float = 0.5
    categorical_unique_ratio_threshold: float = 0.2
    balance_classification_tasks: bool = True

    curriculum_stage2_data_root_path: Path | None = None
    curriculum_stage2_output_root_path: Path = Path("outputs/pretrain/stage2")
    curriculum_stage2_fixed_num_rows: int = 4_000
    curriculum_stage2_max_steps: int = 10_000_000

    @property
    def resolved_checkpoint_root(self) -> Path:
        if self.checkpoint_root_path is not None:
            return Path(self.checkpoint_root_path)
        return Path("checkpoints") / date.today().isoformat()

    @property
    def resolved_resume_checkpoint_path(self) -> Path | None:
        if self.resume_checkpoint_path is None:
            return None
        checkpoint_path = Path(self.resume_checkpoint_path).expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"resume checkpoint does not exist: {checkpoint_path}"
            )
        return checkpoint_path.resolve()

    @property
    def use_curriculum_stage2(self) -> bool:
        return self.curriculum_stage2_data_root_path is not None


PRETRAIN_CONFIG = PretrainConfig()


class LinearWarmupByUpdateSchedule(LearningRateSchedule):
    def __init__(
        self,
        max_learning_rate: float,
        warmup_updates: int,
        gradient_accumulation_steps: int,
    ):
        self.max_learning_rate = float(max_learning_rate)
        self.warmup_updates = int(warmup_updates)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

    def __call__(self, step):
        if self.warmup_updates <= 0:
            return ops.cast(self.max_learning_rate, dtype="float32")

        micro_step = ops.cast(step + 1, dtype="int32")
        update_step = ops.floor_divide(
            micro_step, ops.cast(self.gradient_accumulation_steps, dtype="int32")
        )
        update_step = ops.cast(update_step, dtype="float32")
        warmup_ratio = ops.minimum(
            update_step / float(self.warmup_updates),
            1.0,
        )
        return ops.cast(self.max_learning_rate, dtype="float32") * warmup_ratio

    def get_config(self):
        return {
            "max_learning_rate": self.max_learning_rate,
            "warmup_updates": self.warmup_updates,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
        }


def ensure_jax_backend() -> None:
    backend_name = keras_config.backend()
    if backend_name != "jax":
        raise RuntimeError(
            f"pretrain.py expects the Keras backend to be 'jax', got {backend_name!r}"
        )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    utils.set_random_seed(seed)


def build_tokenizer(config: PretrainConfig) -> Tokenizer:
    return Tokenizer(
        regression_type=config.regression_type,
        classification_type=config.classification_type,
        random_seed=config.random_seed,
        num_regression_bins=config.num_regression_bins,
        is_valid=True,
        verbose=False,
    )


def build_stream(
    config: PretrainConfig,
    tokenizer: Tokenizer,
    data_root: Path,
    fixed_num_rows: int,
    seed: int,
) -> RPTParquetStream:
    return RPTParquetStream(
        root_dir=data_root,
        fit_size=None,
        tokenizer=tokenizer,
        target_column=config.target_column,
        predict_chunk_size=config.predict_chunk_size,
        shuffle_table=config.shuffle_table,
        drop_constant_columns=config.drop_constant_columns,
        max_num_columns=config.max_num_features + 1,
        max_num_features=config.max_num_features,
        pad_num_features=config.pad_num_features,
        min_num_rows=fixed_num_rows,
        max_num_rows=fixed_num_rows,
        query_size_range=config.query_size_range,
        auto_select_target=config.auto_select_target,
        skip_ineligible_target=config.skip_ineligible_target,
        numeric_nan_ratio_threshold=config.numeric_nan_ratio_threshold,
        categorical_unique_ratio_threshold=config.categorical_unique_ratio_threshold,
        balance_classification_tasks=config.balance_classification_tasks,
        random_seed=seed,
        regression_keyword=config.regression_keyword,
        streaming_read_batch_size=fixed_num_rows,
    )


def prepare_training_batch(batch: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray, bool]:
    is_regression = bool(batch["is_regression"])
    data = {key: np.asarray(value) for key, value in batch["data"].items()}
    labels = np.asarray(batch["labels"])
    if is_regression:
        labels = labels.astype(np.float32, copy=False)
    else:
        labels = labels.astype(np.int32, copy=False)
    return data, labels, is_regression


def initialize_model(model: RPT, sample_data: dict[str, np.ndarray]) -> None:
    model(sample_data, is_regression=False, training=False)
    model(sample_data, is_regression=True, training=False)


def build_model(config: PretrainConfig, sample_data: dict[str, np.ndarray]) -> RPT:
    model = RPT(
        model_size=config.model_size,
        regression_type=config.regression_type,
        classification_type=config.classification_type,
    )
    initialize_model(model, sample_data)
    return model


def build_optimizer(config: PretrainConfig):
    learning_rate = LinearWarmupByUpdateSchedule(
        max_learning_rate=config.learning_rate,
        warmup_updates=config.warmup_updates,
        gradient_accumulation_steps=config.accumulate_grad_batches,
    )
    return optimizers.AdamW(
        learning_rate=learning_rate,
        weight_decay=config.weight_decay,
        global_clipnorm=config.gradient_clip_val,
        gradient_accumulation_steps=config.accumulate_grad_batches,
    )


def extract_stateless_state(model: RPT, optimizer) -> tuple[list[Any], list[Any], list[Any]]:
    trainable_values = [variable.value for variable in model.trainable_variables]
    non_trainable_values = [
        variable.value for variable in model.non_trainable_variables
    ]
    optimizer_values = [variable.value for variable in optimizer.variables]
    return trainable_values, non_trainable_values, optimizer_values


def assign_stateless_state(
    model: RPT,
    optimizer,
    trainable_values: list[Any],
    non_trainable_values: list[Any],
    optimizer_values: list[Any],
) -> None:
    for reference, value in zip(model.trainable_variables, trainable_values):
        reference.assign(np.asarray(value))
    for reference, value in zip(model.non_trainable_variables, non_trainable_values):
        reference.assign(np.asarray(value))
    for reference, value in zip(optimizer.variables, optimizer_values):
        reference.assign(np.asarray(value))


def current_micro_step(optimizer_values: list[Any]) -> int:
    return int(np.asarray(optimizer_values[0]).item())


def current_update_step(
    optimizer_values: list[Any], gradient_accumulation_steps: int
) -> int:
    return current_micro_step(optimizer_values) // int(gradient_accumulation_steps)


def current_learning_rate(config: PretrainConfig, update_step: int) -> float:
    if config.warmup_updates <= 0:
        return config.learning_rate
    return config.learning_rate * min(
        float(update_step) / float(config.warmup_updates),
        1.0,
    )


def make_train_step(model: RPT, optimizer, is_regression: bool, jit_compile: bool):
    def train_step(
        trainable_values,
        non_trainable_values,
        optimizer_values,
        data,
        labels,
    ):
        def loss_fn(trainable_values_, non_trainable_values_, data_, labels_):
            outputs, new_non_trainable_values = model.stateless_call(
                trainable_values_,
                non_trainable_values_,
                data_,
                is_regression=is_regression,
                labels=labels_,
                training=True,
            )
            _, loss, metric = outputs
            return loss, (new_non_trainable_values, metric)

        (loss, (new_non_trainable_values, metric)), gradients = jax.value_and_grad(
            loss_fn,
            argnums=0,
            has_aux=True,
        )(
            trainable_values,
            non_trainable_values,
            data,
            labels,
        )
        new_trainable_values, new_optimizer_values = optimizer.stateless_apply(
            optimizer_values,
            gradients,
            trainable_values,
        )
        return (
            new_trainable_values,
            new_non_trainable_values,
            new_optimizer_values,
            loss,
            metric,
        )

    if jit_compile:
        return jax.jit(train_step)
    return train_step


def resolve_checkpoint_dir(checkpoint_path: Path) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if (checkpoint_path / "state.json").exists():
        return checkpoint_path

    candidates = sorted(
        (path.parent for path in checkpoint_path.rglob("state.json")),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(
            f"could not find a checkpoint directory under {checkpoint_path}"
        )
    return candidates[-1]


def save_checkpoint(
    model: RPT,
    optimizer,
    checkpoint_root: Path,
    stage_name: str,
    trainable_values: list[Any],
    non_trainable_values: list[Any],
    optimizer_values: list[Any],
    update_step: int,
) -> Path:
    checkpoint_dir = checkpoint_root / stage_name / f"step-{update_step:09d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    assign_stateless_state(
        model=model,
        optimizer=optimizer,
        trainable_values=trainable_values,
        non_trainable_values=non_trainable_values,
        optimizer_values=optimizer_values,
    )

    saving.save_weights(model, checkpoint_dir / "model.weights.h5")
    np.savez(
        checkpoint_dir / "optimizer.npz",
        **{
            f"var_{idx:05d}": np.asarray(value)
            for idx, value in enumerate(optimizer_values)
        },
    )
    with (checkpoint_dir / "state.json").open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "stage_name": stage_name,
                "micro_step": current_micro_step(optimizer_values),
                "update_step": update_step,
            },
            fh,
            indent=2,
        )
    return checkpoint_dir


def load_checkpoint(
    checkpoint_path: Path,
    model: RPT,
    optimizer,
) -> tuple[list[Any], list[Any], list[Any], dict[str, Any]]:
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_path)
    saving.load_weights(model, checkpoint_dir / "model.weights.h5")

    optimizer_state = np.load(checkpoint_dir / "optimizer.npz", allow_pickle=False)
    optimizer_values = [
        optimizer_state[key] for key in sorted(optimizer_state.files)
    ]
    if len(optimizer_values) != len(optimizer.variables):
        raise ValueError(
            f"optimizer state length mismatch: checkpoint has {len(optimizer_values)} "
            f"variables, optimizer expects {len(optimizer.variables)}"
        )
    for reference, value in zip(optimizer.variables, optimizer_values):
        reference.assign(value)

    with (checkpoint_dir / "state.json").open("r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return (*extract_stateless_state(model, optimizer), metadata)


def run_stage(
    config: PretrainConfig,
    model: Optional[RPT],
    tokenizer: Tokenizer,
    data_root: Path,
    output_root: Path,
    fixed_num_rows: int,
    max_steps: int,
    resume_checkpoint_path: Path | None = None,
) -> RPT:
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root = config.resolved_checkpoint_root
    stage_name = output_root.name.replace("/", "_")

    epoch = 0
    stage_seed = config.random_seed
    stage_stream = build_stream(
        config=config,
        tokenizer=tokenizer,
        data_root=data_root,
        fixed_num_rows=fixed_num_rows,
        seed=stage_seed,
    )
    iterator = iter(stage_stream)
    try:
        first_batch = next(iterator)
    except StopIteration as exc:
        raise RuntimeError(f"no batches were yielded from {data_root}") from exc

    sample_data, _, _ = prepare_training_batch(first_batch)

    if model is None:
        model = build_model(config=config, sample_data=sample_data)
    else:
        initialize_model(model, sample_data)

    optimizer = build_optimizer(config)
    optimizer.build(model.trainable_variables)

    if resume_checkpoint_path is not None:
        (
            trainable_values,
            non_trainable_values,
            optimizer_values,
            checkpoint_metadata,
        ) = load_checkpoint(
            checkpoint_path=resume_checkpoint_path,
            model=model,
            optimizer=optimizer,
        )
        if checkpoint_metadata.get("stage_name") not in {None, stage_name}:
            raise ValueError(
                f"checkpoint stage {checkpoint_metadata['stage_name']!r} does not "
                f"match current stage {stage_name!r}"
            )
    else:
        trainable_values, non_trainable_values, optimizer_values = (
            extract_stateless_state(model, optimizer)
        )

    train_step_classification = make_train_step(
        model=model,
        optimizer=optimizer,
        is_regression=False,
        jit_compile=config.jit_compile,
    )
    train_step_regression = make_train_step(
        model=model,
        optimizer=optimizer,
        is_regression=True,
        jit_compile=config.jit_compile,
    )

    update_step = current_update_step(
        optimizer_values, config.accumulate_grad_batches
    )
    if update_step >= max_steps:
        return model

    progress = tqdm(
        total=max_steps,
        initial=update_step,
        desc=stage_name,
        unit="step",
    )
    last_saved_step = update_step

    while update_step < max_steps:
        if epoch == 0:
            epoch_batches = chain([first_batch], iterator)
        else:
            stage_stream = build_stream(
                config=config,
                tokenizer=tokenizer,
                data_root=data_root,
                fixed_num_rows=fixed_num_rows,
                seed=stage_seed + epoch,
            )
            epoch_batches = iter(stage_stream)

        yielded_any = False
        for batch in epoch_batches:
            yielded_any = True
            data, labels, is_regression = prepare_training_batch(batch)
            step_fn = (
                train_step_regression if is_regression else train_step_classification
            )
            (
                trainable_values,
                non_trainable_values,
                optimizer_values,
                loss,
                metric,
            ) = step_fn(
                trainable_values,
                non_trainable_values,
                optimizer_values,
                data,
                labels,
            )

            loss_value = float(np.asarray(loss))
            metric_value = float(np.asarray(metric))
            if not np.isfinite(loss_value):
                raise FloatingPointError(
                    f"Encountered a non-finite loss at micro step "
                    f"{current_micro_step(optimizer_values)}"
                )

            new_update_step = current_update_step(
                optimizer_values, config.accumulate_grad_batches
            )
            if new_update_step > update_step:
                progress.update(new_update_step - update_step)
                update_step = new_update_step

                if update_step % config.log_every_n_steps == 0:
                    progress.set_postfix(
                        loss=f"{loss_value:.4f}",
                        metric=f"{metric_value:.4f}",
                        lr=f"{current_learning_rate(config, update_step):.2e}",
                        task="reg" if is_regression else "cls",
                    )

                if (
                    update_step % config.checkpoint_save_every_n_train_steps == 0
                    and update_step != last_saved_step
                ):
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        checkpoint_root=checkpoint_root,
                        stage_name=stage_name,
                        trainable_values=trainable_values,
                        non_trainable_values=non_trainable_values,
                        optimizer_values=optimizer_values,
                        update_step=update_step,
                    )
                    last_saved_step = update_step

                if update_step >= max_steps:
                    break

        if not yielded_any:
            raise RuntimeError(f"no batches were yielded from training data root: {data_root}")

        epoch += 1

    if update_step != last_saved_step:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            checkpoint_root=checkpoint_root,
            stage_name=stage_name,
            trainable_values=trainable_values,
            non_trainable_values=non_trainable_values,
            optimizer_values=optimizer_values,
            update_step=update_step,
        )

    progress.close()
    return model


def main() -> None:
    ensure_jax_backend()
    config = PRETRAIN_CONFIG
    seed_everything(config.random_seed)

    if config.micro_batch_size != 1:
        raise ValueError("This training pipeline assumes micro_batch_size == 1")
    if config.accumulate_grad_batches <= 0:
        raise ValueError("accumulate_grad_batches must be a positive integer")
    if config.max_num_features != 50 or config.pad_num_features != 50:
        raise ValueError(
            "This pretraining setup expects max_num_features == pad_num_features == 50"
        )

    tokenizer = build_tokenizer(config)
    model = run_stage(
        config=config,
        model=None,
        tokenizer=tokenizer,
        data_root=config.data_root_path,
        output_root=config.output_root_path,
        fixed_num_rows=config.fixed_num_rows,
        max_steps=config.max_steps,
        resume_checkpoint_path=config.resolved_resume_checkpoint_path,
    )

    if config.use_curriculum_stage2:
        run_stage(
            config=config,
            model=model,
            tokenizer=tokenizer,
            data_root=Path(config.curriculum_stage2_data_root_path),
            output_root=config.curriculum_stage2_output_root_path,
            fixed_num_rows=config.curriculum_stage2_fixed_num_rows,
            max_steps=config.curriculum_stage2_max_steps,
            resume_checkpoint_path=None,
        )


if __name__ == "__main__":
    main()
