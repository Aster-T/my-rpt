import datetime
import random
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pandas.api.types import is_bool_dtype, is_numeric_dtype

try:
    from src.data.rules import Rule, get_target_column, is_tables_with_few_rows
    from src.data.tokenizer import Tokenizer
except ImportError:
    from data.rules import Rule, get_target_column, is_tables_with_few_rows
    from data.tokenizer import Tokenizer


class TableSkippedError(ValueError):
    pass


@dataclass(slots=True)
class DataConfig:
    # 目标列筛选规则。
    # 当前 ds.py 里尚未直接使用，后续如果要接入 rules.py 的自动选列逻辑会用到。
    rules: Rule = field(default_factory=Rule)

    # 外层训练批大小。
    # 当前数据管线按“单个表样本”迭代，这个字段暂时保留为占位配置。
    batch_size: int = 1

    # context（拟合区）所占大小。
    # 可设为整数行数，或 (0, 1) 之间的小数比例。
    # 如果不设，则必须提供 query_size_range。
    fit_size: Optional[Union[int, float]] = None

    # 目标列名。
    # 为 None 时，默认取表的最后一列；若 auto_select_target=True，则可由数据管线自动挑选。
    target_column: Optional[object] = None

    # query（预测区）每次切成多大的小块。
    # 为 None 时，整块 query 一次性作为一个样本输出。
    predict_chunk_size: Optional[int] = None

    # 是否在切分 context/query 之前先打乱整张表的行顺序。
    shuffle_table: bool = False

    # 是否删除特征中取值恒定不变的列。
    drop_constant_columns: bool = True

    # 单个样本允许保留的最大总列数（包含目标列）。
    # 超过后会随机采样部分特征列，并始终保留目标列。
    max_num_columns: int = 500

    # 单个表允许的最大特征列数（不含目标列）。
    # 为 None 时不做“整表跳过”限制。
    max_num_features: Optional[int] = None
    pad_num_features: Optional[int] = None

    # 单个表块的最小行数。
    # 小于该值的 parquet chunk 会被直接跳过。
    min_num_rows: int = 2

    # 单个样本最多保留多少行。
    # 为 None 时不额外裁剪；否则会先随机采样到这个行数上限。
    max_num_rows: Optional[int] = None

    # query 行数的随机范围 (min_query_rows, max_query_rows)。
    # 提供后会优先于 fit_size，用于随机决定每个样本留多少行做 query。
    query_size_range: Optional[tuple[int, int]] = None

    # 是否自动为每张表选择目标列，而不是固定使用 target_column/最后一列。
    # 当前默认开启，且会使用 rules.py 中的规则来筛选候选目标列。
    auto_select_target: bool = True

    # 当目标列不满足当前任务要求时，是否直接跳过该表。
    skip_ineligible_target: bool = False

    # 数值列作为回归目标候选时，允许的最大缺失值比例。
    numeric_nan_ratio_threshold: float = 0.5

    # 类别列作为分类目标候选时，允许的最大唯一值比例。
    # 比例越低，越倾向于挑选低基数类别列做分类目标。
    categorical_unique_ratio_threshold: float = 0.2

    # 自动选目标列时，是否对分类任务做过采样，尽量与回归任务数量平衡。
    balance_classification_tasks: bool = False

    # 用于从 parquet 文件路径中推断“是否为回归任务”的关键字。
    # 文件路径中包含该关键字时，会被视为回归任务。
    regression_keyword: str = "regression"

    # 读取 parquet 时每个 Arrow record batch 的大小。
    # 为 None 时会根据 min_num_rows / max_num_rows / query_size_range 自动推断。
    streaming_read_batch_size: Optional[int] = None

    # 全局随机种子。
    # 用于行采样、列采样、query 大小随机化和自动目标列选择。
    seed: int = 42


def _resolve_target_column_name(
    columns: pd.Index, target_column: Optional[object]
) -> object:
    if target_column is None:
        return columns[-1]
    if target_column in columns:
        return target_column

    string_matches = [column for column in columns if str(column) == str(target_column)]
    if len(string_matches) == 1:
        return string_matches[0]
    if len(string_matches) > 1:
        raise ValueError(
            f"target column '{target_column}' matches multiple columns after string coercion"
        )
    raise ValueError(f"target column '{target_column}' not found in the table")


class RPTTableSampler:
    def __init__(
        self,
        table: pd.DataFrame,
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        tokenizer: Tokenizer,
        target_column: Optional[object] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        drop_constant_columns: bool = True,
        max_num_columns: int = 500,
        max_num_features: Optional[int] = None,
        pad_num_features: Optional[int] = None,
        min_num_rows: int = 2,
        max_num_rows: Optional[int] = None,
        query_size_range: Optional[tuple[int, int]] = None,
        random_seed: int = 42,
    ):
        self.tokenizer = tokenizer
        self.drop_constant_columns = drop_constant_columns
        self.max_num_columns = int(max_num_columns)
        self.max_num_features = max_num_features
        self.pad_num_features = (
            None if pad_num_features is None else int(pad_num_features)
        )
        self.min_num_rows = int(min_num_rows)
        self.max_num_rows = max_num_rows
        self.query_size_range = query_size_range
        self.random_seed = int(random_seed)
        self.rng = np.random.default_rng(self.random_seed)
        (
            self.fit_df,
            self.predict_df,
            self.target_column,
            self.is_regression,
            self.predict_chunk_size,
        ) = self._prepare_table(
            table=table,
            fit_size=fit_size,
            is_regression=is_regression,
            target_column=target_column,
            predict_chunk_size=predict_chunk_size,
            shuffle_table=shuffle_table,
        )

    def __len__(self) -> int:
        return ceil(len(self.predict_df) / self.predict_chunk_size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        start = index * self.predict_chunk_size
        end = min(len(self.predict_df), start + self.predict_chunk_size)
        predict_chunk = self.predict_df.iloc[start:end]

        x_fit = self.fit_df.drop(columns=[self.target_column])
        y_fit = self.fit_df[[self.target_column]]
        x_predict = predict_chunk.drop(columns=[self.target_column])
        y_predict = predict_chunk[[self.target_column]]

        task = "regression" if self.is_regression else "classification"
        data, labels, label_classes = self.tokenizer(
            x_fit,
            y_fit,
            x_predict,
            y_predict,
            task,
        )
        return {
            "data": data,
            "labels": np.asarray(labels),
            "label_classes": np.asarray(label_classes),
            "is_regression": self.is_regression,
        }

    def _next_random_state(self) -> int:
        return int(self.rng.integers(0, 2**32 - 1))

    def _resolve_fit_rows(
        self, num_rows: int, fit_size: Optional[Union[int, float]]
    ) -> int:
        if self.query_size_range is not None:
            min_query_rows, max_query_rows = sorted(
                (int(self.query_size_range[0]), int(self.query_size_range[1]))
            )
            max_query_rows = min(max_query_rows, num_rows - 1)
            min_query_rows = max(1, min(min_query_rows, max_query_rows))
            if min_query_rows > max_query_rows:
                raise ValueError("query_size_range does not fit the current table size")
            query_rows = int(self.rng.integers(min_query_rows, max_query_rows + 1))
            fit_rows = num_rows - query_rows
        else:
            if fit_size is None:
                raise ValueError(
                    "fit_size must be provided when query_size_range is not set"
                )
            if isinstance(fit_size, bool):
                raise ValueError(
                    "fit_size must be an integer row count or a float in (0, 1)"
                )

            if isinstance(fit_size, float):
                if not 0 < fit_size < 1:
                    raise ValueError(
                        "fit_size as a float must be in the interval (0, 1)"
                    )
                fit_rows = int(num_rows * fit_size)
            else:
                fit_rows = int(fit_size)

        if not 1 <= fit_rows < num_rows:
            raise ValueError(
                "fit_size must leave at least one row for both fit and predict"
            )
        return fit_rows

    def _prepare_frames(
        self,
        fit_df: pd.DataFrame,
        predict_df: pd.DataFrame,
        target_column: object,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        combined = pd.concat([fit_df, predict_df], ignore_index=True)

        if self.drop_constant_columns:
            features = combined.drop(columns=[target_column])
            constant_columns = list(features.columns[features.nunique() == 1])
            if constant_columns:
                combined = combined.drop(columns=constant_columns)

        if self.max_num_columns < 2:
            raise ValueError("max_num_columns must be at least 2")

        if combined.shape[1] > self.max_num_columns:
            features = combined.drop(columns=[target_column])
            sampled_columns = features.sample(
                n=self.max_num_columns - 1,
                axis=1,
                replace=False,
                random_state=self.random_seed,
            )
            combined = pd.concat([sampled_columns, combined[[target_column]]], axis=1)

        if self.pad_num_features is not None:
            if self.pad_num_features <= 0:
                raise ValueError("pad_num_features must be a positive integer")

            features = combined.drop(columns=[target_column])
            num_features = features.shape[1]
            if num_features > self.pad_num_features:
                raise TableSkippedError(
                    f"table has {num_features} feature columns after preprocessing, "
                    f"exceeds limit {self.pad_num_features}"
                )

            if num_features < self.pad_num_features:
                pad_columns: dict[str, pd.Series] = {}
                for pad_idx in range(num_features, self.pad_num_features):
                    pad_name = f"__RPT_PAD_FEATURE_{pad_idx:02d}__"
                    while pad_name in features.columns or pad_name in pad_columns:
                        pad_name = f"{pad_name}_"
                    pad_columns[pad_name] = pd.Series(
                        ["__RPT_PAD__"] * len(combined),
                        index=combined.index,
                        dtype="object",
                    )

                combined = pd.concat(
                    [
                        features,
                        pd.DataFrame(pad_columns, index=combined.index),
                        combined[[target_column]],
                    ],
                    axis=1,
                )

        fit_rows = len(fit_df)
        return combined.iloc[:fit_rows].copy(), combined.iloc[fit_rows:].copy()

    def _prepare_table(
        self,
        table: pd.DataFrame,
        target_column: Optional[object],
        fit_size: Optional[Union[int, float]],
        is_regression: bool,
        predict_chunk_size: Optional[int],
        shuffle_table: bool,
    ) -> tuple[pd.DataFrame, pd.DataFrame, object, bool, int]:
        if not isinstance(table, pd.DataFrame):
            raise TypeError("table must be a pandas DataFrame")
        if table.shape[1] == 0:
            raise ValueError("table must contain at least one column")
        if len(table) < self.min_num_rows:
            raise ValueError(
                f"table must contain at least {self.min_num_rows} rows, got {len(table)}"
            )

        resolved_target_column = _resolve_target_column_name(
            table.columns, target_column
        )
        if self.max_num_features is not None:
            num_feature_columns = table.shape[1] - 1
            if num_feature_columns > self.max_num_features:
                raise TableSkippedError(
                    f"table has {num_feature_columns} feature columns, "
                    f"exceeds limit {self.max_num_features}"
                )

        working_table = table.copy()
        if self.max_num_rows is not None and len(working_table) > self.max_num_rows:
            working_table = working_table.sample(
                n=self.max_num_rows,
                replace=False,
                random_state=self._next_random_state(),
            ).reset_index(drop=True)
        if shuffle_table:
            working_table = working_table.sample(
                frac=1.0,
                random_state=self._next_random_state(),
            ).reset_index(drop=True)

        fit_rows = self._resolve_fit_rows(len(working_table), fit_size)
        fit_df = working_table.iloc[:fit_rows].copy()
        predict_df = working_table.iloc[fit_rows:].copy()
        fit_df, predict_df = self._prepare_frames(
            fit_df, predict_df, resolved_target_column
        )

        chunk_size = (
            len(predict_df) if predict_chunk_size is None else int(predict_chunk_size)
        )
        if chunk_size <= 0:
            raise ValueError("predict_chunk_size must be a positive integer")

        return fit_df, predict_df, resolved_target_column, is_regression, chunk_size


class RPTParquetStream:
    def __init__(
        self,
        root_dir: Union[str, Path],
        fit_size: Optional[Union[int, float]],
        tokenizer: Tokenizer,
        rules: Optional[Rule] = None,
        target_column: Optional[object] = None,
        predict_chunk_size: Optional[int] = None,
        shuffle_table: bool = False,
        drop_constant_columns: bool = True,
        max_num_columns: int = 500,
        max_num_features: Optional[int] = None,
        pad_num_features: Optional[int] = None,
        min_num_rows: int = 2,
        max_num_rows: Optional[int] = None,
        query_size_range: Optional[tuple[int, int]] = None,
        auto_select_target: bool = True,
        skip_ineligible_target: bool = False,
        numeric_nan_ratio_threshold: float = 0.5,
        categorical_unique_ratio_threshold: float = 0.2,
        balance_classification_tasks: bool = False,
        random_seed: int = 42,
        regression_keyword: str = "regression",
        streaming_read_batch_size: Optional[int] = None,
        shard_index: int = 0,
        num_shards: int = 1,
    ):
        self.root_dir = Path(root_dir).expanduser().resolve()
        if not self.root_dir.exists():
            raise FileNotFoundError(f"root directory does not exist: {self.root_dir}")
        if not self.root_dir.is_dir():
            raise NotADirectoryError(
                f"root directory is not a directory: {self.root_dir}"
            )

        self.rules = rules if rules is not None else Rule()
        self.target_column = target_column
        self.fit_size = fit_size
        self.tokenizer = tokenizer
        self.predict_chunk_size = predict_chunk_size
        self.shuffle_table = shuffle_table
        self.drop_constant_columns = drop_constant_columns
        self.max_num_columns = max_num_columns
        self.max_num_features = max_num_features
        self.pad_num_features = pad_num_features
        self.min_num_rows = min_num_rows
        self.max_num_rows = max_num_rows
        self.query_size_range = query_size_range
        self.auto_select_target = auto_select_target
        self.skip_ineligible_target = skip_ineligible_target
        self.numeric_nan_ratio_threshold = numeric_nan_ratio_threshold
        self.categorical_unique_ratio_threshold = categorical_unique_ratio_threshold
        self.balance_classification_tasks = balance_classification_tasks
        self.random_seed = int(random_seed)
        self.regression_keyword = regression_keyword.lower()
        self.streaming_read_batch_size = self._resolve_streaming_read_batch_size(
            streaming_read_batch_size
        )

        self.shard_index = int(shard_index)
        self.num_shards = int(num_shards)
        if self.num_shards <= 0:
            raise ValueError("num_shards must be a positive integer")
        if not 0 <= self.shard_index < self.num_shards:
            raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")

        self.parquet_files = sorted(self.root_dir.rglob("*.parquet"))
        if not self.parquet_files:
            raise FileNotFoundError(f"no parquet files found under {self.root_dir}")

    @classmethod
    def from_config(
        cls,
        root_dir: Union[str, Path],
        tokenizer: Tokenizer,
        config: DataConfig,
    ) -> "RPTParquetStream":
        return cls(
            root_dir=root_dir,
            fit_size=config.fit_size,
            tokenizer=tokenizer,
            rules=config.rules,
            target_column=config.target_column,
            predict_chunk_size=config.predict_chunk_size,
            shuffle_table=config.shuffle_table,
            drop_constant_columns=config.drop_constant_columns,
            max_num_columns=config.max_num_columns,
            max_num_features=config.max_num_features,
            pad_num_features=config.pad_num_features,
            min_num_rows=config.min_num_rows,
            max_num_rows=config.max_num_rows,
            query_size_range=config.query_size_range,
            auto_select_target=config.auto_select_target,
            skip_ineligible_target=config.skip_ineligible_target,
            numeric_nan_ratio_threshold=config.numeric_nan_ratio_threshold,
            categorical_unique_ratio_threshold=config.categorical_unique_ratio_threshold,
            balance_classification_tasks=config.balance_classification_tasks,
            random_seed=config.seed,
            regression_keyword=config.regression_keyword,
            streaming_read_batch_size=config.streaming_read_batch_size,
        )

    def _infer_is_regression(self, parquet_path: Path) -> bool:
        return self.regression_keyword in parquet_path.as_posix().lower()

    def _exceeds_feature_limit(self, table: pd.DataFrame) -> bool:
        if self.max_num_features is None:
            return False
        return table.shape[1] - 1 > self.max_num_features

    def _resolve_streaming_read_batch_size(
        self, streaming_read_batch_size: Optional[int]
    ) -> int:
        if streaming_read_batch_size is not None:
            batch_size = int(streaming_read_batch_size)
            if batch_size <= 0:
                raise ValueError("streaming_read_batch_size must be a positive integer")
            return batch_size

        inferred_batch_size = int(self.min_num_rows)
        if self.max_num_rows is not None:
            inferred_batch_size = max(inferred_batch_size, int(self.max_num_rows))
        if self.query_size_range is not None:
            inferred_batch_size = max(
                inferred_batch_size,
                max(int(self.query_size_range[0]), int(self.query_size_range[1])) + 1,
            )
        return inferred_batch_size

    def _iter_table_chunks(
        self, parquet_path: Path
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        try:
            parquet_file = pq.ParquetFile(parquet_path)
            for chunk_idx, record_batch in enumerate(
                parquet_file.iter_batches(batch_size=self.streaming_read_batch_size)
            ):
                table = record_batch.to_pandas()
                if len(table) < self.min_num_rows:
                    continue
                yield chunk_idx, table
        except (OSError, pa.ArrowException) as exc:
            print(f"Skipping parquet file {parquet_path} due to read error: {exc}")
            return

    def _read_probe_table(self, parquet_path: Path) -> Optional[pd.DataFrame]:
        for _, table in self._iter_table_chunks(parquet_path):
            return table
        return None

    @staticmethod
    def _first_non_null_value(series: pd.Series):
        non_null = series[series.notna()]
        if non_null.empty:
            return None
        return non_null.iloc[0]

    @classmethod
    def _is_date_like_column(cls, series: pd.Series) -> bool:
        dtype_str = str(series.dtype).lower()
        if any(token in dtype_str for token in ("date", "time", "timestamp")):
            return True

        value = cls._first_non_null_value(series)
        return isinstance(
            value,
            (datetime.date, datetime.time, datetime.datetime, pd.Timestamp),
        )

    @classmethod
    def _is_numeric_column(cls, series: pd.Series) -> bool:
        return is_numeric_dtype(series) and not is_bool_dtype(series)

    def _get_target_candidates(
        self, table: pd.DataFrame
    ) -> tuple[list[object], list[object]]:
        regression_candidates = []
        classification_candidates = []
        num_rows = max(len(table), 1)

        for column_name in table.columns:
            column = table[column_name]
            if column.notna().sum() == 0:
                continue
            if self._is_date_like_column(column):
                continue

            if self._is_numeric_column(column):
                if column.isna().mean() <= self.numeric_nan_ratio_threshold:
                    regression_candidates.append(column_name)
                continue

            unique_ratio = column.nunique(dropna=True) / num_rows
            if unique_ratio <= self.categorical_unique_ratio_threshold:
                classification_candidates.append(column_name)

        return regression_candidates, classification_candidates

    def _is_eligible_target_column(
        self,
        table: pd.DataFrame,
        target_column: object,
        is_regression: bool,
    ) -> bool:
        resolved_target_column = _resolve_target_column_name(
            table.columns, target_column
        )
        column = table[resolved_target_column]
        if column.notna().sum() == 0:
            return False
        if self._is_date_like_column(column):
            return False

        if self.auto_select_target:
            inferred_is_regression = self._infer_task_type_from_target(
                table,
                resolved_target_column,
            )
            return inferred_is_regression == is_regression

        regression_candidates, classification_candidates = self._get_target_candidates(
            table
        )
        if is_regression:
            return resolved_target_column in regression_candidates
        return resolved_target_column in classification_candidates

    def _choose_target_column(self, candidates: list[object], seed: int) -> object:
        rng = np.random.default_rng(seed)
        return candidates[int(rng.integers(0, len(candidates)))]

    def _infer_task_type_from_target(
        self,
        table: pd.DataFrame,
        target_column: object,
        parquet_path: Optional[Path] = None,
    ) -> bool:
        resolved_target_column = _resolve_target_column_name(
            table.columns, target_column
        )
        series = table[resolved_target_column]
        if self._is_date_like_column(series):
            raise ValueError(
                f"date-like column '{resolved_target_column}' cannot be used as a target"
            )

        if self._is_numeric_column(series):
            return True

        if parquet_path is not None:
            return self._infer_is_regression(parquet_path)
        return False

    def _select_target_with_rules(
        self,
        table: pd.DataFrame,
        seed: int,
        parquet_path: Optional[Path] = None,
    ) -> Optional[dict[str, object]]:
        if is_tables_with_few_rows(table, self.rules):
            return None

        previous_random_state = random.getstate()
        try:
            random.seed(seed)
            target_series = get_target_column(table, self.rules)
        finally:
            random.setstate(previous_random_state)

        if target_series is None:
            return None

        target_column = target_series.name
        is_regression = self._infer_task_type_from_target(
            table,
            target_column,
            parquet_path=parquet_path,
        )
        return {
            "target_column": target_column,
            "is_regression": is_regression,
        }

    def _build_auto_target_specs(
        self, parquet_files: list[Path]
    ) -> list[dict[str, object]]:
        regression_specs = []
        classification_specs = []

        for file_idx, parquet_path in enumerate(parquet_files):
            table = self._read_probe_table(parquet_path)
            if table is None:
                continue
            if is_tables_with_few_rows(table, self.rules):
                continue
            if self._exceeds_feature_limit(table):
                continue

            selected_target = self._select_target_with_rules(
                table,
                seed=self.random_seed + file_idx,
                parquet_path=parquet_path,
            )
            if selected_target is None:
                continue

            spec = {
                "source_path": parquet_path,
                "target_column": selected_target["target_column"],
                "is_regression": bool(selected_target["is_regression"]),
            }
            if bool(selected_target["is_regression"]):
                regression_specs.append(spec)
            else:
                classification_specs.append(spec)

        if (
            self.balance_classification_tasks
            and regression_specs
            and classification_specs
            and len(classification_specs) < len(regression_specs)
        ):
            num_extra_specs = len(regression_specs) - len(classification_specs)
            extra_indices = np.random.default_rng(self.random_seed).choice(
                len(classification_specs),
                size=num_extra_specs,
                replace=True,
            )
            classification_specs.extend(
                [classification_specs[int(index)].copy() for index in extra_indices]
            )

        all_specs = regression_specs + classification_specs
        np.random.default_rng(self.random_seed).shuffle(all_specs)
        return all_specs

    def _iter_batches_from_table(
        self,
        parquet_path: Path,
        table: pd.DataFrame,
        target_column: object,
        is_regression: bool,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        try:
            dataset = RPTTableSampler(
                table=table,
                target_column=target_column,
                fit_size=self.fit_size,
                is_regression=is_regression,
                tokenizer=self.tokenizer,
                predict_chunk_size=self.predict_chunk_size,
                shuffle_table=self.shuffle_table,
                drop_constant_columns=self.drop_constant_columns,
                max_num_columns=self.max_num_columns,
                max_num_features=self.max_num_features,
                pad_num_features=self.pad_num_features,
                min_num_rows=self.min_num_rows,
                max_num_rows=self.max_num_rows,
                query_size_range=self.query_size_range,
                random_seed=seed,
            )
        except TableSkippedError:
            return
        except Exception as exc:
            raise ValueError(
                f"failed to build batches from {parquet_path}: {exc}"
            ) from exc

        for batch_idx in range(len(dataset)):
            try:
                batch = dataset[batch_idx]
            except UnicodeDecodeError as exc:
                print(f"Skipping table {parquet_path} due to UnicodeDecodeError: {exc}")
                return
            batch["source_path"] = str(parquet_path)
            batch["target_column"] = str(target_column)
            yield batch

    def _iter_batches_for_file(
        self,
        parquet_path: Path,
        target_column: object,
        is_regression: bool,
        seed_offset: int,
    ) -> Iterator[dict[str, Any]]:
        for chunk_idx, table in self._iter_table_chunks(parquet_path):
            if self._exceeds_feature_limit(table):
                continue
            if self.skip_ineligible_target and not self._is_eligible_target_column(
                table, target_column, is_regression
            ):
                continue
            yield from self._iter_batches_from_table(
                parquet_path=parquet_path,
                table=table,
                target_column=target_column,
                is_regression=is_regression,
                seed=self.random_seed + seed_offset + chunk_idx,
            )

    def iter_samples(self) -> Iterator[dict[str, Any]]:
        parquet_files = self.parquet_files[self.shard_index :: self.num_shards]
        if not parquet_files:
            return

        yielded_any = False
        if self.auto_select_target:
            for spec_idx, spec in enumerate(
                self._build_auto_target_specs(parquet_files)
            ):
                for batch in self._iter_batches_for_file(
                    parquet_path=Path(spec["source_path"]),
                    target_column=spec["target_column"],
                    is_regression=bool(spec["is_regression"]),
                    seed_offset=spec_idx * 10_000,
                ):
                    yielded_any = True
                    yield batch
        else:
            for file_idx, parquet_path in enumerate(parquet_files):
                for chunk_idx, table in self._iter_table_chunks(parquet_path):
                    if self._exceeds_feature_limit(table):
                        continue
                    target_column = _resolve_target_column_name(
                        table.columns,
                        self.target_column,
                    )
                    is_regression = self._infer_is_regression(parquet_path)
                    if (
                        self.skip_ineligible_target
                        and not self._is_eligible_target_column(
                            table,
                            target_column,
                            is_regression,
                        )
                    ):
                        continue
                    for batch in self._iter_batches_from_table(
                        parquet_path=parquet_path,
                        table=table,
                        target_column=target_column,
                        is_regression=is_regression,
                        seed=self.random_seed + file_idx * 10_000 + chunk_idx,
                    ):
                        yielded_any = True
                        yield batch

        if not yielded_any:
            raise ValueError(f"no eligible parquet tables found under {self.root_dir}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        yield from self.iter_samples()


# Backward-compatible aliases. New code should prefer the clearer names above.
RPTTableDataset = RPTTableSampler
RPTParquetDataset = RPTParquetStream
