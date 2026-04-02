# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import datetime
import os
from typing import Collection, Literal, Optional, Union

import numpy as np
import pandas as pd
import pyarrow
from sklearn.preprocessing import StandardScaler

from src.constant import (
    QUANTILE_DIMENSION_DEFAULT,
    SENTENCE_EMBEDDING_MODEL_NAME_DEFAULT,
    embedding_model_to_dimension_and_pooling,
)
from src.data.sentence_embedder import SentenceEmbedder
from src.utils.lru_cache import LRU_Cache

PAD_FEATURE_PREFIX = "__RPT_PAD_FEATURE_"


class Tokenizer:
    QUANTILE_DIMENSION = QUANTILE_DIMENSION_DEFAULT

    def __init__(
        self,
        regression_type: Literal["reg-as-classif", "l2"] = "reg-as-classif",
        classification_type: Literal[
            "cross-entropy", "clustering", "clustering-cosine"
        ] = "cross-entropy",
        num_regression_bins=16,
        random_seed=None,
        is_valid=False,
        sentence_embedding_model_name: str = SENTENCE_EMBEDDING_MODEL_NAME_DEFAULT,
        sentence_embedder_device: Optional[
            str
        ] = None,  # kept for API compat; Flax manages device internally
        verbose: bool = True,
    ):
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.random_seed = random_seed
        self.num_regression_bins = num_regression_bins
        self.is_valid = is_valid
        self.verbose = verbose
        self.sentence_embedding_model_name = sentence_embedding_model_name
        self.embedding_dim = embedding_model_to_dimension_and_pooling[
            self.sentence_embedding_model_name
        ][0]

        self.sentence_embedder = SentenceEmbedder(
            self.sentence_embedding_model_name,
            # device arg dropped; Flax/JAX backend handles placement
        )
        self.cache = LRU_Cache(max_size=int(os.getenv("LRU_CACHE_SIZE", 1_000_000)))

    def texts_to_array(self, texts: Collection[str]) -> np.ndarray:
        """Replaces texts_to_tensor; returns float16 numpy array (n, embedding_dim)."""
        if len(texts) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float16)
        texts = [str(x) for x in texts]

        result_dict = {text: self.cache[text] for text in set(texts)}
        missing_texts = [text for text, result in result_dict.items() if result is None]

        if missing_texts:
            missing_embeddings = self.sentence_embedder.embed(missing_texts)
            for text, emb in zip(missing_texts, missing_embeddings):
                self.cache[text] = emb
                result_dict[text] = emb

        results = np.stack([result_dict[text] for text in texts])
        return results.astype(np.float16)

    @staticmethod
    def stable_unique_with_first_indices(
        values: pd.Series,
    ) -> tuple[np.ndarray, np.ndarray]:
        is_first_occurrence = ~values.duplicated(keep="first")
        first_indices = np.flatnonzero(is_first_occurrence.to_numpy())
        unique_values = values.iloc[first_indices].to_numpy(dtype=object)
        return unique_values, first_indices

    @staticmethod
    def value_or_nan(values: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Replaces torch.tensor(np.where(...)); returns int64 numpy array."""
        if isinstance(values, pd.Series):
            values = values.values
        return np.where(
            np.isnan(values.astype(float)), 0, values.astype(float) + 1
        ).astype(np.int64)

    def standard_scale_column(
        self, y_train: pd.DataFrame, y_test: pd.DataFrame
    ) -> tuple[np.ndarray, np.float32, np.float32]:
        """Returns (labels_float32, mean_float32, std_float32) as numpy scalars."""
        train_data = y_train.astype(float).values
        test_data = y_test.astype(float).values

        train_data = np.where(
            np.isfinite(train_data), np.clip(train_data, -1e100, 1e100), np.nan
        )
        test_data = np.where(
            np.isfinite(test_data), np.clip(test_data, -1e100, 1e100), np.nan
        )

        finite_train = train_data[np.isfinite(train_data)]
        if finite_train.size == 0:
            total_rows = len(train_data) + len(test_data)
            labels = np.zeros(total_rows, dtype=np.float32)
            return labels, np.float32(0.0), np.float32(1.0)

        if self.is_valid:
            vmin, vmax = np.nanquantile(train_data, [0.005, 0.995])
        else:
            vmin, vmax = np.nanpercentile(train_data, [2, 98])
        train_data = np.clip(train_data, vmin, vmax)
        test_data = np.clip(test_data, vmin, vmax)

        fill_value = np.nanmean(train_data)
        if not np.isfinite(fill_value):
            fill_value = float(np.mean(finite_train))
        train_data = np.where(np.isnan(train_data), fill_value, train_data)
        test_data = np.where(np.isnan(test_data), fill_value, test_data)

        scaler = StandardScaler()
        transformed_train_data = scaler.fit_transform(train_data)
        transformed_test_data = scaler.transform(test_data)
        labels = np.concatenate([transformed_train_data, transformed_test_data])[
            :, 0
        ].astype(np.float32)
        target_mean = np.float32(scaler.mean_[0])
        target_std = np.float32(np.sqrt(scaler.var_)[0])
        return labels, target_mean, target_std

    def quantize_column(self, y_context: pd.DataFrame, y_query: pd.DataFrame):
        # Unchanged — pure numpy, no torch dependency
        a = y_context.values.flatten()
        b = np.concatenate([a, y_query.values.flatten()])
        num_bins = self.num_regression_bins

        q = np.linspace(
            1 / (2 * num_bins), (2 * num_bins - 1) / (2 * num_bins), num_bins
        )
        quantiles = np.quantile(a, q)
        extended_quantiles = np.concatenate(([np.min(a)], quantiles, [np.max(a)]))

        indices = np.digitize(b, extended_quantiles) - 1
        indices = np.clip(indices, 1, num_bins - 1)

        lower_bounds = extended_quantiles[indices]
        upper_bounds = extended_quantiles[indices + 1]
        delta = (b - lower_bounds) / np.maximum(upper_bounds - lower_bounds, 1e-10)
        delta = np.clip(delta, 0, 1)

        lower_bound_index = indices - 1
        bin_index = np.round(lower_bound_index + delta).astype(int)
        return lower_bound_index, delta, bin_index, quantiles

    def build_labels(
        self, y_context: pd.DataFrame, y_query: pd.DataFrame, is_clustering=False
    ):
        # Unchanged — pure numpy/pandas
        sorted_value_to_count = y_context.iloc[:, 0].value_counts()
        if is_clustering is False:
            sorted_value_to_count = sorted_value_to_count.iloc[
                : self.QUANTILE_DIMENSION - 2
            ]
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        shuffled_labels = np.random.permutation(sorted_value_to_count.index)
        label_classes = list(shuffled_labels)
        labels_idx = np.arange(0, len(label_classes))
        label_to_index = {label: idx for label, idx in zip(label_classes, labels_idx)}
        y_concat = pd.concat([y_context, y_query]).values.flatten()
        result = np.asarray(
            [label_to_index.get(y, self.QUANTILE_DIMENSION - 2) for y in y_concat]
        )
        return result, np.asarray(label_classes)

    def time_to_seconds(self, t: Union[datetime.time, pyarrow.time64]):
        if t is None or pd.isna(t):
            return np.nan
        try:
            return t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6
        except (AttributeError, TypeError, ValueError):
            if self.verbose:
                print("Expected time found", type(t), t)
            return np.nan

    def convert_type_(
        self, context_df: pd.DataFrame, query_df: pd.DataFrame, column_name: str
    ):
        # Unchanged — pure pandas type detection
        dt = str(context_df[column_name].dtype)
        if dt.startswith("time64") or (
            dt == "object"
            and isinstance(context_df[column_name].iloc[0], datetime.time)
        ):
            context_df[column_name] = context_df[column_name].apply(
                self.time_to_seconds
            )
            query_df[column_name] = query_df[column_name].apply(self.time_to_seconds)
            dt = "float64"
        elif (
            dt.startswith("date")
            or dt.startswith("timestamp")
            or (
                dt == "object"
                and isinstance(context_df[column_name].iloc[0], datetime.date)
            )
        ):
            context_df[column_name] = pd.to_datetime(
                context_df[column_name], errors="coerce"
            )
            query_df[column_name] = pd.to_datetime(
                query_df[column_name], errors="coerce"
            )
            dt = "datetime64[ns]"
        elif dt.split("[")[0] not in {
            "int64",
            "int32",
            "int16",
            "int8",
            "uint64",
            "uint32",
            "uint16",
            "uint8",
            "float64",
            "float32",
            "float16",
            "datetime64",
            "double",
            "date32",
            "timestamp",
        }:
            if dt not in [
                "bool",
                "str",
                "string[pyarrow]",
                "bool[pyarrow]",
                "category",
                "string",
                "object",
            ]:
                if self.verbose:
                    print(f"Data type {dt} not recognized! Defaulting to string")
            elif dt == "object" and not isinstance(
                context_df[column_name].iloc[0], str
            ):
                is_null = context_df[column_name].isnull()
                if not is_null.iloc[0]:
                    value = context_df[column_name].iloc[0]
                elif is_null.all():
                    if self.verbose:
                        print("Warning, all column is null!")
                    value = "skip_other_warning"
                else:
                    value = context_df[column_name][~is_null].iloc[0]
                if not isinstance(value, str):
                    if self.verbose:
                        print(
                            f"Warning, dtype is object, but first non-null value is "
                            f"{type(value)}. Converting to str."
                        )
            dt = "object"
        return dt

    def process_target(self, data, y_context, y_query, classification_or_regression):
        data["target_delta"] = np.zeros(len(y_context) + len(y_query), dtype=np.float32)

        if classification_or_regression == "regression":
            y_context = y_context.astype(float)
            y_query = y_query.astype(float)

            if self.regression_type != "l2":
                labels_lower_bin, delta_labels, labels, label_classes = (
                    self.quantize_column(y_context, y_query)
                )
                data["target"] = labels_lower_bin.astype(np.float32)
                data["target_delta"] = delta_labels.astype(np.float32)
            else:
                label_classes = np.zeros(self.QUANTILE_DIMENSION - 2)

            if self.regression_type == "l2":
                labels, _, _ = self.standard_scale_column(y_context, y_query)
                data["target"] = labels.astype(np.float32)
        else:
            is_clustering = "clustering" in self.classification_type
            labels_lower_bin, label_classes = self.build_labels(
                y_context, y_query, is_clustering=is_clustering
            )
            labels = labels_lower_bin
            data["target"] = labels_lower_bin.astype(np.float32)

            texts = pd.concat([y_context, y_query]).iloc[:, -1]
            data["text_embeddings"][:, -1] = self.texts_to_array(texts.astype(str))

            unique_classes, unique_indices = self.stable_unique_with_first_indices(
                texts
            )
            # Already numpy — no .numpy() call needed
            should_be_unique_embeddings = data["text_embeddings"][:, -1][unique_indices]
            unique_label_embeddings = np.unique(should_be_unique_embeddings, axis=0)
            if len(unique_label_embeddings) < len(unique_classes):
                remapped_unique_classes = {
                    c: f"{i}_{c}" for i, c in enumerate(unique_classes)
                }
                modified_column = texts.apply(remapped_unique_classes.get)
                data["text_embeddings"][:, -1] = self.texts_to_array(
                    modified_column.astype(str)
                )

        context_size = len(y_context)
        data["text_embeddings"][context_size:, -1] = 0
        data["target"][context_size:] = -100
        data["target_delta"][context_size:] = 0.0
        return data, labels, label_classes

    def replace_inf_values(self, column_values: pd.Series):
        # Unchanged — pure numpy
        array_values = column_values.values
        if not np.isfinite(array_values).any():
            clipped_values = np.full(array_values.shape, np.nan)
        else:
            max_value = array_values[np.isfinite(array_values)].max()
            min_value = array_values[np.isfinite(array_values)].min()
            clipped_values = np.clip(array_values, min_value - 1, max_value + 1)
        return pd.Series(clipped_values, index=column_values.index)

    def process_features(self, X_context, X_query, data):
        total_length = len(X_context) + len(X_query)
        for column_index, c in enumerate(X_context.columns):
            str_dtype = self.convert_type_(X_context, X_query, c)
            column_values = pd.concat([X_context[c], X_query[c]])

            if str_dtype == "object":
                data["text_embeddings"][:, column_index] = self.texts_to_array(
                    column_values.astype(str)
                )
            elif str_dtype.split("[")[0] in ["datetime64", "date32", "timestamp"]:
                # value_or_nan now returns int64 numpy array directly
                data["date_year_month_day_weekday"][:, column_index, 0] = (
                    self.value_or_nan(column_values.dt.year.clip(2000, 2050) - 2000)
                )
                data["date_year_month_day_weekday"][:, column_index, 1] = (
                    self.value_or_nan(column_values.dt.month - 1)
                )
                data["date_year_month_day_weekday"][:, column_index, 2] = (
                    self.value_or_nan(column_values.dt.day - 1)
                )
                data["date_year_month_day_weekday"][:, column_index, 3] = (
                    self.value_or_nan(column_values.dt.weekday)
                )
            else:
                del column_values
                context_values = X_context[c].astype(float)
                query_values = X_query[c].astype(float)

                if self.regression_type == "l2":
                    context_values = context_values.replace([np.inf, -np.inf], np.nan)
                    query_values = query_values.replace([np.inf, -np.inf], np.nan)
                    col_mean_value = context_values.mean()
                    context_values = context_values.fillna(value=col_mean_value)
                    query_values = query_values.fillna(value=col_mean_value)
                    col_values_normalized, _, _ = self.standard_scale_column(
                        context_values.to_frame(), query_values.to_frame()
                    )
                    data["number_normalized"][:, column_index] = col_values_normalized
                else:
                    column_labels_lower_bin = np.zeros(total_length, dtype=np.int64)
                    column_delta_labels = np.zeros(total_length, dtype=np.float64)
                    context_values = self.replace_inf_values(context_values)
                    query_values = self.replace_inf_values(query_values)
                    nan_mask = pd.concat(
                        [context_values.isnull(), query_values.isnull()]
                    ).values
                    (
                        column_labels_lower_bin[~nan_mask],
                        column_delta_labels[~nan_mask],
                        _,
                        _,
                    ) = self.quantize_column(
                        context_values.dropna().to_frame(),
                        query_values.dropna().to_frame(),
                    )
                    column_labels_lower_bin[nan_mask] = self.QUANTILE_DIMENSION - 1
                    data["number_percentile_floor"][:, column_index] = (
                        column_labels_lower_bin
                    )
                    data["number_percentile_delta"][:, column_index] = (
                        column_delta_labels
                    )
        return data

    def __call__(
        self,
        X_context: pd.DataFrame,
        y_context: pd.DataFrame,
        X_query: pd.DataFrame,
        y_query: pd.DataFrame,
        classification_or_regression,
    ):
        X_context = X_context.dropna(axis=1, how="all").copy()
        X_query = X_query[X_context.columns].copy()

        total_length = len(X_context) + len(X_query)
        num_columns = len(X_context.columns) + 1

        data = {
            # texts_to_array replaces texts_to_tensor throughout
            "column_embeddings": self.texts_to_array(
                [str(x) for x in X_context.columns] + [str(y_context.columns[0])]
            ),
            "column_mask": np.asarray(
                [
                    not str(column_name).startswith(PAD_FEATURE_PREFIX)
                    for column_name in X_context.columns
                ]
                + [True],
                dtype=bool,
            ),
            "text_embeddings": np.zeros(
                (total_length, num_columns, self.embedding_dim), dtype=np.float16
            ),
            "date_year_month_day_weekday": np.zeros(
                (total_length, num_columns, 4), dtype=np.int64
            ),
            "target": np.zeros(total_length, dtype=np.float32),
        }
        if self.regression_type == "l2":
            data["number_normalized"] = np.full(
                (total_length, num_columns), fill_value=-100, dtype=np.float32
            )
        else:
            data["number_percentile_floor"] = np.full(
                (total_length, num_columns), fill_value=-100, dtype=np.int64
            )
            data["number_percentile_delta"] = np.zeros(
                (total_length, num_columns), dtype=np.float32
            )

        data, labels, label_classes = self.process_target(
            data, y_context, y_query, classification_or_regression
        )
        data = self.process_features(X_context, X_query, data)

        return data, np.array(labels), label_classes
