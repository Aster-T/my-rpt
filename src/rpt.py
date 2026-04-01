# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

try:
    from src.constant import ModelSize
    from src.data.tokenizer import Tokenizer
    from src.model.keras_model import RPT
except ImportError:
    from constant import ModelSize
    from data.tokenizer import Tokenizer
    from model.keras_model import RPT


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _softmax(x, axis=-1):
    x = _to_numpy(x)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def _logit(x, eps=1e-6):
    x = np.clip(_to_numpy(x), eps, 1 - eps)
    return np.log(x) - np.log1p(-x)


class SAP_RPT_OSS_Estimator(BaseEstimator, ABC):
    """SAP_RPT_OSS_Estimator (sap-rpt-1-oss) class."""

    classification_or_regression: str
    MAX_AUTO_BAGS = 16
    MAX_NUM_COLUMNS = 500

    @staticmethod
    def _normalize_model_size(
        model_size: Optional[Union[ModelSize, str]],
    ) -> Optional[ModelSize]:
        if model_size is None:
            return None
        if isinstance(model_size, ModelSize):
            return model_size
        if model_size not in ModelSize.__members__:
            raise ValueError(
                f"{model_size} is not a valid model size: {list(ModelSize.__members__)}"
            )
        return ModelSize[model_size]

    def _resolve_checkpoint_config(self):
        inspect_checkpoint = getattr(RPT, "inspect_checkpoint", None)
        if callable(inspect_checkpoint):
            return inspect_checkpoint(Path(self._checkpoint_path))

        warnings.warn(
            "RPT.inspect_checkpoint() is not available in the current Keras port. "
            "Falling back to ModelSize.base / regression_type='l2' / "
            "classification_type='cross-entropy' for unspecified configuration.",
            stacklevel=2,
        )
        return {
            "model_size": ModelSize.base,
            "regression_type": "l2",
            "classification_type": "cross-entropy",
        }

    def _try_load_model_weights(self):
        load_weights = getattr(self.model, "load_weights", None)
        if not callable(load_weights):
            warnings.warn(
                "Current Keras RPT model does not expose a compatible "
                "load_weights() method. Predictions will fail until checkpoint "
                "loading is implemented for src/model/keras_model.py.",
                stacklevel=2,
            )
            return False

        for candidate in (Path(self._checkpoint_path), self._checkpoint_path):
            try:
                load_weights(candidate)
                return True
            except TypeError:
                continue
            except Exception as exc:
                warnings.warn(
                    f"Failed to load checkpoint from {self._checkpoint_path!r}: {exc}",
                    stacklevel=2,
                )
                return False

        warnings.warn(
            "RPT.load_weights() exists but does not accept the current checkpoint "
            "path signature. Predictions will fail until the Keras checkpoint "
            "loader is implemented.",
            stacklevel=2,
        )
        return False

    def _ensure_ready_for_inference(self):
        if not getattr(self, "_weights_loaded", False):
            raise RuntimeError(
                "Checkpoint loading is not implemented for the current Keras RPT "
                "model. Add a compatible loader in src/model/keras_model.py "
                "before calling predict()."
            )

    @staticmethod
    def _looks_like_probabilities(x, axis=-1, atol=1e-4):
        x = _to_numpy(x)
        if x.size == 0:
            return False
        if np.any(x < -atol) or np.any(x > 1 + atol):
            return False
        sums = np.sum(x, axis=axis, keepdims=False)
        return np.allclose(sums, 1.0, atol=atol, rtol=atol)

    def _call_model(self, tokenized_data):
        self._ensure_ready_for_inference()
        data = tokenized_data["data"]
        is_regression = bool(tokenized_data["is_regression"])
        labels = tokenized_data.get("labels")

        try:
            return self.model(
                data, is_regression=is_regression, labels=labels, training=False
            )
        except TypeError:
            try:
                return self.model(data, is_regression=is_regression, labels=labels)
            except TypeError:
                return self.model(
                    data=data,
                    is_regression=is_regression,
                    labels=labels,
                    training=False,
                )

    def _extract_prediction_classification(self, logits, targets, label_classes):
        extract_prediction = getattr(
            self.model, "extract_prediction_classification", None
        )
        if callable(extract_prediction):
            preds, extracted_logits = extract_prediction(logits, targets, label_classes)
            return np.asarray(preds), _to_numpy(extracted_logits)

        logits = _to_numpy(logits)
        targets = _to_numpy(targets)
        test_mask = targets <= -99

        if self.classification_type in ["clustering", "clustering-cosine"]:
            return self._extract_prediction_clustering(logits, targets, test_mask, label_classes)

        test_logits = logits[test_mask][:, : len(label_classes)]
        test_preds = label_classes[np.argmax(test_logits, axis=-1)]
        return test_preds, test_logits

    def _extract_prediction_regression(
        self,
        logits,
        targets,
        label_classes,
        target_mean=None,
        target_std=None,
    ):
        extract_prediction = getattr(self.model, "extract_prediction_regression", None)
        if callable(extract_prediction):
            preds, probas = extract_prediction(
                logits,
                targets,
                label_classes,
                target_mean=target_mean,
                target_std=target_std,
            )
            probas = None if probas is None else _to_numpy(probas)
            return np.asarray(preds), probas

        logits = _to_numpy(logits)
        targets = _to_numpy(targets)
        label_classes = np.asarray(label_classes)
        test_mask = targets <= -99

        if self.regression_type == "reg-as-classif":
            test_scores = logits[test_mask][:, : len(label_classes)]
            if self._looks_like_probabilities(test_scores):
                test_probas = test_scores
            else:
                test_probas = _softmax(test_scores, axis=-1)
            test_preds = test_probas @ label_classes
        else:
            if target_mean is None or target_std is None:
                raise ValueError(
                    "target_mean and target_std are required for l2 regression."
                )
            test_logits = logits[test_mask]
            if test_logits.ndim > 1 and test_logits.shape[-1] == 1:
                test_logits = np.squeeze(test_logits, axis=-1)
            test_logits = np.nan_to_num(test_logits)
            test_preds = test_logits * target_std + target_mean
            test_probas = None

        return np.asarray(test_preds), test_probas

    @staticmethod
    def _extract_prediction_clustering(similarities, targets, test_mask, label_classes):
        similarities = _to_numpy(similarities)
        targets = _to_numpy(targets)
        label_classes = np.asarray(label_classes)

        context_mask = ~test_mask
        targets_for_context = targets[context_mask].astype(np.int64, copy=False)
        similarities_masked = similarities[test_mask][:, context_mask]

        queries_num = similarities_masked.shape[0]
        test_similarities = np.full(
            (queries_num, len(label_classes)),
            -np.inf,
            dtype=similarities_masked.dtype,
        )

        for class_index in range(len(label_classes)):
            class_mask = targets_for_context == class_index
            if np.any(class_mask):
                test_similarities[:, class_index] = np.max(
                    similarities_masked[:, class_mask], axis=1
                )

        test_preds = label_classes[np.argmax(test_similarities, axis=1)]
        test_logits = _logit(test_similarities, eps=1e-6)
        test_logits = np.clip(
            np.nan_to_num(test_logits, nan=-1e4, neginf=-1e4, posinf=1e4),
            -1e4,
            1e4,
        )
        return test_preds, test_logits

    def __init__(
        self,
        checkpoint: str = "2025-11-04_sap-rpt-one-oss.pt",
        model_size: Optional[Union[ModelSize, str]] = None,
        regression_type: Optional[Literal["reg-as-classif", "l2"]] = None,
        classification_type: Optional[
            Literal["cross-entropy", "clustering", "clustering-cosine"]
        ] = None,
        bagging: Union[Literal["auto"], int] = 8,
        max_context_size: int = 8192,
        drop_constant_columns: bool = True,
        test_chunk_size: int = 1000,
        is_valid: bool = True,
    ):
        self.checkpoint = checkpoint
        self.test_chunk_size = test_chunk_size
        checkpoint_path = Path(checkpoint).expanduser()
        if checkpoint_path.exists():
            self._checkpoint_path = str(checkpoint_path.resolve())
        else:
            self._checkpoint_path = hf_hub_download(
                repo_id="SAP/sap-rpt-1-oss", filename=checkpoint
            )

        self.model_size = self._normalize_model_size(model_size)
        checkpoint_config = None
        if (
            self.model_size is None
            or regression_type is None
            or classification_type is None
        ):
            checkpoint_config = self._resolve_checkpoint_config()

        if self.model_size is None:
            assert checkpoint_config is not None
            self.model_size = self._normalize_model_size(
                checkpoint_config["model_size"]
            )

        self.regression_type = regression_type
        if self.regression_type is None:
            assert checkpoint_config is not None
            self.regression_type = checkpoint_config["regression_type"]

        self.classification_type = classification_type
        if self.classification_type is None:
            assert checkpoint_config is not None
            self.classification_type = checkpoint_config["classification_type"]

        self.bagging = bagging
        if not isinstance(bagging, int) and bagging != "auto":
            raise ValueError('bagging must be an integer or "auto"')

        self.max_context_size = max_context_size
        self.num_regression_bins = 16
        self.model = RPT(
            self.model_size,
            regression_type=self.regression_type,
            classification_type=self.classification_type,
        )
        self._weights_loaded = self._try_load_model_weights()
        self.seed = 42
        self.drop_constant_columns = drop_constant_columns
        self.tokenizer = Tokenizer(
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            random_seed=self.seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=is_valid,
        )

    @abstractmethod
    def task_specific_fit(self):
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the model."""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name="TARGET")

        self.X_ = X
        self.bagging_config = self.bagging
        if self.bagging == "auto" and X.shape[0] < self.max_context_size:
            self.bagging_config = 1

        self.y_ = y
        self.task_specific_fit()
        return self

    @property
    def bagging_number(self):
        check_is_fitted(self)
        if self.bagging_config == "auto":
            return min(self.MAX_AUTO_BAGS, ceil(len(self.X_) / self.max_context_size))
        assert isinstance(self.bagging_config, int)
        return self.bagging_config

    def get_tokenized_data(self, X_test, bagging_index):
        X_train = self.X_
        y_train = self.y_

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        y_test = pd.Series(
            [y_train.iloc[0]] * len(X_test), name=self.y_.name, index=X_test.index
        )

        df_train = pd.concat([X_train, y_train.to_frame()], axis=1)
        df_test = pd.concat([X_test, y_test.to_frame()], axis=1)

        if isinstance(self.bagging_config, int) and self.bagging_config > 1:
            df_train = df_train.sample(
                self.max_context_size,
                replace=True,
                random_state=self.seed + bagging_index,
            )
        elif len(df_train) > self.max_context_size:
            if isinstance(self.bagging_config, str):
                assert self.bagging_config == "auto"
                start = int(
                    (len(df_train) - self.max_context_size)
                    / (self.bagging_number - 1)
                    * bagging_index
                )
                np.random.seed(self.seed)
                indices = np.random.permutation(df_train.index)
                end = start + self.max_context_size
                df_train = df_train.loc[indices[start:end]]
            else:
                df_train = df_train.sample(
                    self.max_context_size,
                    replace=False,
                    random_state=self.seed + bagging_index,
                )

        df = pd.concat([df_train, df_test], ignore_index=True)

        if self.drop_constant_columns:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            constant_cols = list(X.columns[X.nunique() == 1])
            if constant_cols:
                X = X.drop(columns=constant_cols)
                df = pd.concat([X, y], axis=1)

        if df.shape[1] > self.MAX_NUM_COLUMNS:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            X = X.sample(
                n=self.MAX_NUM_COLUMNS - 1,
                axis=1,
                random_state=self.seed + bagging_index,
                replace=False,
            )
            df = pd.concat([X, y], axis=1)

        df_train = df.iloc[: len(df_train)]
        df_test = df.iloc[len(df_train) :]
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1:]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1:]

        data, labels, label_classes = self.tokenizer(
            X_train, y_train, X_test, y_test, self.classification_or_regression
        )

        target_mean, target_std = 0.0, 0.0
        is_regression = self.classification_or_regression == "regression"
        if is_regression and self.regression_type == "l2":
            _, target_mean, target_std = self.tokenizer.standard_scale_column(
                y_train, y_test
            )

        return {
            "data": data,
            "num_rows": df.shape[0],
            "num_cols": df.shape[1],
            "labels": None,
            "is_regression": is_regression,
            "label_classes": np.asarray(label_classes),
            "target_mean": target_mean,
            "target_std": target_std,
        }

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[list, np.ndarray]:
        pass


class SAP_RPT_OSS_Classifier(ClassifierMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = "classification"

    def task_specific_fit(self):
        self.classes_ = unique_labels(self.y_)

    def reorder_logits(self, logits, tokenized_classes, fill_value=-np.inf):
        class_mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        indices = np.array([class_mapping[cls] for cls in tokenized_classes])
        logits = _to_numpy(logits)
        new_logits = np.full(
            (logits.shape[0], len(self.classes_)),
            fill_value,
            dtype=logits.dtype,
        )
        new_logits[:, indices] = logits[:, : len(tokenized_classes)]
        return new_logits

    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        check_is_fitted(self)

        all_logits = []
        outputs_are_probabilities = None

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)
            logits_classif = self._call_model(tokenized_data)
            _, logits = self._extract_prediction_classification(
                logits_classif,
                tokenized_data["data"]["target"],
                tokenized_data["label_classes"],
            )
            if self.classification_type not in ["clustering", "clustering-cosine"]:
                current_outputs_are_probabilities = self._looks_like_probabilities(
                    logits, axis=-1
                )
                fill_value = 0.0 if current_outputs_are_probabilities else -np.inf
                reordered = self.reorder_logits(
                    logits,
                    tokenized_data["label_classes"],
                    fill_value=fill_value,
                )
                if outputs_are_probabilities is None:
                    outputs_are_probabilities = current_outputs_are_probabilities
                else:
                    outputs_are_probabilities = (
                        outputs_are_probabilities and current_outputs_are_probabilities
                    )
            else:
                reordered = self.reorder_logits(
                    logits, tokenized_data["label_classes"]
                )

            all_logits.append(reordered)

        all_logits = np.stack(all_logits)
        if self.classification_type in ["clustering", "clustering-cosine"]:
            probs = _softmax(np.max(all_logits, axis=0), axis=-1)
        elif outputs_are_probabilities:
            probs = np.mean(all_logits, axis=0)
        else:
            probs = np.mean(_softmax(all_logits, axis=-1), axis=0)

        return probs

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> list:
        """Predict the class labels for the provided input dataframe."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            probs.append(self._predict(X.iloc[start:end]))
        probs = np.concatenate(probs, axis=0)

        preds = np.argmax(probs, axis=-1)
        return [self.classes_[p] for p in preds]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input dataframe."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            probs.append(self._predict(X.iloc[start:end]))
        return np.concatenate(probs, axis=0)


class SAP_RPT_OSS_Regressor(RegressorMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = "regression"

    def task_specific_fit(self):
        self.y_ = self.y_.astype(float)

    def _predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable."""
        check_is_fitted(self)

        all_preds = []
        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)
            logits_reg = self._call_model(tokenized_data)
            label_classes = tokenized_data["label_classes"]

            if self.regression_type != "l2" and len(label_classes) != self.num_regression_bins:
                raise ValueError(
                    f"Expected {self.num_regression_bins} classes, got {len(label_classes)}"
                )

            preds, _ = self._extract_prediction_regression(
                logits_reg,
                tokenized_data["data"]["target"],
                tokenized_data["label_classes"],
                target_mean=tokenized_data.get("target_mean"),
                target_std=tokenized_data.get("target_std"),
            )
            all_preds.append(preds)

        return np.mean(np.asarray(all_preds), axis=0)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        preds = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            preds.append(self._predict(X.iloc[start:end]))
        return np.concatenate(preds, axis=0)
