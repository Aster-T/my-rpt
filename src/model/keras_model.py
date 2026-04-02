# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
from keras import ops
from keras.activations import gelu
from keras.layers import Dense, Layer

try:
    import torch
except ImportError:
    torch = None

try:
    from src.configs import RobertaConfig
    from src.constant import ModelSize
    from src.data.tokenizer import Tokenizer
    from src.model.attention import TwoDimensionAttentionLayer
    from src.model.embeddings import CellEmbeddings
except ImportError:
    from configs import RobertaConfig
    from constant import ModelSize
    from data.tokenizer import Tokenizer
    from model.attention import TwoDimensionAttentionLayer
    from model.embeddings import CellEmbeddings


def _to_numpy(x):
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "numpy"):
        return np.asarray(x.numpy())
    return np.asarray(x)


def _binary_cross_entropy(probs, targets):
    probs = ops.clip(probs, x_min=1e-7, x_max=1.0 - 1e-7)
    return -(targets * ops.log(probs) + (1.0 - targets) * ops.log(1.0 - probs))


def _binary_cross_entropy_with_logits(logits, targets):
    zeros = ops.zeros_like(logits)
    relu_logits = ops.maximum(logits, zeros)
    return relu_logits - logits * targets + ops.log1p(ops.exp(-ops.abs(logits)))


class RPT(Layer):
    """RPT (sap-rpt-oss) model class."""

    def __init__(
        self,
        model_size: ModelSize,
        regression_type: Literal["reg-as-classif", "l2"] = "reg-as-classif",
        classification_type: Literal[
            "cross-entropy", "clustering", "clustering-cosine"
        ] = "cross-entropy",
        checkpointing_segments=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        num_layers, hidden_size = model_size.value
        self.config = RobertaConfig(
            num_layers=num_layers,
            hidden_dim=hidden_size,
            intermediate_dim=hidden_size * 4,
            num_heads=hidden_size // 64,
            layer_norm_eps=1e-5,
            type_vocab_size=1,
            dropout=0.1,
        )
        self.regression_type = regression_type
        self.classification_type = classification_type
        max_number_of_labels = Tokenizer.QUANTILE_DIMENSION

        if self.classification_type in ["clustering", "clustering-cosine"]:
            self.cluster_dense = Dense(self.config.hidden_dim, name="cluster_dense")
            self.cluster_out_dim = self.config.hidden_dim
            self.cluster_output_head = Dense(
                self.cluster_out_dim, name="cluster_output_head"
            )
        else:
            self.dense_classif = Dense(
                self.config.hidden_dim, name="dense_classification"
            )
            self.output_head_classif = Dense(
                max_number_of_labels, name="classification_head"
            )

        self.dense_reg = Dense(self.config.hidden_dim, name="dense_regression")
        if self.regression_type == "l2":
            self.output_head_reg = Dense(1, name="l2_head")
        else:
            self.output_head_reg = Dense(
                max_number_of_labels, name="regression_as_classification_head"
            )

        assert 0 <= checkpointing_segments <= self.config.num_layers
        self.checkpointing_segments = checkpointing_segments

        self.embeddings = CellEmbeddings(
            self.config,
            regression_type=regression_type,
            is_target_content_mapping=(classification_type != "cross-entropy"),
        )
        self.in_context_encoder = [
            TwoDimensionAttentionLayer(self.config)
            for _ in range(self.config.num_layers)
        ]

    @staticmethod
    def build_context_attention_mask(data):
        """
        Builds a context attention mask of shape (num_rows, num_rows).
        True means "can attend", False means "masked out".
        """
        target = data["target"]
        if target.ndim != 1:
            raise ValueError(
                f"Expected target to be a 1D tensor, got shape: {target.shape}"
            )

        num_rows = ops.shape(target)[0]
        row_ids = ops.arange(num_rows)
        diagonal = ops.equal(
            ops.expand_dims(row_ids, axis=1),
            ops.expand_dims(row_ids, axis=0),
        )
        context_rows = target > -99
        context_columns = ops.broadcast_to(
            ops.expand_dims(context_rows, axis=0),
            (num_rows, num_rows),
        )
        return ops.logical_or(diagonal, context_columns)

    @staticmethod
    def compute_classif_loss_and_metric(logits, labels, train_target):
        labels = ops.cast(labels, dtype="int32")
        is_test_mask = train_target <= -99
        mask = ops.cast(is_test_mask, dtype=logits.dtype)
        denominator = ops.maximum(ops.sum(mask), ops.cast(1.0, logits.dtype))

        safe_labels = ops.where(is_test_mask, labels, ops.zeros_like(labels))
        num_classes = ops.shape(logits)[-1]
        label_one_hot = ops.one_hot(safe_labels, num_classes)
        log_probs = logits - ops.logsumexp(logits, axis=-1, keepdims=True)
        per_row_loss = -ops.sum(label_one_hot * log_probs, axis=-1)
        loss_classif = ops.sum(per_row_loss * mask) / denominator

        prediction = ops.argmax(logits, axis=-1)
        accuracy = (
            ops.sum(
                ops.cast(ops.equal(prediction, safe_labels), dtype=logits.dtype) * mask
            )
            / denominator
        )

        context_mask = train_target > -99
        safe_context_targets = ops.where(
            context_mask,
            ops.cast(train_target, dtype="int32"),
            ops.zeros_like(ops.cast(train_target, dtype="int32")),
        )
        context_one_hot = ops.one_hot(
            safe_context_targets, Tokenizer.QUANTILE_DIMENSION
        )
        context_one_hot = context_one_hot * ops.cast(
            ops.expand_dims(context_mask, axis=-1), dtype=context_one_hot.dtype
        )
        dummy_prediction = ops.argmax(ops.sum(context_one_hot, axis=0))
        dummy_accuracy = (
            ops.sum(
                ops.cast(
                    ops.equal(safe_labels, ops.cast(dummy_prediction, dtype="int32")),
                    dtype=logits.dtype,
                )
                * mask
            )
            / denominator
        )
        metric_classif = ops.clip(
            (accuracy - dummy_accuracy) / (1.0 - dummy_accuracy + 1e-5),
            x_min=0.0,
            x_max=1.0,
        )
        return loss_classif, metric_classif

    @staticmethod
    def memory_efficient_cosine_similarity(x, batch_size=1000):
        """
        Computes cosine similarity between all pairs of vectors in x.
        """
        del batch_size
        squared_norm = ops.sum(ops.square(x), axis=1, keepdims=True)
        x_normalized = x / ops.sqrt(ops.clip(squared_norm, x_min=1e-12, x_max=None))
        return ops.matmul(x_normalized, ops.transpose(x_normalized))

    def forward_clustering_head(
        self,
        encoder_outputs,
        out_layer_1,
        out_layer_2,
        use_cosine_similarity=False,
    ):
        cluster_out = out_layer_1(encoder_outputs)
        cluster_out = gelu(cluster_out)
        cluster_out = out_layer_2(cluster_out)

        if use_cosine_similarity:
            out_clustering = self.memory_efficient_cosine_similarity(cluster_out)
        else:
            out_clustering = ops.matmul(cluster_out, ops.transpose(cluster_out))
        return out_clustering

    @staticmethod
    def compute_clustering_output_loss_and_metric(
        logits,
        labels,
        train_target,
        is_mask_out_context=False,
        is_cosine_similarity=False,
    ):
        adjacency_matrices = ops.cast(
            ops.equal(
                ops.expand_dims(labels, axis=-1), ops.expand_dims(labels, axis=-2)
            ),
            dtype=logits.dtype,
        )

        if is_cosine_similarity:
            loss_cluster = _binary_cross_entropy(
                ops.clip(logits, x_min=0.0, x_max=1.0),
                adjacency_matrices,
            )
        else:
            loss_cluster = _binary_cross_entropy_with_logits(logits, adjacency_matrices)

        if is_mask_out_context:
            is_context = train_target > -99
            off_diagonal_mask = ops.cast(
                ops.logical_and(
                    ops.expand_dims(is_context, axis=-1),
                    ops.logical_not(ops.expand_dims(is_context, axis=-2)),
                ),
                dtype=loss_cluster.dtype,
            )
            loss_cluster = loss_cluster * off_diagonal_mask
            denominator = ops.maximum(
                ops.sum(off_diagonal_mask), ops.cast(1.0, loss_cluster.dtype)
            )
        else:
            denominator = ops.cast(ops.size(loss_cluster), dtype=loss_cluster.dtype)

        loss_cluster = ops.sum(loss_cluster) / denominator
        loss_cluster = loss_cluster * ops.cast(3.0, dtype=loss_cluster.dtype)

        if is_cosine_similarity:
            out_clustering = logits
        else:
            out_clustering = ops.sigmoid(logits)
        out_clustering = (
            out_clustering + ops.transpose(out_clustering, axes=(1, 0))
        ) / 2.0

        metric_cluster = ops.cast(out_clustering > 0.5, dtype="int32")
        adjacency_int = ops.cast(adjacency_matrices, dtype="int32")
        metric_mask = ops.logical_or(metric_cluster == 1, adjacency_int == 1)
        metric_mask_f = ops.cast(metric_mask, dtype=logits.dtype)
        metric_denominator = ops.maximum(
            ops.sum(metric_mask_f), ops.cast(1.0, logits.dtype)
        )
        metric_cluster = (
            ops.sum(
                ops.cast(ops.equal(metric_cluster, adjacency_int), dtype=logits.dtype)
                * metric_mask_f
            )
            / metric_denominator
        )
        return out_clustering, loss_cluster, metric_cluster

    def compute_regression_output_loss_and_metric(self, logits, labels, train_target):
        if self.regression_type == "reg-as-classif":
            loss_reg, metric_reg = self.compute_classif_loss_and_metric(
                logits, labels, train_target
            )
        else:
            if logits.shape[-1] == 1:
                logits = ops.squeeze(logits, axis=-1)
            test_mask = train_target <= -99
            mask = ops.cast(test_mask, dtype=logits.dtype)
            denominator = ops.maximum(ops.sum(mask), ops.cast(1.0, logits.dtype))

            labels = ops.cast(labels, dtype=logits.dtype)
            masked_labels = ops.where(test_mask, labels, ops.zeros_like(labels))
            masked_logits = ops.where(test_mask, logits, ops.zeros_like(logits))
            masked_logits = ops.where(
                ops.isfinite(masked_logits),
                masked_logits,
                ops.zeros_like(masked_logits),
            )

            squared_error = ops.square(masked_logits - masked_labels) * mask
            loss_reg = ops.sum(squared_error) / denominator
            loss_reg = ops.clip(loss_reg, x_min=0.0, x_max=10.0)

            label_mean = ops.sum(masked_labels) / denominator
            ss_res = ops.sum(squared_error)
            ss_tot = ops.sum(ops.square(masked_labels - label_mean) * mask)
            metric_reg = ops.where(
                ss_tot > 0,
                1.0 - (ss_res / ss_tot),
                ops.cast(0.0, dtype=logits.dtype),
            )
            metric_reg = ops.clip(metric_reg, x_min=-1.0, x_max=1.0)
        return logits, loss_reg, metric_reg

    def forward_heads(
        self,
        encoder_outputs,
        is_regression: bool,
        labels: Optional[object] = None,
        target: Optional[object] = None,
        target_delta: Optional[object] = None,
    ):
        """
        Last part of the "forward" method.
        It takes the encoder outputs (one token per row) and applies the heads
        and losses (if labels are provided).
        """
        is_classification = not is_regression

        if is_classification:
            if self.classification_type in ["clustering", "clustering-cosine"]:
                use_cosine_similarity = self.classification_type == "clustering-cosine"
                out = self.forward_clustering_head(
                    encoder_outputs,
                    self.cluster_dense,
                    self.cluster_output_head,
                    use_cosine_similarity=use_cosine_similarity,
                )
            else:
                out = self.dense_classif(encoder_outputs)
                out = gelu(out)
                out = self.output_head_classif(out)
        else:
            out = self.dense_reg(encoder_outputs)
            out = gelu(out)
            out = self.output_head_reg(out)

        if labels is None:
            if is_classification:
                if self.classification_type == "clustering":
                    out = ops.sigmoid(out)
                if self.classification_type in ["clustering", "clustering-cosine"]:
                    out = (out + ops.transpose(out, axes=(1, 0))) / 2.0
                if self.regression_type == "l2":
                    out = ops.squeeze(out, axis=-1) if out.shape[-1] == 1 else out
            return out

        if target is None:
            raise ValueError("target must be provided when labels are provided")

        if is_classification:
            if self.classification_type in ["clustering", "clustering-cosine"]:
                out, loss, metric = self.compute_clustering_output_loss_and_metric(
                    out,
                    labels,
                    target,
                    is_cosine_similarity=self.classification_type
                    == "clustering-cosine",
                )
            else:
                loss, metric = self.compute_classif_loss_and_metric(out, labels, target)
        else:
            if target_delta is None:
                raise ValueError(
                    "target_delta must be provided for regression when labels are provided"
                )
            real_target = ops.cast(ops.round(target + target_delta), dtype="int32")
            out, loss, metric = self.compute_regression_output_loss_and_metric(
                out, labels, real_target
            )

        return out, loss, metric

    @staticmethod
    def copy_last_layer_weights_to_all(state_dict):
        encoder_layers = [
            key for key in state_dict.keys() if "in_context_encoder" in key
        ]

        layer_numbers = []
        for key in encoder_layers:
            match = re.search(r"in_context_encoder\.(\d+)", key)
            if match:
                layer_numbers.append(int(match.group(1)))
        if not layer_numbers:
            return state_dict

        last_layer_num = max(layer_numbers)
        for key in list(state_dict.keys()):
            if f"in_context_encoder.{last_layer_num}." in key:
                for layer_idx in range(last_layer_num):
                    state_dict[
                        key.replace(
                            f"in_context_encoder.{last_layer_num}.",
                            f"in_context_encoder.{layer_idx}.",
                        )
                    ] = state_dict[key]
        return state_dict

    @staticmethod
    def _extract_state_dict(checkpoint: object) -> dict[str, object]:
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Checkpoint must be a mapping, got {type(checkpoint).__name__}"
            )

        state_dict = checkpoint.get("state_dict", checkpoint)
        if not isinstance(state_dict, Mapping):
            raise TypeError(
                "Checkpoint 'state_dict' entry must be a mapping of parameter names "
                f"to tensors, got {type(state_dict).__name__}"
            )

        return dict(state_dict)

    @staticmethod
    def _normalize_state_dict_keys(
        state_dict: Mapping[str, object],
    ) -> dict[str, object]:
        normalized_state_dict: dict[str, object] = {}

        for key, value in state_dict.items():
            normalized_key = key
            previous_key = None
            while normalized_key != previous_key:
                previous_key = normalized_key
                for prefix in ("module.", "model."):
                    if normalized_key.startswith(prefix):
                        normalized_key = normalized_key.removeprefix(prefix)
            normalized_state_dict[normalized_key] = value

        return normalized_state_dict

    @staticmethod
    def _load_checkpoint(checkpoint_path: Union[str, Path]) -> Mapping[str, object]:
        if torch is None:
            raise ImportError(
                "torch is required to inspect PyTorch checkpoints such as .pt or .ckpt files."
            )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Checkpoint must be a mapping, got {type(checkpoint).__name__}"
            )
        return checkpoint

    @classmethod
    def inspect_checkpoint(
        cls, checkpoint_path: Union[str, Path]
    ) -> dict[str, Union[ModelSize, str]]:
        checkpoint = cls._load_checkpoint(checkpoint_path)
        state_dict = cls._normalize_state_dict_keys(cls._extract_state_dict(checkpoint))
        hyper_parameters = checkpoint.get("hyper_parameters", {})

        model_size = None
        if isinstance(hyper_parameters, Mapping):
            hyper_model_size = hyper_parameters.get("model_size")
            if isinstance(hyper_model_size, ModelSize):
                model_size = hyper_model_size
            elif (
                isinstance(hyper_model_size, str)
                and hyper_model_size in ModelSize.__members__
            ):
                model_size = ModelSize[hyper_model_size]
        if model_size is None:
            model_size = cls._infer_model_size_from_state_dict(state_dict)

        regression_type = "l2"
        inferred_regression_type = cls._infer_regression_type_from_state_dict(
            state_dict
        )
        if isinstance(hyper_parameters, Mapping):
            hyper_regression_type = hyper_parameters.get("regression_type")
            if hyper_regression_type in {"l2", "reg-as-classif"}:
                regression_type = hyper_regression_type
            elif inferred_regression_type == "reg-as-classif":
                regression_type = "reg-as-classif"
        elif inferred_regression_type == "reg-as-classif":
            regression_type = "reg-as-classif"

        classification_type = "cross-entropy"
        inferred_classification_type = cls._infer_classification_type_from_state_dict(
            state_dict
        )
        if isinstance(hyper_parameters, Mapping):
            hyper_classification_type = hyper_parameters.get("classification_type")
            if hyper_classification_type in {
                "cross-entropy",
                "clustering",
                "clustering-cosine",
            }:
                classification_type = hyper_classification_type
            elif inferred_classification_type != "cross-entropy":
                classification_type = inferred_classification_type
        elif inferred_classification_type != "cross-entropy":
            classification_type = inferred_classification_type

        return {
            "model_size": model_size,
            "regression_type": regression_type,
            "classification_type": classification_type,
        }

    @staticmethod
    def _infer_model_size_from_state_dict(
        state_dict: Mapping[str, object],
    ) -> ModelSize:
        layer_numbers = []
        for key in state_dict:
            match = re.search(r"in_context_encoder\.(\d+)", key)
            if match:
                layer_numbers.append(int(match.group(1)))

        if not layer_numbers:
            raise ValueError("Could not infer model depth from checkpoint state_dict")
        num_layers = max(layer_numbers) + 1

        hidden_size = None
        for key in (
            "dense_reg.weight",
            "dense_classif.weight",
            "cluster_dense.weight",
            "output_head_reg.weight",
        ):
            tensor = state_dict.get(key)
            if tensor is None:
                continue
            shape = getattr(tensor, "shape", None)
            if shape is not None and len(shape) >= 2:
                hidden_size = int(shape[1])
                break

        if hidden_size is None:
            raise ValueError("Could not infer hidden size from checkpoint state_dict")

        for model_size in ModelSize:
            if model_size.value == (num_layers, hidden_size):
                return model_size

        raise ValueError(
            "Checkpoint architecture does not match any known ModelSize: "
            f"num_layers={num_layers}, hidden_size={hidden_size}"
        )

    @staticmethod
    def _infer_regression_type_from_state_dict(state_dict: Mapping[str, object]) -> str:
        output_head = state_dict.get("output_head_reg.weight")
        shape = getattr(output_head, "shape", None)
        if shape is None or len(shape) < 2:
            return "l2"
        return "l2" if int(shape[0]) == 1 else "reg-as-classif"

    @staticmethod
    def _infer_classification_type_from_state_dict(
        state_dict: Mapping[str, object],
    ) -> str:
        if (
            "cluster_dense.weight" in state_dict
            or "cluster_output_head.weight" in state_dict
        ):
            return "clustering"
        return "cross-entropy"

    def load_weights(
        self,
        checkpoint_path: Union[str, Path],
        device=None,
        is_copy_last_layer=True,
    ):
        """
        Keras version placeholder for checkpoint loading.

        We support checkpoint inspection for PyTorch `.pt`/`.ckpt` files, but direct
        parameter transplantation into the current Keras architecture is not
        implemented yet because the Keras layer naming/layout does not currently
        match the original PyTorch `state_dict` 1:1.
        """
        del device, is_copy_last_layer
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.suffix in {".pt", ".ckpt"}:
            raise NotImplementedError(
                "Loading PyTorch checkpoints into the Keras 3 RPT model is not "
                "implemented yet. `inspect_checkpoint()` works, but the actual "
                "weight-name and tensor-layout conversion still needs to be added."
            )
        raise NotImplementedError(
            f"Unsupported checkpoint format for Keras RPT: {checkpoint_path}"
        )

    def extract_prediction_classification(
        self, logits, targets, label_classes: np.ndarray
    ):
        logits = _to_numpy(logits)
        targets = _to_numpy(targets)
        test_mask = targets <= -99

        if self.classification_type in ["clustering", "clustering-cosine"]:
            return self._extract_prediction_clustering(
                logits, targets, test_mask, label_classes
            )

        test_logits = logits[test_mask][:, : len(label_classes)]
        test_preds_indices = np.argmax(test_logits, axis=-1)
        test_preds = label_classes[test_preds_indices]
        return test_preds, test_logits

    def extract_prediction_regression(
        self,
        logits,
        targets,
        label_classes: Union[np.ndarray, object],
        target_mean: Optional[object] = None,
        target_std: Optional[object] = None,
    ):
        logits = _to_numpy(logits)
        targets = _to_numpy(targets)
        label_classes = _to_numpy(label_classes)
        test_mask = targets <= -99

        if self.regression_type == "reg-as-classif":
            test_logits = logits[test_mask]
            test_scores = test_logits[:, : len(label_classes)]
            test_scores = test_scores - np.max(test_scores, axis=1, keepdims=True)
            test_probas = np.exp(test_scores)
            test_probas = test_probas / np.sum(test_probas, axis=1, keepdims=True)
            test_preds = test_probas @ label_classes
        else:
            if target_mean is None or target_std is None:
                raise ValueError(
                    "target_mean and target_std are required for l2 regression."
                )
            test_logits = logits[test_mask]
            if test_logits.ndim > 1 and test_logits.shape[-1] == 1:
                test_logits = np.squeeze(test_logits, axis=-1)
            test_preds = np.asarray(
                test_logits * target_std + target_mean, dtype=np.float32
            )
            test_probas = None
        return test_preds, test_probas

    @staticmethod
    def _extract_prediction_clustering(
        similarities,
        targets,
        test_mask,
        label_classes: np.ndarray,
    ):
        """
        similarities has shape (num_rows, num_rows) and contains similarities
        between all pairs of rows.
        """
        similarities = _to_numpy(similarities)
        targets = _to_numpy(targets).astype(np.int64, copy=False)
        label_classes = np.asarray(label_classes)

        context_mask = ~test_mask
        targets_for_context = targets[context_mask]
        similarities_masked = similarities[test_mask][:, context_mask]

        queries_num = similarities_masked.shape[0]
        test_similarities = np.full(
            (queries_num, len(label_classes)),
            float("-inf"),
            dtype=similarities_masked.dtype,
        )

        for class_index in range(len(label_classes)):
            class_mask = targets_for_context == class_index
            if np.any(class_mask):
                test_similarities[:, class_index] = np.max(
                    similarities_masked[:, class_mask], axis=1
                )

        test_preds = label_classes[np.argmax(test_similarities, axis=1)]
        test_logits = np.log(np.clip(test_similarities, 1e-6, 1 - 1e-6))
        test_logits -= np.log1p(-np.clip(test_similarities, 1e-6, 1 - 1e-6))
        test_logits = np.clip(
            np.nan_to_num(test_logits, nan=-1e4, neginf=-1e4, posinf=1e4),
            -1e4,
            1e4,
        )
        return test_preds, test_logits

    def call(
        self,
        data: dict[str, object],
        is_regression: bool,
        labels=None,
        training=None,
        **kwargs,
    ):
        del kwargs
        input_embeds = self.embeddings(
            data,
            is_regression=is_regression,
            training=training,
        )
        attention_mask = self.build_context_attention_mask(data)
        column_mask = data.get("column_mask")

        encoder_outputs = input_embeds
        for layer in self.in_context_encoder:
            encoder_outputs = layer(
                encoder_outputs,
                attention_mask=attention_mask,
                column_mask=column_mask,
            )

        target_column_output = encoder_outputs[:, -1]
        target_delta = data.get("target_delta")

        return self.forward_heads(
            target_column_output,
            is_regression,
            labels,
            data["target"],
            target_delta,
        )
