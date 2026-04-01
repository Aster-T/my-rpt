# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Literal

from keras import ops
from keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization

try:
    from src.configs import RobertaConfig
    from src.data.tokenizer import Tokenizer
except ImportError:
    from configs import RobertaConfig
    from data.tokenizer import Tokenizer


class DateEmbeddings(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.year_embeddings = Embedding(52, hidden_size, name="year_embeddings")
        self.month_embeddings = Embedding(13, hidden_size, name="month_embeddings")
        self.day_embeddings = Embedding(32, hidden_size, name="day_embeddings")
        self.weekday_embeddings = Embedding(8, hidden_size, name="weekday_embeddings")

    def call(self, date_year_month_day_weekday):
        # date_year_month_day_weekday has shape (num_rows, num_cols, 4)
        date_year_month_day_weekday = ops.cast(
            date_year_month_day_weekday, dtype="int32"
        )
        year_embeds = self.year_embeddings(date_year_month_day_weekday[:, :, 0])
        month_embeds = self.month_embeddings(date_year_month_day_weekday[:, :, 1])
        day_embeds = self.day_embeddings(date_year_month_day_weekday[:, :, 2])
        weekday_embeds = self.weekday_embeddings(date_year_month_day_weekday[:, :, 3])

        return year_embeds + month_embeds + day_embeds + weekday_embeds


class CellEmbeddings(Layer):
    """
    Embedding module for self supervised learning.
    On the input side, it sums four contributions:
    - Numbers
    - Dates
    - Column names
    - String contents
    """

    def __init__(
        self,
        config: RobertaConfig,
        regression_type: Literal["reg-as-classif", "l2"] = "reg-as-classif",
        is_target_content_mapping: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_dim

        if regression_type == "l2":
            self.number_embeddings = Dense(config.hidden_dim, name="number_embeddings")
            self.target_embedding_layer_reg = Dense(
                config.hidden_dim, name="target_embedding_layer_reg"
            )
        else:
            self.number_embeddings = Embedding(
                Tokenizer.QUANTILE_DIMENSION,
                config.hidden_dim,
                name="number_embeddings_as_classification",
            )
            self.target_embedding_layer_reg = Embedding(
                Tokenizer.QUANTILE_DIMENSION,
                config.hidden_dim,
                name="target_embedding_layer_reg_as_classification",
            )

        self.regression_type = regression_type
        self.is_target_content_mapping = is_target_content_mapping

        self.target_embedding_layer_classif = Embedding(
            Tokenizer.QUANTILE_DIMENSION,
            config.hidden_dim,
            name="target_embedding_layer_classif",
        )

        self.date_embeddings = DateEmbeddings(config.hidden_dim)
        self.column_remapping = Dense(config.hidden_dim, name="column_remapping")
        self.content_remapping = Dense(config.hidden_dim, name="content_remapping")
        if self.is_target_content_mapping:
            self.target_content_remapping = Dense(
                config.hidden_dim, name="target_content_remapping"
            )

        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps)
        self.dropout = Dropout(config.dropout)

    def increase_by_one_and_map_negative_to_zero(self, x):
        """
        In dataset, "valid" labels are 0, 1, 2, ... and masked values are -100.
        We want to map them to 1, 2, 3, ... and 0.
        """
        x = ops.cast(x, dtype="int32")
        return ops.where(x < 0, ops.zeros_like(x), x + 1)

    def _zero_last_column(self, x):
        return ops.concatenate([x[:, :-1], ops.zeros_like(x[:, -1:])], axis=1)

    def _pad_last_column(self, reference, last_column):
        return ops.concatenate(
            [ops.zeros_like(reference[:, :-1]), ops.expand_dims(last_column, axis=1)],
            axis=1,
        )

    def call(self, input_dict: Dict, is_regression: bool, training=None):
        is_classification = not is_regression

        if self.regression_type == "l2":
            numbers_normalized = ops.expand_dims(
                input_dict["number_normalized"], axis=-1
            )
            numbers_normalized = ops.cast(
                numbers_normalized, dtype=self.number_embeddings.compute_dtype
            )
            number_embeds = self.number_embeddings(numbers_normalized)
            number_embeds = ops.where(
                numbers_normalized <= -99,
                ops.zeros_like(number_embeds),
                number_embeds,
            )
        else:
            number_perc_floor = ops.cast(
                input_dict["number_percentile_floor"], dtype="int32"
            )
            mask = number_perc_floor > -99
            safe_floor = ops.where(
                mask, number_perc_floor, ops.zeros_like(number_perc_floor)
            )

            number_embeds = self.number_embeddings(safe_floor)
            next_perc = ops.minimum(safe_floor + 1, Tokenizer.QUANTILE_DIMENSION - 1)
            number_embeds_plus_one = self.number_embeddings(next_perc)
            delta = ops.expand_dims(input_dict["number_percentile_delta"], axis=-1)
            delta = ops.cast(delta, dtype=number_embeds.dtype)
            number_embeds = number_embeds * (1 - delta) + number_embeds_plus_one * delta
            number_embeds = ops.where(
                ops.expand_dims(mask, axis=-1),
                number_embeds,
                ops.zeros_like(number_embeds),
            )

        date_embeds = self.date_embeddings(input_dict["date_year_month_day_weekday"])

        column_embeddings = ops.expand_dims(input_dict["column_embeddings"], axis=0)
        column_embeddings = ops.cast(
            column_embeddings, dtype=self.column_remapping.compute_dtype
        )
        column_embeds = self.column_remapping(column_embeddings)

        text_embeddings = input_dict["text_embeddings"]
        target_text_embeddings = text_embeddings[:, -1]
        text_embeddings = self._zero_last_column(text_embeddings)

        text_embeddings = ops.cast(
            text_embeddings, dtype=self.content_remapping.compute_dtype
        )
        content_embeds = self.content_remapping(text_embeddings)
        if self.is_target_content_mapping:
            content_embeds = self._zero_last_column(content_embeds)

        input_embeds = column_embeds + content_embeds + number_embeds + date_embeds

        if is_classification and self.is_target_content_mapping:
            target_text_embeddings = ops.cast(
                target_text_embeddings,
                dtype=self.target_content_remapping.compute_dtype,
            )
            target_content_embeds = self.target_content_remapping(
                target_text_embeddings
            )
            target_embeds = ops.cast(target_content_embeds, dtype=number_embeds.dtype)
        elif is_classification:
            target_values_classif = self.increase_by_one_and_map_negative_to_zero(
                input_dict["target"]
            )
            target_embeds_classif = self.target_embedding_layer_classif(
                target_values_classif
            )
            target_embeds = ops.cast(target_embeds_classif, dtype=number_embeds.dtype)
        else:
            if self.regression_type == "l2":
                target_values_reg = ops.expand_dims(input_dict["target"], axis=-1)
                target_values_reg = ops.cast(
                    target_values_reg,
                    dtype=self.target_embedding_layer_reg.compute_dtype,
                )
                target_embeds_reg = self.target_embedding_layer_reg(target_values_reg)
                target_embeds_reg = ops.where(
                    target_values_reg <= -99,
                    ops.zeros_like(target_embeds_reg),
                    target_embeds_reg,
                )
                target_embeds = ops.cast(target_embeds_reg, dtype=number_embeds.dtype)
            else:
                target_values_reg = self.increase_by_one_and_map_negative_to_zero(
                    input_dict["target"]
                )
                target_embeds_reg = self.target_embedding_layer_reg(target_values_reg)
                target_plus_one_embeds_reg = self.target_embedding_layer_reg(
                    target_values_reg + 1
                )
                delta = ops.expand_dims(input_dict["target_delta"], axis=-1)
                delta = ops.cast(delta, dtype=target_embeds_reg.dtype)
                target_embeds = (
                    target_embeds_reg * (1 - delta) + target_plus_one_embeds_reg * delta
                )
                target_embeds = ops.cast(target_embeds, dtype=number_embeds.dtype)

        input_embeds = input_embeds + self._pad_last_column(
            number_embeds, target_embeds
        )
        input_embeds = self.layer_norm(input_embeds)
        input_embeds = self.dropout(input_embeds, training=training)
        return input_embeds
