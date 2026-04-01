from typing import Literal

from keras.layers import Dense, Dropout, Embedding, Layer, LayerNormalization

from configs import RobertaConfig
from data.tokenizer import Tokenizer


class DateEmbeddings(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.year_embeddings = Dense(hidden_size, name="year_embeddings")
        self.month_embeddings = Dense(hidden_size, name="month_embeddings")
        self.day_embeddings = Dense(hidden_size, name="day_embeddings")
        self.weekday_embeddings = Dense(hidden_size, name="weekday_embeddings")

    def call(self, data_year_month_day_weekday):
        # date_year_month_day_weekday has shape (num_rows, num_cols, 4)
        year_embeds = self.year_embeddings(data_year_month_day_weekday[:, :, 0])
        month_embeds = self.month_embeddings(data_year_month_day_weekday[:, :, 1])
        day_embeds = self.day_embeddings(data_year_month_day_weekday[:, :, 2])
        weekday_embeds = self.weekday_embeddings(data_year_month_day_weekday[:, :, 3])

        return year_embeds + month_embeds + day_embeds + weekday_embeds


class CellEmbeddings(Layer):
    def __init__(
        self,
        config: RobertaConfig,
        regression_type: Literal["reg-as-classif", "l2"] = "l2",
        is_target_content_mapping: bool = False,
    ):
        super().__init__()
        self.hidden_size = config.hidden_dim

        # Regression
        if regression_type == "l2":
            self.number_embeddings = Dense(config.hidden_dim, name="number_embeddings")
            self.target_embedding_layer_reg = Dense(
                config.hidden_dim, name="target_embedding_layer_reg"
            )
        else:
            self.number_embeddings = Dense(
                Tokenizer.QUANTILE_DIMENSION, name="number_embeddings_as_classification"
            )
            self.target_embedding_layer_reg = Embedding(
                Tokenizer.QUANTILE_DIMENSION,
                config.hidden_dim,
                name="target_embedding_layer_reg_as_classification",
            )

        self.regression_type = regression_type
        self.is_target_content_mapping = is_target_content_mapping

        # Classification
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
