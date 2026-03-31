from typing import Literal

from keras.layers import Dense, Layer

from configs import RobertaConfig


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
