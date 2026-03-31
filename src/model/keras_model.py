from typing import Literal

from keras.layers import Dense, Layer

from configs import RobertaConfig
from constant import ModelSize
from src.data.tokenizer import Tokenizer


class RPT(Layer):
    def __init__(
        self,
        model_size: ModelSize,
        regression_type: Literal["reg-as-classif", "l2"] = "l2",
        classification_type="cross-entropy",
        checkpoint_segments=1,
        **kwargs,
    ):
        super().__init__()
        self.model_size = model_size
        self.regression_type = regression_type
        self.classification_type = classification_type
        self.checkpoint_segments = checkpoint_segments
        self.config = RobertaConfig()
        max_number_of_labels = Tokenizer.QUANTILE_DIMENSION

        # classification head
        self.dense_classif = Dense(
            self.config.hidden_dim,
            activation=self.config.activation,
            name="dense_classification",
        )
        self.output_head_classif = Dense(
            max_number_of_labels, activation="softmax", name="classification_head"
        )

        # regression head
        self.dense_reg = Dense(
            self.config.hidden_dim,
            activation=self.config.activation,
            name="dense_regression",
        )
        if self.regression_type == "l2":
            self.output_head_reg = Dense(1, name="l2_head")
        else:
            self.output_head_reg = Dense(
                max_number_of_labels,
                activation="softmax",
                name="regression_as_classification_head",
            )

        assert 0 <= checkpoint_segments <= self.config.num_layers
        self.checkpoint_segments = checkpoint_segments
