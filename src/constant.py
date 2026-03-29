import argparse
from enum import Enum

SENTENCE_EMBEDDING_MODEL_NAME_DEFAULT = (
    "flax-sentence-embeddings/all_datasets_v3_mpnet-base"
)

embedding_model_to_dimension_and_pooling = {
    SENTENCE_EMBEDDING_MODEL_NAME_DEFAULT: (768, "mean"),
}

QUANTILE_DIMENSION_DEFAULT = 64


class ModelSize(Enum):
    # The two values are the number of layers and the hidden size
    tiny = (2, 128)
    mini = (4, 256)
    small = (4, 512)
    medium = (8, 512)
    base = (12, 768)
    large = (24, 1024)
    xlarge = (24, 2048)


class ModelSizeAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values not in ModelSize.__members__ or not isinstance(values, str):
            raise ValueError(
                f"{values} is not a valid value for ModelSize: {ModelSize.__members__.keys()}"
            )
        value = ModelSize[values]
        setattr(namespace, self.dest, value)
