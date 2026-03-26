import dataclasses
import inspect
from dataclasses import dataclass


@dataclass(slots=True)
class RobertaConfig:
    vocabulary_size: int = 50265
    num_layers: int = 12
    hidden_dim: int = 768
    # 占位符，实际值在 __post_init__ 中计算
    intermediate_dim: int = 0
    num_heads: int = 0
    layer_norm_eps: float = 1e-5
    type_vocab_size: int = 1
    dropout: float = 0.1
    max_sequence_length: int = 512
    activation: str = "gelu"

    def __post_init__(self):
        if self.intermediate_dim == 0:
            self.intermediate_dim = self.hidden_dim * 4
        if self.num_heads == 0:
            self.num_heads = self.hidden_dim // 64

    def to_kwargs(self, cls):
        valid = inspect.signature(cls.__init__).parameters.keys()
        return {k: v for k, v in dataclasses.asdict(self).items() if k in valid}
