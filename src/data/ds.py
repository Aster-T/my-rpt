from dataclasses import dataclass

from rules import Rule


@dataclass(slots=True)
class DataConfig:
    rules: Rule = Rule()
    batch_size: int = 32
    shuffle_buffer: int = 1_000
    test_size_min: float = 0.1
    test_size_max: float = 0.9
    seed: int = 42
