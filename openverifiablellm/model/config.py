from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 1024       # smaller hidden size for tiny verifiable setup (~1B parameters config scalable)
    intermediate_size: int = 2816
    num_hidden_layers: int = 22
    num_attention_heads: int = 16
    num_key_value_heads: Optional[int] = 4  # GQA
    max_position_embeddings: int = 2048
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    initializer_range: float = 0.02

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
