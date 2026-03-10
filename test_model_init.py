import hashlib
import torch
from openverifiablellm.model import ModelConfig, VerifiableLLM

def get_model_hash(model):
    hasher = hashlib.sha256()
    for name, param in model.named_parameters():
        hasher.update(name.encode('utf-8'))
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.hexdigest()

print("Initializing Model 1 with seed 42...")
config = ModelConfig(hidden_size=256, num_hidden_layers=4, num_attention_heads=8, intermediate_size=512)
model1 = VerifiableLLM(config)
model1.init_weights_deterministically(seed=42)
hash1 = get_model_hash(model1)
print(f"Model 1 hash: {hash1}")

print("Initializing Model 2 with seed 42...")
model2 = VerifiableLLM(config)
model2.init_weights_deterministically(seed=42)
hash2 = get_model_hash(model2)
print(f"Model 2 hash: {hash2}")

print("Initializing Model 3 with seed 99...")
model3 = VerifiableLLM(config)
model3.init_weights_deterministically(seed=99)
hash3 = get_model_hash(model3)
print(f"Model 3 hash: {hash3}")

assert hash1 == hash2, "Determinism failed! Hashes differ for the same seed."
assert hash1 != hash3, "Hashes should differ for different seeds."
print("Determinism test PASSED.")
