import os

# Keep Triton FMA fusion enabled (Triton 3.10 flips the default to disabled).
# Must be set before torch/triton are imported; package __init__ runs before any matcha.* entry point.
os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
