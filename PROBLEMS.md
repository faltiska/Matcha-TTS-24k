Matmul precision set to medium or high was affecting duration loss computation so the loss was shooting up at some point and the model learned "sounds" but not "speech".

Diffusion loss was calculated on padding too, so it was small and progressing very slow, plus, it was wasting a lot of GPU compute power.

Regular dataloader created "a lot" of padding, so the memory was filled with stuff ignored in gradient computation, wasting a lot of GPU memory bandwidth.

Inference was losing a lot of precision by rounding down each phoneme position resulting in metallic sounds padding each phoneme.

Prior loss, though very large in value, was progressing so little (delta 0.0001), it was overcome by the other losses and the encoder model was not learning.