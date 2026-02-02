Matmul precision set to medium or high was affecting duration loss computation so the loss was shooting up at some point and the model learned "sounds" but not "speech".
I've set it to highest

Diffusion loss was calculated on padding too, so it was small and progressing very slow, plus, it was wasting a lot of GPU compute power.
I've applied a masking to the data on which it is calculated. This made training as fast as the model compilation did.

Regular dataloader created "a lot" of padding, so the memory was filled with stuff ignored in gradient computation, wasting a lot of GPU memory bandwidth.
I've created a data loader which sorts samples by size, and loads similar sized samples in the same batch, so there's less padding.
It's "big", it fits *6 minutes* of audio in one batch. Because of this, I cannot compile the model anymore.

Inference was losing a lot of precision by rounding down each phoneme position resulting in metallic sounds padding each phoneme.
The duration estimator estimates phoneme durations. To find positions, it adds up all previous phonemes from the synthesized sentence.
The rounding was applied before summing up. I am now applying it after summing up.   

Prior loss, though very large in value, was progressing so little (delta 0.0001), and the encoder model was not learning.
I have modified the loss formula so it does not calculate the square of the differences and does not reduce by half.  
This made training *much more stable* in early stages.
Unexpectedly, it also made the *duration loss* learn *much faster*.