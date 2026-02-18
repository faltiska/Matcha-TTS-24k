1. Matmul precision set to medium or high was affecting duration loss computation so the loss was shooting up at some point and the model learned "sounds" but not "speech".
I've left it to default, which is "highest", and the problem was fixed.

2. Diffusion loss was calculated on padding too, so it was small and progressing very slow, plus, it was wasting a lot of GPU compute power.
I've applied a masking to the data on which it is calculated. This made training as fast as the model compilation did.
I consider this problem fixed.

3. Regular dataloader created "a lot" of padding, so the memory was filled with stuff ignored in gradient computation, wasting a lot of GPU memory bandwidth.
I've created a data loader which sorts samples by size, and loads similar sized samples in the same batch, so there's less padding.
It's "big", it fits *6 minutes* of audio in one batch. Because of this, I cannot compile the model anymore.
I consider this a great improvement. 

4. Inference was losing a lot of precision by rounding down each phoneme position resulting in metallic sounds padding each phoneme.
The duration estimator estimates phoneme durations. To find positions, it adds up all previous phonemes from the synthesized sentence.
The rounding was applied before summing up. I am now applying it after summing up.   
I think this fixed the problem.

5. Prior loss, though very large in value, was progressing so little (delta 0.0001), and the encoder model was not learning.
I have modified the loss formula so it does not calculate the square of the differences and does not reduce by half.  
This made training *much more stable* in early stages.
Unexpectedly, it also made the *duration loss* learn *much faster*.
By itself, this change is probably good.

6. The Diffusion Loss gradients were flowing back to the encoder.
The Decoder was able to influence the encoder this way, pushing it in a direction that makes the diffusion job easier. But the encoder is supposed to generate a mel that looks as close to the original as possible.
I have detached the encoder output before feeding it into the decoder loss formula. 
The Duration Loss gradients were already detached.

7. The temperature parameter was breaking inference.
During training, the model learned how to produce a mel starting from random noise with a distribution of 1.0
During inference, the temperature param was feeding noise with a different distribution into the model.
The author thought this was going to introduce some variation thus making the speech more natural.
But the model did not see such noise in training, thus it was generating speech that did not sound great.
I have removed the concept of temperature entirely and tht improved the MCD by almost 1dB. 