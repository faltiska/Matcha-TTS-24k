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

8. One single speaker embedding tensor, shared by all 3 components, but optimized only by encoder gradients.
Encoder, duration predictor, decoder all use same `spk_emb` vector
The Duration Predictor had to continuously adapt to the encoder speaker embeddings that continuously change during training.
The fix was to introduce separate speaker embeddings for the duration predictor. 
It learns faster now.

9. I have removed the speaker embedding from CFM.
Given the encoder generated mel and the actual ground truth mel as inputs, the CFM can learn to find the velocity field that takes 
one to the other without any speaker conditioning. The optimal path for that does not depend on the speaker.
The encoder mel is almost identical to the ground truth. I have converted it to audio, it sounds like the ground truth, except a 
bit more metallic and a with some rare cracks and pops. The CFM just has to add finer detail to it.
It learns faster now.

10. Separators inserted in between annotation symbols and the annotated phonemes
They are required between voiced phonemes, for a simple reason. In between 2 phoneme sounds, there is always a short
period when a phoneme morphs into the next. That does not sound like the previous phoneme or the next.
By inserting a separator, we allow the model to assign that transitional sound to something.
But some symbols produced by eSpeak are just annotations, like the stress marker that means "put an emphasis on the next 
vowel", or the duration annotation that means "elongate the previous vowel", like in this example: "ˈɔː". 
We do not need separators between those.
Because of how mel bins work, there is a minimum mel duration for any symbol. In my case, the minimum is 11ms. 
That separator between a stress symbol and its vowel will be at least 11ms long.
I added a method that groups annotations with their phonemes. 
Rules are extremely complex, I have a feeling this hasn't been done before. 

11. Kernel size too small to see past the separators
Look at the phonetic representation for this text (the pipe is the separator described above).
    Input text:      <Didn't you?>
    Phonetised text: < |d|ˈ|ɪ|d|n|t| |j|u|ː|?>
If the kernel is 3 (1 symbol on each side) or even 5 (2 symbols on each side), the model will not see the question mark 
after the "u" and will not learn to raise the pitch for interrogative sentences.
I mean, it will, because at the next layer down in the encoder FFN each mixed phoneme will be mixed even more, but with a
a larger kernel the signal will be very strong even at the input.

12. The duration loss was calculated on a log scale, presumably to counter the fact that long phoneme losses would
have had much more weight than the los from short phonemes. It uses MSE loss and when the estimator predicts 9 frames 
instead of 8, the loss is 9 ** 2 - 8 ** 2 = 17 but if the model predicted 2 instead of 3, the loss would have been just 5
making the model much more forgiving with errors on short phonemes.
But the author did not realize ln(2) = 0.69 and the MSE losses are really forgiving with subunitary numbers.
The fix was to add a 2 before calculating the logs, since ln(3) > 1. 
It has a huge effect, duration estimation loss drops like a rock with this change.