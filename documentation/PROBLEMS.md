1. The vocoders selected by the author were not the best.
Whatever the model produces still has to be turned into audio by the vocoder, and the ones shipped with the original
did not sound good enough. They were also trained on 22KHz audio while only covering frequencies only up to 8KHz, so the
whole band between 8 and 11KHz was thrown away before the model even got a chance at it. 
I switched to Vocos, using a model trained on 24KHz audio. It is the only vocoder I found that sounds good and is still
fast enough to be usable, given my hardware constriants. 

2. Matmul precision set to medium or high was affecting duration loss computation so the loss was shooting up at some point
and the model learned "sounds" but not "speech". I've left it to default, which is "highest", and the problem was fixed.

3. Diffusion loss was calculated on padding too, so it was small and progressing very slow, plus, it was wasting a lot of GPU compute power.
I've applied a masking to the data on which it is calculated. This made training as fast as the model compilation did.
I consider this problem fixed.

4. Regular dataloader created "a lot" of padding, so the memory was filled with stuff ignored in gradient computation,
wasting a lot of GPU memory bandwidth.
I've created a data loader which sorts samples by size, and loads similar sized samples in the same batch, so there's less padding.
It's "big", it fits *6 minutes* of audio in one batch. Because of this, I cannot compile the model anymore.
I consider this a great improvement. 

5. Inference was losing a lot of precision by rounding down each phoneme position resulting in metallic sounds padding each phoneme.
The duration estimator estimates phoneme durations. To find positions, it adds up all previous phonemes from the synthesized sentence.
The rounding was applied before summing up. I am now applying it after summing up.   
I think this fixed the problem.

6. Weight decay was not implemented correctly
The original used an optimizer that folds weight decay into the gradient. The optimizer then scales that gradient by its
own running estimate of how steep each weight's history has been, so the amount of decay each weight actually received
depended on its gradient history rather than on the configured decay.
I switched to AdamW, which subtracts the decay from the weights directly, separately from the gradient step. The
configured value now means what it says.

7. Prior loss, though very large in value, was progressing so slowly, and the encoder model was not learning.
I have modified the loss formula from MSE to Huber, which penalizes big errors like MSE, but protects against 
large outliers early in the training. This made training *much more stable* in early stages.
Then did the same for Duration loss. Only the Decoder still uses MSE.
I have introduced metrics that tell me how to choose thew Huber thresholds properly for both prior and duration.

8. The same spectrograms were computed from the audio over and over, every epoch.
The mel spectrograms are a fixed function of the audio files, so recomputing them on every pass through the corpus is
pure waste, and it happens on the critical path while the GPU waits for data.
I compute them once with a script and store them next to the audio. Training reads them from disk.

9. The project implemented its own ODE solver.
I switched to the solver that comes with torch. Less code to maintain, and supports more solvers.

10. The project implemented its own MAS algorithm.
I switched to the Super-MAS library from GitHub; less code to maintain, runs on GPU.

11. The model could not be compiled hurting performance.
I now compile the Encoder and the Decoder with the shapes declared as variable, so one compilation covers all lengths,
and I keep out the few small operations that torch could not handle. Training got about 20% faster and synthesis about
three times faster.

12. The Diffusion Loss gradients were flowing back to the encoder.
The Decoder was able to influence the encoder this way, pushing it in a direction that makes the diffusion job easier. 
But the encoder is supposed to generate a mel that looks as close to the original as possible.
I have detached the encoder output before feeding it into the decoder loss formula. 
The Duration Loss gradients were already detached.

13. The temperature parameter was breaking inference.
During training, the model learned how to produce a mel starting from random noise with a distribution of 1.0
During inference, the temperature param was feeding noise with a different distribution into the model.
The author thought this was going to introduce some variation thus making the speech more natural.
But the model did not see such noise in training, thus it was generating speech that did not sound great.
I have removed the concept of temperature entirely and tht improved the MCD by almost 1dB.

14. One single speaker embedding tensor, shared by all 3 components, but optimized only by encoder gradients.
Encoder, duration predictor, decoder all use same `spk_emb` tensor.
The Duration Predictor had to continuously adapt to the encoder speaker embeddings that continuously change during training.
The fix was to introduce separate speaker embeddings for the duration predictor. Big difference.

15. I have removed the speaker embedding from CFM.
Given the encoder generated mel and the actual ground truth mel as inputs, the CFM can learn to find the velocity field that takes 
one to the other without any speaker conditioning. The optimal path for that does not depend on the speaker.
The encoder mel is almost identical to the ground truth. I have converted it to audio, it sounds like the ground truth, except a 
bit more metallic and a with some rare cracks and pops. The CFM just has to add finer detail to it.
It learns faster now.

16. Separators in between every two phonemes
They are required between voiced phonemes, for a simple reason. In between 2 phoneme sounds, there is always a short
period when a phoneme morphs into the next. It is called a "Formant".
By inserting a separator, we allow the model to assign that transitional sound to something.
But the same separator symbol was used everywhere. The Formants are different, depending on the surrounding phonemes.
They are different both in acoustical content and in duration.
With a single symbol everywhere, the model had a hard time learning the characteristics.
I have modified the phonemization scheme, replacing each voiced phoneme p1 (only vowels and consonants) with a tuple 
of (pre1, p1, post1).

17. MEL frame too long for MAS to be able to find durations
The only Vocos checkpoint I could find uses 24KHz sounds a hop length of 256, resulting in mel frames that are 10.6ms. 
The phonetic representation with 3 symbols per voiced phoneme means the shortest sound cannot be shorted than 31.8ms
which is too long for modeling consonants. MAS was really the weak link in the Matcha solution. 
This trick fixes it entirely. I am now computing the mels with a hop length of 128, reducing the frame duration 
from the 10.6ms to just 5.3ms, and the minimum duration per symbol to 15.9ms. I still have to produce a standard resolution 
mel as I am bound by Vocos, there's no better option fast enough for my needs. I decided to downsample the fine res mel
to a std res mel, and I am doing it before the Decoder, which is the heaviest/slowest model component. Running the Decoder
in fine rest would have doubled the training and inference time, for no good reason. After all, I only needed the fine res
trick for the duration detection bit.
I have a feeling this hasn't been done before in any model. 

18. Kernel size too small to see past the separators
Look at the phonetic representation for this text (the pipe is the separator described above).
    Input text:      <Didn't you?>
    Phonetised text: < |d|ˈ|ɪ|d|n|t| |j|u|ː|?>
If the kernel is 3 (1 symbol on each side) or even 5 (2 symbols on each side), the model will not see the question mark 
after the "u" and will not learn to raise the pitch for interrogative sentences.
I mean, it will, because at the next layer down in the encoder FFN each mixed phoneme will be mixed even more, but with a
a larger kernel the signal will be very strong even at the input.

19. The duration loss was calculated on a log scale, presumably to counter the fact that long phoneme losses would
have had much more weight than the los from short phonemes. 
With a MSE loss, when the estimator predicts 9 frames instead of 8, the loss is 9 ** 2 - 8 ** 2 = 17 but 
if the model predicted 2 instead of 3, the loss would have been just 5. That was making the model much more forgiving with
error on short phonemes. The fix was to add a 2 before calculating the logs, so that all errors have an equal say. 
It has a huge effect, duration estimation loss drops like a rock with this change.

20. Speaker embedding was concatenated into the Encoder input
Update all components to use FiLM, fixed initialization, and generally replaced the original components with more 
modern versions.

21. All 3 losses backpropagated to all subcomponents.  
I have separated them, each one is now detached from the other parts of the model, and is allowed to backpropagate
only to its subcomponent.

22. The Decoder's attention layers did not know how far along the trip they were
How big a correction to make depends on how far along the trip the Decoder is: at the start the mel is mostly noise, and
it should move boldly, near the end it should only polish. That position on the trip is the timestep, and it used to be
handed only to the convolution blocks, which simply added it to what they produced. The attention layers, the part that
lets each frame look at its neighbors, were handed it but quietly ignored it.
The timestep now reaches the attention layers too, as a handful of values that stretch and shift the signal and decide
how much of the attention and the feed-forward result to keep. 

23. No way to test the quality other than by subjective tests.
I've introduced an MCD script I can use to compare the quality of each checkpoint along the way during training
and then to compare current training to previous ones.
I also used that to find the right ODE solver and the right number of steps for inference.
To my surprize, midpoint with 4 steps is better than any other solver and the increasing the number of steps over 
4 actually reduces the MCD score.

24. The prior loss was giving equal importance to frequencies above 7KHz which are barely audible as to those 
at around 2KHz which are very important to speech.   
I've introduced a percentual weighting scheme in the Prior Loss computation. 
I am using a recent white paper, not the original MCD formula, which does not cover all frequencies properly.
I am using the same extended range formula for the MCD test script too.

25. Prior loss was giving more weight to the long phonemes. If many long phonemes sound fine but there are short phonemes
that sound bad, the short one are ignored as they don't contribute much to the prior loss value. It was like the long
phonemes were getting more votes than short ones (one vote per frame).
I've introduced a correction to the prior loss computation, dividing each individual phoneme error by the number of frames
that phoneme spans thus achieving one vote per phoneme.

26. Decoder was spending equal time learning how to get around at any step on the way to destination.
Near the start, the mel is mostly noise, many destinations fit it, and the best possible answer is an average, so
the model does not have a clear decision no matter how much you train. 
Near the destination the answer is almost determined, so the decisions there are easier, no need to train much. 
But uniform step sampling was spending half the budget, and most of the gradient, on those parts that cannot be learned.
I now bias the step sampling towards the middle, where there is something to learn, and the decisions are not easy.

27. Decoder was taking equally long steps along the trajectory from noise to destination 
The solver should take short steps near the beginning of the trajectory and longer ones near the destination. 
This was described in "Sway Sampling", by Chen et al. 2024, F5-TTS.
The coarse structure of the speech is formed during the early steps, so giving the solver more resolution there was 
reported to improve quality at a low number of steps, at no extra cost.