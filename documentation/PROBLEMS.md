1. The vocoders selected by the author were not the best.
Whatever the model produces still has to be turned into audio by the vocoder, and the ones shipped with the original
did not sound good enough. They were also trained on 22KHz audio while only covering frequencies only up to 8KHz, so the
whole band between 8 and 11KHz was thrown away before the model even got a chance at it. 
I switched to Vocos, using a model trained on 24KHz audio. It is the only vocoder I found that sounds good and is still
fast enough to be usable, given my hardware constraints. 

2. Diffusion loss was calculated on padding too, so it was progressing very slow, and wasting a lot of GPU compute power.
I've applied a masking to the data on which it is calculated. This made training as fast as the model compilation did.
I consider this problem fixed.

3. Regular dataloader created "a lot" of padding, so the memory was filled with stuff ignored in gradient computation,
wasting a lot of GPU memory bandwidth.
I've created a data loader which sorts samples by size, and loads similar sized samples in the same batch, so there's
less padding. It's "big", it fits *6 minutes* of audio in one batch. I consider this a great improvement. 

4. Inference was losing a lot of precision by rounding up each phoneme position.
The duration estimator estimates phoneme durations. To find positions, it adds up all previous phonemes from the
synthesized sentence. I am now using round() instead of ceil(), and enforcing a minimum of 1 frame per phoneme. 
I think this fixed the problem.

5. Trainer was using plain Adam
I switched to AdamW.

6. Prior loss, though very large in value, was progressing so slowly, and the encoder model was not learning.
I have modified the loss formula from MSE to Huber, which penalizes big errors like MSE, but protects against 
large outliers early in the training. This made training *much more stable* in early stages.
Then did the same for Duration loss. Only the Decoder still uses MSE.
I have introduced metrics that tell me how to choose the Huber thresholds properly for both prior and duration.

7. The same spectrograms were computed from the audio over and over, every epoch.
The mel spectrograms are a fixed function of the audio files, so recomputing them on every pass through the corpus is
pure waste, and it happens on the critical path while the GPU waits for data.
I compute them once with a script and store them next to the audio. Training reads them from disk.

8. The project implemented its own ODE solver.
I switched to the solver that comes from the torchdiffeq library. Less code to maintain, and supports more solvers.

9. The project implemented its own MAS algorithm.
I switched to the Super-MAS library from GitHub; less code to maintain, runs on GPU.

10. The model could not be compiled.
I now compile the Encoder and the Decoder with the shapes declared as variable, so one compilation covers all lengths,
and I keep out the few small operations that torch could not handle. Training got about 20% faster and synthesis about
three times faster.

11. The Diffusion Loss gradients were flowing back to the encoder.
The Decoder was able to influence the encoder this way, pushing it in a direction that makes the diffusion job easier. 
But the encoder is supposed to generate a mel that looks as close to the original as possible.
I have detached the encoder output before feeding it into the decoder loss formula. 
The Duration Loss gradients were already detached.

12. The temperature parameter was breaking inference.
During training, the model learned how to produce a mel starting from random noise with a distribution of 1.0
During inference, the temperature param was feeding noise with a different distribution into the model.
The author thought this was going to introduce some variation thus making the speech more natural.
But the model did not see such noise in training, thus it was generating speech that did not sound great.
I have removed the concept of temperature entirely and atht improved the MCD by almost 1dB.

13. One single speaker embedding tensor, shared by all 3 components.
Encoder, duration predictor, decoder all use same `spk_emb` tensor.
The Duration Predictor had to continuously adapt to the encoder speaker embeddings that continuously change during training.
The fix was to introduce separate speaker embeddings for the duration predictor. 

14. I have removed the speaker embedding from CFM.
Given the encoder generated mel and the actual ground truth mel as inputs, the CFM can learn to find the velocity field  
that takes one to the other without any speaker conditioning. The optimal path for that does not depend on the speaker.
The encoder mel is almost identical to the ground truth. I have converted it to audio, it sounds like the ground truth,  
except a bit more metallic. That means Encoder mel is rich enough, containing all the speaker characteristics the Decoder 
needs to generate the final mel. The Decoder just has to add finer details to it. 
Decoder learns faster now.

15. Separators in between every two phonemes
They are required between voiced phonemes, for a simple reason. In between 2 phoneme sounds, there is always a short
period when a phoneme morphs into the next. It is called a "Formant".
By inserting a separator, we allow the model to assign that transitional sound to something.
But the same separator symbol was used everywhere. The Formants are different, depending on the surrounding phonemes.
They are different both in acoustical content and in duration.
With a single symbol everywhere, the model had a hard time learning the characteristics.
I have modified the phonemization scheme, replacing each voiced phoneme p1 (only vowels and consonants) with a tuple 
of (pre1, p1, post1).

16. MEL frame too long for MAS to be able to find durations
The only Vocos checkpoint I could find uses 24KHz and a hop length of 256, resulting in mel frames that are 10.6ms. 
The phonetic representation with 3 symbols per voiced phoneme means the shortest sound cannot be shorter than 31.8ms
which is too long for modeling consonants. MAS was really the weak link in the Matcha solution. 
This trick fixes it entirely. I am now computing the mels with a hop length of 128, reducing the frame duration 
from the 10.6ms to just 5.3ms, and the minimum duration per symbol to 15.9ms. I still have to produce a standard resolution 
mel as I am bound by Vocos, there's no better option fast enough for my needs. I decided to downsample the fine res mel
to a std res mel, and I am doing it before the Decoder, which is the heaviest/slowest model component. Running the Decoder
in fine res would have doubled the training and inference time, for no good reason. After all, I only needed the fine res
trick for the duration detection bit.
I have a feeling this hasn't been done before in any model. 

17. Kernel size too small to see past the separators
Look at the phonetic representation for this text (the pipe is the separator described above).
    Input text:      <Didn't you?>
    Phonetised text: < |d|ˈ|ɪ|d|n|t| |j|u|ː|?>
With a larger kernel the signal from the ? is stronger, so the model now started to pronounce ? and ! better.

18. The duration loss is calculated on a log scale, so that long phonemes, which have larger errors in frames, do not
get more weight than short ones. On a log scale, being 10% off costs the same for any phoneme.
But the log curve is very steep near zero. With 5.3ms frames and 3 symbols per voiced phoneme, many durations found by
MAS are only 1 to 3 frames. From 1 to 2 frames the log jumps by 0.69, from 8 to 9 frames by only 0.12. So a one frame
error on a short symbol, often just MAS noise, cost about 6 times more than on a long one, and short symbols dominated
the loss.
The fix was to add 4 frames before calculating the log, and subtract them back at inference. This moves all durations
into the flatter part of the curve: from 5 to 6 frames the log jumps by 0.18, from 12 to 13 by 0.08, about 2 times
instead of 6. It has a huge effect.

19. Speaker embedding was concatenated into the Encoder input
Update all components to use FiLM, fixed initialization, and generally replaced the original components with more 
modern versions.

20. All 3 losses backpropagated to all subcomponents.  
I have separated them, each one is now detached from the other parts of the model, and is allowed to backpropagate
only to its subcomponent.

21. The Decoder's attention layers did not know how far along the trip they were
How big a correction to make depends on how far along the trip the Decoder is: at the start the mel is mostly noise, and
it should move boldly, near the end it should only polish. That position on the trip is the timestep, and it used to be
handed only to the convolution blocks, which simply added it to what they produced. The attention layers, the part that
lets each frame look at its neighbors, were handed it but quietly ignored it.
The timestep now reaches the attention layers too, as a handful of values that stretch and shift the signal and decide
how much of the attention and the feed-forward result to keep. 

22. No way to test the quality other than by subjective tests.
I've introduced an MCD script I can use to compare the quality of each checkpoint along the way during training
and then to compare current training to previous ones.
I also used that to find the right ODE solver and the right number of steps for inference.
The standard MCD test uses only 13 coefficients, which does not cover all frequencies properly, so my script uses 30,
computed at 24KHz.

23. ~~The prior loss was giving equal importance to every frequency, but the ear is not equally sensitive to all of them.
It is most sensitive around 3 to 4KHz, and much less at the very low and very high end.
I now weight each mel bin in the Prior Loss by how sensitive the ear is at that frequency, following the equal-loudness
contour (ISO 226), as described in "A Perceptually Motivated Loss Function for Speech Enhancement Based on
Equal-Loudness Contours" (2025). I lowered the top of the curve a bit more, since the human voice does not reach those
frequencies. The weights only range from 0.7 to 1.3, so the loss is nudged towards the important bins without starving 
the extremes; the high frequencies still carry timbre and speaker identity.~~

I tried this, and then removed it in v24, as it appeared to cause artifacts in some male voices. I can only describe them as a "sore throat", a coarseness in male voices, especially speaker 4 (Brian). I do not have any test that detects it. Neither MCD nor UTMOS capture this aspect, and subjective tests are the only way to assess the presence of those artifacts, but in subjective tests I am fairly sure the Brian sounds better after I trained without it.

24. Decoder was spending equal training time on every part of the trajectory from noise to destination.
Not all parts matter equally. I measured the Decoder's error along the trajectory: near the destination it is 75 times
larger than near the start. An error made early is damped about 18 times before the solver finishes, while one made
late reaches the output almost untouched. Put together, roughly 99% of the final error comes from the last third.
The very end is the exception, the solver never visits it: midpoint with 4 steps stops at 0.81, heun3 with 8 steps at
0.935. Training above 0.94 is wasted. I now draw the training steps from a logit-normal distribution 
("Scaling Rectified Flow Transformers", Esser et al. 2024) shifted towards the destination: under 3% of the budget 
goes below 0.3, under 5% above 0.94, the rest in between. 
I also added a validation loss measured at 0.935, which tracks output quality far more closely than the total does.

25. Decoder was taking equally long steps along the trajectory from noise to destination.
The coarse structure of the speech is formed during the early steps, so the solver needs more resolution there.
I bend the step grid towards the start, as described in "Sway Sampling" (Chen et al. 2024, F5-TTS), using the strongest
setting that still keeps the steps in order. With 4 steps, the solver now spends 3 of them on the first 62% of the
trajectory and one long step on the rest. Same cost, reported to improve quality at a low number of steps.
That long last step is also why the solver never visits the very end of the trajectory, see 24.

26. The Decoder kept forgetting what it had learned about the timesteps it had not seen recently.
Every training step draws a new timestep for each sample, so each recording is seen at a different point of the
trajectory every epoch. Each step pulls the weights towards whatever lowers the loss for the timesteps it just drew,
and away from what was learned for others, which may have been even better trained. Getting better at the timesteps
of step N can mean getting worse at those of step N-1, so every checkpoint is biased towards the timesteps it saw last.
I now keep a second copy of the Decoder weights, an exponential moving average, that moves a small fraction of the way
towards the live weights after every optimizer step. With a decay of 0.9999 it averages roughly the last 10000 steps,
about 21 epochs, so it blends what the Decoder learned across thousands of different timestep draws instead of keeping
only the latest. It is never trained directly. Only the Decoder is averaged. 
Validation reports the losses for both copies, on the same timesteps and noise, so they can be compared. Inference,
the validation scripts and the production checkpoint use the averaged weights.

27. The Decoder input signal was getting less and less rich over time.
The better the Encoder gets, the more precise MAS will become, returning probably the exact same alignment with each 
train step. That means the Decoder sees the same alignment over and over again. If the alignment was slightly different
each time, the Decoder would have much more examples to learn from.
This is especially true when I freeze the prior for fine-tuning, Decoder keeps seeing one fixed input per sample, 
and can memorize it. I now move each phoneme boundary a little on every training step, before building the mel the 
Decoder is conditioned on. The jitter comes after the duration losses computation, so the Duration Predictor still learns
the true MAS durations. It lowered the Decoder validation loss quite a lot.

28. Prior loss was giving more weight to the long phonemes. 
If many long phonemes sound fine but there are short phonemes that sound bad, the short ones are ignored as they don't 
contribute much to the prior loss value. It was like the long phonemes were getting more votes than short ones 
(one vote per frame).
The fix is to divide each individual phoneme error by the number of frames that phoneme spans (one vote
per phoneme).
This does not work at the beginning of the training, the model will not be able to learn durations. I have introduced a
hyperparam which switches from one vot per frame to one vote per phoneme, so I can turn this on in later stages of the 
training.