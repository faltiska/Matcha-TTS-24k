# Matcha-TTS Components

## Overview

An IPA Phonemizer converts sentences to phonemes.
They are fed into an Encoder model which generates a mel frame for each phoneme.

The frames are fed into MAS to find durations for each phoneme, aligning the frames to the full mel

Then the mel frames + durations are assembled into a full mel spectrogram.
The assembled mel is compared to the ground truth mel to compute the encoder loss.

The assembled mel is then fed into a Decoder that generates the final high quality mel. 
The Decoder generated mel is compared to the ground truth mel to compute the Decoder loss.

A Duration Prediction model is trained to predict phoneme durations from phonemes. 
The predicted durations are compared to the durations detected by MAS to compute the Duration Predictr loss. 

### The three losses are summed, but they do not influence one another

The training step computes three losses — the Prior loss (also called encoder loss), the Duration Predictor loss,
and the Decoder loss (also called flow matching loss) — and adds them into a single number for the optimizer.

Adding them does NOT mean they interact. Each loss trains only its own submodel, and only its own submodel,
because the tensors that connect the submodels are detached at every boundary.

The Mel Predictor reads an undetached Encoder output but then:
- The Duration Predictor reads a _detached_ copy of the Encoder output.
- The Decoder reads a _detached_ copy of the assembled mel.

## What Each Module Does During Training

### Phonemizer
Supported languages: en-us, en-gb, ro, fr-fr, it.
NeMo text normalization runs before eSpeak (numbers, dates, abbreviations, etc.). 
Not available for RO, where eSpeak's own normalization is used.
A leading space is prepended to the phoneme stream so the model gets a brief initial silence to ease into the utterance.

The Phonemizer design is driven by the use of MAS downstream.
MAS is used to find phoneme durations, and to do that it needs to identify mel frames in the input spectrograms.
The Encoder generates the mel frame for each phoneme, and MAS can detect something like "this frame is repeated 5 times 
in the given spectrogram".
But the input mel spectrogram contains more than just the individual phonemes. The transition from one phoneme to another
(called coarticulation) does not sound like exactly like the surrounding phonemes. We need to insert an additional symbol
between every 2 voiced phonemes, so that the Encoder will model the coarticulation there, so that MAS can identify it.
There are 3 possible ways to do this:
1. A single symbol is inserted in between all phonemes (original Matcha design, worst design, since the 
model sees the same symbol pronounced in about 5000 different ways, with 5000 different lengths).
2. A distinct symbol for each possible pair of voiced phoneme gets inserted in between them (creates a huge dictionary
of around 5000 symbols, that occur so infrequently in the corpus, that the model cannot really learn)
3. A mid-ground solution, currently implemented, where each voiced phoneme (and only voiced phonemes) is replaced by a tuple of
(pre-phoneme, phoneme, post-phoneme), with a dictionary of less than 600 symbols. 
 
The problem with the approach currently implemented is that each coarticulation is modeled by 2 symbols.
A pair of phonemes will be represented as 6 symbols. Since MAS must assign at least 1 frame to each symbol, it means the
2 phonemes will take at least 6 mel frames. If they are 2 short consonants, that means 6 x 10.6ms = 64ms

The 10.6 ms figure is imposed by the Vocoder. I will explain later why I chose Vocos 24KHz with a hop of 256. 

64ms is too long. Short consonants are 15 - 20ms. MAS eats up the space in the input spectrogram, and runs out of frames 
to assign before it finishes the input sentence phonemes. This lead to multiple skipped phonemes, while other were 
elongated unnaturally.

To fix this, I am running the Encoder and MAS in a higher resolution. 
I am using 24KHz / hop 128 input mel spectrograms, that have frames of 5.3ms.
The finer resolution buys MAS smaller duration steps, not more acoustic detail, since both resolutions are extracted
with the same analysis window. See the Mel Analysis Window section below.

### Text Encoder
A small stack of convolutions with SiLU activation, Norm and Dropout, called a pre-net, takes phonemes and processes 
them before they are fed into the Encoder.

The main component is the Encoder, an attention model that takes the pre-net output and produces an internal representation.
It uses Rotary Positional Embeddings (RoPE) applied to half of each head's embedding.

The encoder output is fed into a Mel Predictor that outputs one mel frame per phoneme. The Mel Predictor is a simple 
sequence of Conv → SiLU → Conv.
The Encoder uses feature-wise linear modulation (FiLM) to incorporate speaker embeddings that encode the acoustic characteristics of each speaker. For every Transformer layer, the speaker embedding is projected into two scale-and-shift pairs: one is applied after the attention normalization and the other after the feed-forward normalization. The projection is initialized as a no-op (scale = 1, shift = 0), allowing speaker conditioning to be learned gradually.
The Mel Predictor input is the speaker-conditioned Encoder output.
The gradients from the Mel Predictor flow back into the Encoder too.

The Encoder output is also fed into a Duration Predictor that guesses the duration per phoneme.
The gradients from the Duration Predictor don't flow back into the Encoder.
The Duration Predictor has its own speaker embeddings that encode the rhythm characteristics of each speaker.
It is a stack of Conv1d → ReLU → LayerNorm → FiLM → Dropout layers. The speaker embedding is projected through a single 
linear layer into a `gamma` (scale) and `beta` (shift) pair that is applied after LayerNorm at every layer. 
The projection is initialized so FiLM starts as a no-op (`gamma=1`, `beta=0`), letting the model converge before the 
speaker conditioning starts to drive.

### Reference Durations (MAS)
The reference durations used to train the Duration Predictor come from MAS (Monotonic Alignment Search), which compares 
the mel frames generated by the Encoder against the ground truth mel frame-by-frame and finds the best alignment. 
This is computed continuously during training.

MAS itself and the prior loss are computed in fp32, with autocast explicitly disabled around the matmuls. 
I thought I saw some instability in BF16. I should revisit this.

The MAS detected durations are used to assemble the mel frames from the Encoder into full length spectrograms which can
then be compared to the input spectrograms to calculate the Prior Loss that drives the Encoder and the Mel Predictor.
The Prior is a Huber loss, using MSE is causing a very large validation to train loss gap.

Also, the Duration Predictor loss is calculated by comparing its output to the durations detected by MAS. 
It is also a Huber loss.

### Decoder
This is a generative model trained with Conditional Flow Matching, a sort of diffusion model.
It is larger and slower than the rest of the components. It takes the spectrogram assembled as described above.
It learns the path for taking each value in that spectrogram into the corresponding value in the ground
truth spectrogram. Receives only the assembled mel and the ground truth mel.
At any random timestep, it interpolates between noise and ground truth to get a noisy sample, then learns to predict the
velocity field (direction) from noise toward ground truth.

But the amount noise is different at each timestep. At the start of the trajectory the signal is mostly noise and the
correction should be large; near the end it is almost clean and only fine details need adjusting. To help the model
decide how much noise is still in the signal, and how aggressively to correct the mel we add the timestep as a 
conditioning signal.

Architecturally it is a small U-Net: each down/mid/up stage is a convolutional residual block (ResNet), followed by a stack
of attention blocks. I have 2 types of attention blocks, Transformer and Conformer, configurable, but I trained 
with Transformer so far.

The diffusion timestep is injected via a sinusoidal positional embedding + MLP. It reaches the model in two ways:
- The ResNet blocks add it directly to the activations, shifting the feature map at that point in time.
- The Transformer attention blocks use it via Adaptive Layer Norm Zero (adaLN-Zero): the timestep embedding is projected
  into 6 modulation parameters — a scale and shift applied to the hidden states before attention, a gate on the attention
  output, and the same triplet for the feed-forward sub-layer. The gates are initialised to zero, so early in training
  the block behaves as a plain residual transformer; the timestep conditioning activates gradually as the model learns to use it.

ODE starts from the assembled mel plus noise as input, both during training and inference.
This was described in the original Matcha paper, but it is not what is implemented ikn their github code.

The Decoder gradients don't flow back into the Encoder, Mel Predictor, or Duration Predictor.
The Decoder does not have speaker embeddings. The speaker characteristics are already encoded in the assembled mel.
Tone, rhythm, prosody, is all there. If you took the mel assembled from the Encoder output and MAS durations, 
and generate an audio file from it, you would find that it sounds almost like the ground truth, though a little choppy.
It has some clicks and pops. The Decoder's job is to convert the jagged Encoder spectrogram into one that sounds smooth.

At this point, we can no longer work in fine resolution. The Encoder training would be too slow, and its output would not
match what the Vocoder expects. I need to go back to standard resolution at this point. I'm downsampling the assembled mel
that had 5.3ms frames into a spectrogram with 10.6ms frames with `box_downsample`, an [1, 1, 1] / 3 box filter
(avg_pool1d with a kernel of 3 and a stride of 2). It averages across frame pairs rather than within them, so it smooths
as well as decimates. That smoothing is deliberate, see the Mel Analysis Window section below.

As such, the Decoder needs a different set of ground truth mels for computing losses, mels generated with a hop of 256, 
not the fine resolution mels used as to calculate the Encoder loss.

### Vocoder
I needed a high quality vocoder that is not horribly slow. The only one I could find that meets both criteria is Vocos
and the best one I found was trained on 24KHz mels, with 100 bins and a hop of 256. It is almost as good, but 50 times
faster than the amazing BigVGAN from nVidia.

### Mel Analysis Window
Both mel spectrogram variants, the standard resolution one with a hop of 256 samples and the fine resolution one with
a hop of 128 samples, are extracted with the same 1024 sample analysis window, which is 42.7ms of audio at 24KHz.
Only the hop differs. Two consecutive fine resolution frames therefore share 87.5% of their samples.

A measurable consequence is that the extra frames of the fine resolution mel carry almost no new acoustic detail:
the in-between frames are 78.5% predictable by simply interpolating between their neighbours. This is not a defect.
It confirms what the fine resolution is for. It buys MAS finer *granularity*, durations in steps of 5.3ms instead of
10.6ms, so a short consonant can be given 3 fine frames instead of being forced up to a multiple of 10.6ms. That is
what fixed the skipped phoneme problem described in the Phonemizer section, and it has nothing to do with the window
length.

#### Why a shorter window looked attractive
A 1024 sample window smears every frame over 42.7ms of audio. A 15 - 20ms consonant is therefore spread across a
window several times longer than the sound itself, which blurs exactly the boundaries MAS has to find. Shortening the
window would make the extra fine resolution frames genuinely informative and give MAS a sharper view of the audio.

#### Why we did not do it
Frequency resolution is inversely proportional to window length. A 1024 sample window at 24KHz resolves about 23.4Hz
per bin; a 640 sample window resolves only about 37.5Hz. Voiced speech is a comb of harmonics spaced at the speaker's
fundamental frequency, roughly 85 - 180Hz for male voices and 165 - 255Hz for female ones, and the window must span
several periods of that fundamental to separate one harmonic from the next. At 1024 samples the window covers about
four periods of a 100Hz voice; at 640 samples it covers under three, and the harmonic comb starts to smear into a
single broad hump.

That comb is where pitch and speaker identity live, and the Decoder has no speaker embedding of its own. The
assembled mel it receives is its only source of speaker information. Losing harmonic structure there means losing
voice identity, which is a much worse trade than a slightly blurred consonant boundary.

Measured on the corpus: the efficient exchange rate runs out at around a 640 sample window, where roughly 11% of
pitch fidelity is unrecoverable. This is an information limit rather than a modelling limit, because a linear fit and
a small neural network both plateau near 89%.

The risk did not fall where expected. The low pitched speakers were unaffected and scored above the corpus mean.
It fell on the two speakers with the least harmonic structure to begin with: the whispering speaker at 66%, and
speaker 11 (Lewis) at 70%. For the whispering speaker the failure mode was specific — the shorter window produced
80% of her harmonic ripple magnitude while matching only 66% of it, meaning it invents harmonic structure in the
wrong places.

The idea was dropped because it targets MAS, and MAS performs well at the current window. Certain cost, speculative
gain.

#### The standard resolution mel is exactly every other fine resolution frame
Because both variants share the 1024 sample window and only the hop differs, the standard resolution frames land on
the same sample positions as the even numbered fine resolution frames. Taking every other fine frame reproduces the
standard mel exactly, measured as zero error across the corpus. This question is closed, there is nothing to recover
by filtering.

It follows that ranking the downsampling filters in `matcha/utils/model.py` by their error against the standard mel
only ranks them by how little they filter, since doing nothing scores zero. That error is not a quality criterion.
The filters differ in how much they smooth the assembled mel, and nothing else, so the choice is a preference about
smoothing rather than about correctness. Training with a sharp filter while running inference with a blurrier one
leaves the Decoder output deliberately under sharpened, which is a softening knob and not a fix for an artifact.

#### Why we smooth on purpose
Exactness is available and we do not want it. `box_downsample` uses an [1, 1, 1] / 3 filter, so each standard frame is
mixed with the two fine frames on either side of it, and the result is measurably flatter than the true standard mel.
This is the better input for this Decoder, and the reason is the Decoder's job rather than the mel's accuracy.
Because use_mu_prior is true, the Decoder does not map noise to speech, it learns a correction from the assembled mel
toward the ground truth. A correction learner absorbs a consistent bias almost for free, because it can learn one rule
and apply it everywhere. It handles an error that varies frame to frame much worse, because it first has to work out
which case it is in. The box filter's error is large but consistent in direction and magnitude; a sharper filter's
error is smaller on average but depends on how fast the mel is moving locally.
This was tested rather than assumed. A sharper [1, 2, 1] / 4 triangular filter was trained for 2000 epochs in v22 as
the only change, and it came out worse both subjectively and by MCD, with harshness concentrated exactly where the mel
changes fastest. That filter has been deleted. See the Downsampler mechanism section in IMPROVEMENT_IDEAS.md for the
full account and the ideas it produced.
Both training and inference use the same filter, so the Decoder is never asked to correct a mel it did not see in
training.

### Speaker Embeddings
Two separate embedding tables: one for the Encoder, one for the Duration Predictor.
The Encoder applies its embedding through FiLM conditioning after both normalization steps in every Transformer layer.
The Duration Predictor uses embeddings for FiLM conditioning.
The Decoder has no speaker conditioning.

---

## What Each Module Does During Inference

Duration Predictor's output is used to decide how many frames each phoneme spans, which determines the total output length.
The Decoder runs a configurable ODE solver, but I found that "midpoint" is the best option.
The Encoder generates a mel frame per phoneme. We use durations and mel frames to assemble a complete fine resolution 
spectrogram. We downsample it to standard resolution and feed it into the Decoder.
The Decoder generates the final spectrogram which is fed into the Vocoder. 
The Vocoder generates the final audio.

At inference, you can also blend multiple speakers. The mix is applied to both embedding tables (encoder and duration predictor)
independently, producing a voice that is acoustically and rhythmically a weighted average of its sources.

---

## Where to Find Each Component

- **Text Encoder & Duration Predictor**: `matcha/models/components/text_encoder.py`
- **Flow Matching Decoder**: `matcha/models/components/flow_matching.py` and `matcha/models/components/decoder.py`
- **MAS Alignment**: `matcha/models/matcha_tts.py` (uses `super_monotonic_align`)
- **Main Model**: `matcha/models/matcha_tts.py`
- **Vocos Vocoder**: `matcha/vocos24k/vocos_wrapper.py`
- **CLI Interface**: `matcha/cli.py`
