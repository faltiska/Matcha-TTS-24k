# Alignment investigation (v24, November 2026): context for continuing in a new chat

Written for the assistant, not as user documentation. Read `.amazonq/rules/assistant-rules.md` and
`documentation/components.md` / `documentation/PROBLEMS.md` first.

## How to talk to the user
- Structured answers: short sections, tables, plain language. No code variable names in prose ("Duration
  Predictor", "the alignment", "prior loss", not `logw`, `attn`, `mu_x`). Explain the chain of reasoning step by step;
  a dense stream of findings was rejected as "makes sense only for yourself".
- Do not be dismissive; this is a year of work. Do not say "the problem is not the Encoder": the Encoder and
  the aligner are one coupled system (Monotonic Alignment Search (MAS) runs on the Encoder's own output).
- Do not assume a sound belongs to its core symbol: each voiced phoneme is a triple (opening separator,
  phoneme, closing separator) and the model spreads sounds over the triple. Reason per phoneme group.
- Do not call stress marks, length marks or separators "silent" or "symbols with no sound of their own". The model
  does not know a stress mark should take zero frames, and the aligner cannot assign zero frames anyway. It learns,
  little by little, to give acoustic content to every symbol: annotation, voiced phoneme or coarticulation. That
  is how the scheme works, not a defect. Do not propose fixes that force the model to treat symbols the way a
  linguist would (zero-length marks, folding stress or length into the vowel to remove frames, and so on).
- The user resists "the audio is wrong" framings: the model chooses the alignment, the loss is its only teacher.
  Be precise: the hard-path loss reinforces the current alignment (Viterbi / hard-EM training).
- State which file a timestamp refers to. Ground truth `data/corpus-24k/wav/4/1423.wav` and inference output
  `speaker_004_encoder.wav` differ by ~0.2 s after "form" (Duration Predictor stretching). This caused confusion.
- Plan first, implement after confirmation. GPU: training may run locally; CPU diagnostics are preferred. The user
  can pause training when GPU is needed. Exclude `logs` and `.venv` from greps. Activate `.venv` before Python.
- The user has tried external aligners (Montreal Forced Aligner, wav2vec2 phoneme models) and rejects them:
  the separator scheme is unique. Do not propose them again.

## Key facts about the setup
- Fine mel: hop 128 at 24 kHz, 5.3 ms per frame. Alignment runs on fine mels with `super_monotonic_align`
  (GPU). Min 1 frame per symbol. Symbol ids: core < 200, opening separator = core + 200, closing = core + 400,
  space id 24. `LOG_DURATION_OFFSET = 4`.
- Alignment score per (symbol, frame) = -0.5·|frame|² + symbol·frame - 0.5·|symbol|² (see
  `MatchaTTS.find_alignment`). Duration Predictor is trained on MAS durations, input is the detached Encoder state.
- `prior_loss_per_phoneme` ("one vote per symbol") divides each frame's error by its symbol's frame count
  (`per_phoneme_prior_loss` in `matcha_tts.py`). Using it from epoch 0 prevents duration learning (user's experience).
- Checkpoints: old v24 run `logs/train/v24/checkpoint_epoch={124,279,479,789}.ckpt` (one vote per frame),
  EC2 run resumed from 789 with one vote per symbol: `logs/ec2/train/v24/runs/2026-11-11_09-20-16/checkpoints/`
  (1014, 1054). New run resumed from 124 with one vote per symbol:
  `logs/train/v24/runs/2026-11-11_21-08-01/checkpoints/` (199-244, 5-epoch steps).

## What was found
1. The audible Encoder errors come from wrong alignments on the ground truth, which the Duration Predictor
   then reproduces at inference. Verified: my CPU MAS gives output identical to the GPU kernel.
2. 4/1423 "In some form or other we need fun." (speaker 4, en-gb). Ground truth: f 0.56-0.67 s, "orm or o" vowels
   0.68-1.09, ð (dip) 1.11-1.15, ə 1.16-1.34, real pause 1.38-1.48 (about -100 dB), "we need fun" from 1.50.
   Alignment: ɔ group of "form" takes 0.82-1.31, m of "form" on the fade 1.35-1.42, the word gap after "form"
   takes the pause, "or other" squeezed onto the start of "we" (1.46-1.63, ~1 frame per symbol). Unchanged at
   epochs 124, 279, 479, 789, 1014, 1054 and in the new run at 244. User confirmed by ear (ð at the dip).
3. 4/1428 "Next Tuesday we must vote." Real pause 1.19-1.42 s after "Tuesday". Alignment: j of "Tuesday" takes
   0.59-1.09 s (swallows uːzdeɪ), z, d, eɪ are placed inside the pause, the word gap gets 1 frame. Sounds like
   "Tjuuu…" with no "day" (user confirmed). Same family as 4/1423: a pause not written in the text, placed wrongly,
   and a vowel-like symbol swallowing its neighbours. Unverified guess: quiet consonants (z, d closure) next to a pause
   are cheap to drop into silence.
4. Mechanism: hard MAS + prior loss on the chosen path = self-reinforcing. Long vowels are the first anchors
   (one vote per frame lets them dominate), and early anchor errors lock in. The alignment is already settled by
   epoch 124: boundaries moved >2 frames per 100 epochs: 124→279 2.6%, 279→479 1.3%, 479→789 0.7%.
5. One vote per symbol helps, slowly: collapsed vowels (vowel triple ≤ 3 frames) on 50 val files/speaker:
   old run 124: 1.8%, 279: 2.0%, 789: 2.0%; EC2 1054: 1.6%; new run (from 124) 199: 1.6%, 244: 1.7%, with duration
   error 0.111 (old run 279: 0.113), so durations were not hurt when switching at 124. The new run moves the
   alignment about 2x faster per epoch (5.7% and 4.9% per 100 epochs). Per-speaker, 100 val files, old 279 vs new 244:
   better 0, 1, 10, 11, 13; worse 3, 5, 8. Neither problem file fixed yet. The user hears 1054 as clearer
   (its Duration Predictor stretches "form" less: 0.98 s vs 1.07 s); the alignment did not change.
6. Worst speakers for collapsed vowels: 11 (8-9%, runs up to 10 vowels), 13 (4-5%), 8 (3%). Speaker 11 has
   digital-zero gaps mid-sentence (noise-gated). The user says it sounds fine and asked to ignore speaker 11.
7. Unwritten pauses (≥100 ms, < -50 dB, inside the speech) beyond the punctuation count, validation set: speaker 11
   about 151, 6 about 65, 4 about 42, 3 about 41, 12 about 38, others 0. Speaker 6 has many and aligns fine, so pauses
   trigger the problem but do not explain it alone.
8. Validation set leak: 291 of 2625 validation wavs are byte-identical to training wavs (speaker 4: 45/203,
   speaker 5: 49/52, 0/1/2/10: ~49 each). 4/1423 = 4/0254 and 4/1428 = 4/0350. Not fixed.
9. Errors by symbol class, epoch 789 (one vote per frame), 150 train files per speaker, speaker 11 excluded,
   ~77k phoneme groups. "Swollen" = group longer than 2.5x the speaker's median for that phoneme (can include real
   pauses, not only errors). Long vowels and diphthongs are the most reliable groups, so the earlier guess that
   long vowels / diphthongs / the length mark cause the errors is not supported.
   | Kind | Collapsed | Swollen |
   |---|---|---|
   | Short vowel, stressed | 0.07% | 3.35% |
   | Consonant | 0.79% | 2.32% |
   | Diphthong half, unstressed | 0.36% | 1.89% |
   | Short vowel, unstressed | 0.42% | 1.58% |
   | Diphthong half, stressed | 0.12% | 0.90% |
   | Long vowel, stressed | 0.04% | 0.75% |
   - Most often swollen: t, b, d, p, k (3-4.3%), then ɪ, ʌ. In swollen consonants the closing separator's share of
     the group rises from 33% to 41%; in swollen diphthong halves from 32% to 44%. Fits the guess that stop closures
     (near silence) are the cheapest place for unwritten pauses.
   - Annotations get real frames, as the model gives every symbol acoustic content: primary stress inside a word
     median 21 ms, mean 27 ms (it sits in the consonant-to-vowel transition, between the closing and the opening
     separator); length mark median 21 ms, about 20% of its long vowel, more than half of it only 0.9% of the time.

## Ideas discussed
- Rejected by the user: forcing a minimum of 2-3 frames per symbol (it brings back the standard-resolution problems),
  external aligners, and resuming to find an epoch "before the error exists" (the goal is the best epoch to switch the loss on).
- Tickets the user logged or is logging:
  1. Alignment noise (VITS2-style), decaying. Works as prevention when used from epoch 0, not only as a correction.
  2. Soft alignment / forward-sum loss in addition to the hard-path prior loss.
  3. Curriculum: strength s in [0, 1], frame weight = 1 / duration^s, normalised by the sum of weights; s ramps
     from 0 (one vote per frame) to the target (one vote per symbol or lower) over the epochs, starting at epoch 0;
     optionally combined with noise that decays as s rises. Log the plain per-frame prior loss separately.
  4. Separate aligner trained before the main training (RAD-TTS style: symbol encoder + mel encoder, distance-based
     attention, forward-sum loss, diagonal prior, gradual binarization) on the full symbol set incl. separators;
     extract durations to `<utt>.durations.npy`; the main training restricts the alignment to a band around them,
     widening over the epochs.
- Discussed, not ticketed: pause anchors (silences ≥100 ms may only go to word gaps or punctuation), a decaying
  position (beta-binomial) prior, and a rule-based broad-class aligner from the mel (silence, vowel, sonorant,
  fricative, closure).
- Suggested check: 4/1423 and 4/1428 again around epoch 350-400 of the new run; if they are unchanged, prioritise the noise ticket.

## Tools
- All in `matcha/utils/` (not yet committed, CPU only, safe while training holds the GPU):
  - `alignment_diagnostics.py`: shared helpers (load Encoder from a checkpoint, CPU MAS identical to the GPU
    kernel, phoneme groups, word segments, Audacity labels).
  - `measure_collapsed_vowels.py -i <data yaml> -c <ckpt> [--split train] [--files_per_speaker N]`
  - `inspect_alignment.py -i <data yaml> -u 4/1428 -c <ckpt...> [--labels_dir DIR] [--wav_dir DIR]`: per-phoneme-group
    alignment vs prediction, wav silences, Audacity label tracks, Vocos renders (ground truth, Encoder with
    ground-truth alignment, Encoder with predicted durations).
  - `compare_alignment_over_checkpoints.py -i <data yaml> -c <ckpt...> --track 4/1423 4/1428`: collapse share,
    duration error, word placement of tracked files, boundary movement per 100 epochs between consecutive checkpoints.
  - `measure_alignment_by_symbol_class.py -i <data yaml> -c <ckpt> [--split train] [--files_per_speaker 150]
    [--exclude_speakers 11] [--alignment_cache /tmp/x.pkl]`: frames on stress / length marks, collapse and swell
    rates by vowel kind (short, long, diphthong half; stressed or not) and for consonants, separator shares inside
    swollen groups. Cache for 789: `/tmp/alignments_789_train.pkl` (~70 s to recompute on 8 threads).
- Run naming: `*b.ckpt` in `logs/train/v24/` = second run (from 124, one vote per symbol). Compare 124→279 (first run)
  against 124→249b (second run): moved per 100 epochs 2.6% vs 4.2%; collapsed 1.9% vs 1.6%; duration error 0.113 vs
  0.110. 4/1428 at 249b: "Tuesday" now ends at 1.42 s, "we" starts at 1.43 s (correct word boundary), but the pause
  1.19-1.42 s is still inside "Tuesday". 4/1423 unchanged.
- Audacity label tracks in the project root: `4_1423_ground_truth_labels.txt` (epoch 1014),
  `speaker_004_encoder_labels.txt` (inference durations), `4_1428_ground_truth_labels.txt` (new run epoch 244).
  Format: start\tend\tlabel per phoneme group, `_` = word gap.

## Decision, 12 November: change the phonemization scheme first
The alignment work (tickets above, re-check of 4/1423 and 4/1428 around epoch 350-400) is paused. We change the
phonemization scheme first and train the next version from scratch with it.

Why: the old scheme wrote each phoneme as (opening separator, phoneme, closing separator), so every transition had
two separators, three when a stress or length mark sat in between. Nothing in the audio decides where one of those
separators ends and the next begins, and in swollen consonants the closing separator is the part that grows.

The model gives every symbol acoustic content, annotations included (stress marks, length marks, nasal tilde,
palatalization, syllabic mark). It does not treat them as zero-length marks, and we do not try to force it to.

### Rules (implemented in `matcha/text/phonemizers.py`, `insert_separators`)
1. Everything eSpeak outputs is kept as is.
2. A single separator symbol is used everywhere.
3. A separator goes between two neighbouring symbols only when at least one of them is a voiced phoneme (any vowel or
   consonant) and neither of them is an annotation.

Reasons:
1. Separators are only needed to model transitions: between two phonemes, and between a phoneme and silence.
2. The Encoder can use annotation symbols to model transitions, so extra separators aren't needed next to them.
3. The Encoder can learn both an annotation's intended function and the sound it holds.

Example: eSpeak's ` tjˈuːzdeɪ wiː` becomes ` |t|jˈuːz|d|e|ɪ| |w|iː`. Separators stay around word gaps and before
punctuation, except where an annotation touches the gap or the punctuation (`iː ` and ` ˈʌ` get none, as before).

### Measurements behind the rules (epoch 789, one vote per frame, 150 train files per speaker, speaker 11 excluded)
Distances are between the mean audio frames each symbol holds, normalized mel units; frames vary by about 0.5
around their own symbol's mean, neighbouring symbols inside a word differ by about 1.3.
- Separators at word gaps, in files without collapsed vowels: with no pause, closing separator / word gap / opening
  separator differ by 1.2 each and the two separators by 1.8, so they are successive slices of the cross-word
  transition. Next to a pause the separators hold the fade-out (2.2 from the word end, 2.0 from the pause) and the
  onset (1.5 from the pause, 3.3 from the next phoneme). So separators around word gaps are kept.
- Annotations between phonemes sit between the two neighbouring separators (about 1.3 from each) and are far from
  the phoneme they annotate (2.0-3.5): stress in en/fr/it/ro, length mark in en, nasal tilde in fr, Italian double
  consonant length mark. They hold a slice of the transition, not more of the phoneme.
- Annotations next to a word gap (`iː `, ` ˈʌ`) differ from the gap and from the phoneme (about 1.3-1.4): they already
  act as the fade-out and the onset. The old scheme never put a separator there either.
- In all five corpus languages the stress mark sits directly before the vowel (5 exceptions in the corpus).
- Open question only training can answer: an annotated transition goes from three symbols (spanning a distance of
  about 2.1) to one, a plain transition from two (1.3 apart) to one.
- Kept unchanged by rule 1: Romanian doubled palatalization (`pʲʲ`, about 1200 times, also `ʲˌʲ`), Italian `ː` before
  stress, the French liaison hyphen (`le-z`), diphthong and affricate halves (separated like any two phonemes).

### State of the code
- Done: `symbols.py` (single separator id 0, `annotation_ids`, vocabulary of 147 symbols instead of 600),
  `phonemizers.py`, tests in `tests/test_phonemizers.py` (hand-written case per situation, plus the rules checked on
  live eSpeak output for every supported language). All 98 phonemizer tests pass. On the corpus the sequences are 72%
  of their old length; the longest is 404 symbols.
- Not done: regenerating the CSVs (`validate_corpus_ipa.py` rewrites them in place, so write new ones while v24 still
  needs the old ids), updating the alignment diagnostic scripts (they assume phoneme triples), the new experiment config.
- The working tree can no longer load v24 checkpoints (vocabulary size changed). The v24 versions of `symbols.py`
  and `phonemizers.py` are in git HEAD.
