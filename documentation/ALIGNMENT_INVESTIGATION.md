# Alignment investigation (v24/v25, November 2026): context for continuing in a new chat

Written for the assistant, not as user documentation. Read `.amazonq/rules/assistant-rules.md` and
`documentation/components.md` / `documentation/PROBLEMS.md` first.

KEY FINDING (13 Nov): the Duration Predictor (DP) starts fine, then learns the alignment search's (MAS) mistake as
training progresses. MAS is wrong for 4/1423 and 4/1428 from the first epoch and gives the same wrong durations at
every checkpoint (004c-154c). The Encoder mel built with the DP durations sounds right up to about epoch 19 (user's
ears), because the DP gives durations it learned from other sentences. As training goes on, the DP fits this
sentence's wrong MAS durations, and the error becomes audible and stable. MAS does not drift; the DP does. Details:
"13 November: alignment search (MAS) or Duration Predictor (DP)?".

WHERE WE STOPPED (17 Nov):
- The goal: several of Brian's (4) utterances are pronounced wrongly. We traced it to the alignment search (MAS)
  squeezing vowels to one frame ("vowel collapse") around pauses the text does not mark, which the Duration Predictor
  then learns. The open question is whether fixing the corpus alone removes the collapse; if not, fix MAS itself
  (see "Ideas for an alignment fix").
- Corpus fixes (missing commas, transcript errors, audio anomalies) are now done with tools in the corpus project,
  see "Corpus fixes". The pause and comma tools of this project were deleted on 17 Nov; only the alignment and
  collapsed-vowel checks remain here (see Tools).
- Run b `logs/train/v24/runs/2026-11-16_21-09-01/` (all corpus fixes except speakers 10, 12, 14) was still training
  at epoch 99; compared with run a in the 17 November section.
- After corpus edits: bring the corpus over, rerun `validate_corpus_ipa`, then train.

## How to talk to the user
- Structured answers: short sections, tables, plain language. No code variable names in prose ("Duration
  Predictor", "the alignment", "prior loss", not `logw`, `attn`, `mu_x`). Explain the chain of reasoning step by step;
  a dense stream of findings was rejected as "makes sense only for yourself".
- Do not be dismissive; this is a year of work. Do not say "the problem is not the Encoder": the Encoder and
  the aligner are one coupled system (Monotonic Alignment Search (MAS) runs on the Encoder's own output).
- Do not assume a sound belongs to its core symbol. In the old scheme each voiced phoneme is a triple (opening
  separator, phoneme, closing separator) and the model spreads sounds over the triple; in the new scheme separators
  are shared between neighbours. Reason per phoneme group, not per symbol.
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
  the separator scheme is unique. Do not propose them again. (MMS in the corpus project's `find_pauses.py` only
  places words to find missing commas; it does not replace the model's alignment.)
- Code comments and docs describe the code as it is now, never its history (no mention of the old triple scheme in
  v25 code). "Voiced phoneme" in this project means any vowel or consonant; do not nitpick that t is voiceless.
- Prefer measuring on the existing checkpoints before recommending; the user asked for evidence twice, and one
  recommendation (no separators at word gaps) was reversed by a measurement.
- Commits, branch switches, deletions of runs: ask first. The user deletes test runs himself.
- When stopping a run of mine, say why; I stopped a noise run once to run a baseline and had to explain afterwards.

## Key facts about the setup
- Fine mel: hop 128 at 24 kHz, 5.3 ms per frame. Alignment runs on fine mels with `super_monotonic_align`
  (GPU). Min 1 frame per symbol. Old scheme ids: core < 200, opening separator = core + 200, closing = core + 400,
  space id 24. New scheme: single separator id 0, space id 24, 147 symbols. `LOG_DURATION_OFFSET = 4`.
- Alignment score per (symbol, frame) = -0.5·|frame|² + symbol·frame - 0.5·|symbol|² (see
  `MatchaTTS.find_alignment`). Duration Predictor is trained on MAS durations, input is the detached Encoder state.
- `prior_loss_per_phoneme` ("one vote per symbol") divides each frame's error by its symbol's frame count
  (`per_phoneme_prior_loss` in `matcha_tts.py`). User's experience from epoch 0: sounds fine for 40-50 epochs, then
  drifts into gibberish. In 5 epochs it learned durations as well as one vote per frame (see 12 November tests).
- Checkpoints (all old scheme): first v24 run `logs/train/v24/checkpoint_epoch={124,279,479,789}.ckpt` (one vote per
  frame), EC2 run resumed from 789 with one vote per symbol: `logs/ec2/train/v24/runs/2026-11-11_09-20-16/checkpoints/`
  (1014, 1054, 1074). Second run resumed from 124 with one vote per symbol, stopped at 269:
  `logs/train/v24/checkpoint_epoch={254b,269b}.ckpt` (its run folder `runs/2026-11-11_21-08-01/` no longer has
  checkpoints). New scheme: `logs/train/v24/runs/2026-11-12_16-20-22/checkpoints/` (epochs 0-4).
  The old-scheme checkpoints need the OLD `symbols.py` to be read by the diagnostic scripts (ids up to 600).

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
   Refined on 12 Nov: 4/1423 is wrong after the very first epoch in every setup tried, so not every error is a late
   lock-in; 4/1428 starts right and drifts wrong. See the 12 November tests.
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
  1. Alignment noise (VITS2-style), decaying. TESTED 12 Nov from epoch 0 at scale 0.01: unhelpful, code reverted.
  2. Soft alignment / forward-sum loss in addition to the hard-path prior loss.
  3. Curriculum: strength s in [0, 1], frame weight = 1 / duration^s, normalised by the sum of weights; s ramps
     from 0 (one vote per frame) to the target (one vote per symbol or lower) over the epochs, starting at epoch 0;
     optionally combined with noise that decays as s rises. Log the plain per-frame prior loss separately.
  4. Separate aligner trained before the main training (RAD-TTS style: symbol encoder + mel encoder, distance-based
     attention, forward-sum loss, diagonal prior, gradual binarization) on the full symbol set incl. separators;
     extract durations to `<utt>.durations.npy`; the main training restricts the alignment to a band around them,
     widening over the epochs.
- Discussed, not ticketed: pause anchors (silences ≥100 ms may only go to word gaps or punctuation; my current
  proposal, see the 12 November conclusions), a decaying position (beta-binomial) prior, and a rule-based broad-class
  aligner from the mel (silence, vowel, sonorant, fricative, closure).
- Superseded: the suggested check of the second run around epoch 350-400 (that run was stopped at 269).

## Tools
- All in `matcha/utils/`, CPU only, safe while training holds the GPU. They read the current `symbols.py`, so they
  only work with checkpoints of the current phonemization scheme.
  - `alignment_diagnostics.py`: shared helpers (load the Encoder from a checkpoint, CPU MAS identical to the GPU
    kernel, word segments, Audacity labels). `phoneme_groups` = one segment per symbol (separators are shared, not
    merged), `vowel_spans` = a vowel plus the length mark and diacritics written right after it; a stress mark
    before the vowel is left out.
  - `measure_collapsed_vowels.py -i <data yaml> -c <ckpt> [--split train] [--files_per_speaker N]`: share of vowel
    spans of 3 frames or fewer. Not comparable with the old triple-based numbers in the older sections.
  - `compare_alignment_over_checkpoints.py -i configs/data/corpus-24k.yaml -c <ckpt...> --track 4/1423 4/1428`:
    collapse share, duration error, word placement of tracked files, boundary movement per 100 epochs between
    consecutive checkpoints.
  - `inspect_alignment.py -i <data yaml> -u <utt> -c <ckpt...>`: per checkpoint, symbol and word tables of MAS vs DP
    durations; `--wav_dir` renders the Encoder mel with MAS and with DP durations (`_encoder_mas_durations` /
    `_encoder_dp_durations`; never call MAS output "ground truth"), `--labels_dir` writes `_mas_labels.txt` /
    `_dp_labels.txt`. The user listens to these renders to judge an utterance.
- Deleted on 17 Nov, still in commit 2456048 if ever needed: `measure_pause_placement.py` (where MAS puts each
  pause, "pauses that move words", the old missing comma detector), `align_with_text.py` (MAS with a changed text),
  `measure_alignment_noise.py` (noise sweep, pinned-path test), `measure_alignment_by_symbol_class.py` (old scheme
  only). The sections below still quote their results.
- The one-frame-run counts (files with >= 4 / >= 8 consecutive one-frame symbols) came from temporary scripts that
  were never kept.
- The render folders (`mas_vs_dp*/`), label tracks and review lists named in the sections below are gone from the
  project root.
- `/tmp/print_tb.py <tensorboard dir...>` prints the per-epoch duration / prior losses of runs side by side (temporary,
  /tmp may be wiped; ~20 lines with tensorboard's EventAccumulator).
- Run naming: `*b.ckpt` in `logs/train/v24/` = second run (from 124, one vote per symbol). Compare 124→279 (first run)
  against 124→249b (second run): moved per 100 epochs 2.6% vs 4.2%; collapsed 1.9% vs 1.6%; duration error 0.113 vs
  0.110. 4/1428 at 249b: "Tuesday" now ends at 1.42 s, "we" starts at 1.43 s (correct word boundary), but the pause
  1.19-1.42 s is still inside "Tuesday". 4/1423 unchanged. (Later runs reuse letters: 099a/134a, 064b-099b, see 17 Nov.)

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

### State of the code (end of 12 November; the CSV parts are superseded by "WHERE WE STOPPED")
- The working tree is on branch `v24` (last commit aad0c56) with these UNCOMMITTED changes:
  - The new phonemization, brought over from branch `v25` by the user: `matcha/text/symbols.py`
    (single separator id 0, `annotation_ids`, `SEPARATOR_ID`, 147 symbols instead of 600), `matcha/text/phonemizers.py`
    (`insert_separators`), `tests/test_phonemizers.py` (98 pass). So the working tree now phonemizes with the NEW scheme,
    and `data/corpus-24k/train.csv` / `validate.csv` (old ids) no longer match the code.
  - `configs/experiment/v24.yaml`: checkpoints and validation every epoch (user's change for the short tests).
  - `matcha/utils/alignment_diagnostics.py` and `matcha/utils/measure_collapsed_vowels.py` rewritten for the new
    scheme (see Tools).
- Branch `v25` (commit 0046602 "A new phonemization scheme") branched from 838dba6, BEFORE the v24 commits ace1aa9 and
  aad0c56. It lacks the diagnostic scripts and this md, and has an older pyproject/uv.lock (torch from the test
  channel, torchaudio 2.11.0+cu132). Checking it out and running `uv sync` would change the venv. The phonemization
  commit only touches `phonemizers.py`, `symbols.py`, `tests/test_phonemizers.py`; it can be cherry-picked onto v24.
  Nothing about the branches has been decided yet; ask the user before any git operation.
- New-scheme CSVs: `data/corpus-24k/train.v25.csv` (34975 lines) and `validate.v25.csv` (2625), written by
  `python -m matcha.utils.validate_corpus_ipa -i configs/data/corpus-24k-v25.yaml` (takes ~11 min, rewrites the CSVs
  named in the config in place, so the old CSVs were copied first). Max id 144 < 147, length 71.8% of the old one.
  The old CSVs are unchanged (md5 train 9de461e9..., validate 1995e4bf...). Mels and mel statistics do not depend on
  the phonemes. Train on the new scheme with `data=corpus-24k-v25` on the command line.
  4/1423 new: ` |ɪ|n| |sˌʌ|m| |fˈɔːm| |ɔːɹ| ˈʌ|ð|ə| |w|iː |nˈiːd| |fˈʌ|n|.`
  4/1428 new: ` |nˈɛ|k|s|t| |t|jˈuːz|d|e|ɪ| |w|iː |mˈʌ|s|t| |vˈə|ʊ|t|.`
- Training reads phoneme ids from CSV column 5 (`text_mel_datamodule.py`, never re-phonemizes); the embedding size
  comes from `symbols.N_VOCAB`. New code + old CSVs crash on the first batch (id out of range); new code cannot load
  v24 checkpoints. The collate pads with id 0, which is now the separator; harmless because every Encoder layer is
  masked by `x_lengths`, but keep it in mind if anything ever reads ids without the mask.
- Rotary cache `max_seq_len = 4000` in `attention.py`; its comment still mentions the old scheme. Longest new sequence
  is 404 symbols.
- Alignment noise was implemented, tested and REVERTED at the user's request (model code and configs back to the
  commit). Only the measurement script remains.

## 12 November: five-epoch tests from epoch 0
The user's method from now on: try an idea for 5 epochs from epoch 0 (about 33 min, ~6.5 min per epoch, 724 batches
per epoch at 20k frames), then keep it or move on. Possible because 4/1423 is already wrong after the first epoch.
Launch pattern (in tmux, so it survives): `python -m matcha.train +experiment=v24 ckpt_path=null
model.prior_loss_per_phoneme=false trainer.max_epochs=5 [data=corpus-24k-v25]`. Each run goes to
`logs/train/v24/runs/<timestamp>/`. The user deletes test runs he no longer needs; only the new-scheme test run
`2026-11-12_16-20-22` (epochs 0-4, new scheme) still has checkpoints. `2026-11-12_17-05-21` was started by the user
after I stopped; I do not know what it is.

Comparisons must use a baseline with the CURRENT config. The first v24 run (`2026-10-09_22-34-40`) is not a fair
baseline for early epochs: 30k frames per batch (fewer steps per epoch), duration loss threshold 1.0 instead of 0.9,
and the prior loss still weighted the mel bins by ear sensitivity. Its validation duration loss was 0.042, 0.036,
0.033, 0.032 over epochs 0-3 and looked better than anything we ran, for those reasons.

Results, all one vote per frame unless stated, 30 validation files per speaker, all speakers:

| Run (old scheme unless stated) | Val duration loss ep 0/1/2/3 | Val prior loss ep 3 | Collapsed vowels ep 3 | Boundary shift per epoch | 4/1423 | 4/1428 |
|---|---|---|---|---|---|---|
| Baseline, no noise | 0.061 / 0.050 / 0.045 / 0.042 | 0.663 | 3.2% | 8 ms | wrong from epoch 0 | right at 0-3 |
| Noise 0.01, Duration Predictor on noisy alignment (stopped at 3) | val dur 0.059 at ep 3 | 0.680 | – | – | – | – |
| Noise 0.01, Duration Predictor on alignment without noise | 0.076 / 0.065 / 0.061 / 0.057 | 0.683 | 3.4% | 10 ms | wrong from epoch 0 | wrong from epoch 1 |
| One vote per symbol, no noise (5 epochs) | 0.063 / 0.049 / 0.047 / 0.043 (ep 4: 0.040) | not comparable | 5.2% (ep 4: 4.4%) | 15-18 ms, not slowing | mostly right at ep 4, pause inside "we" | wrong every epoch, changes each epoch |
| NEW scheme, no noise (5 epochs) | 0.115 / 0.085 / 0.083 / 0.071 (ep 4: 0.068) | 0.774 (ep 4: 0.727) | new definition, 29% | 17 ms, slowing (13 ms at 3→4) | almost right at ep 0, wrong from ep 1 | right at 0-1, wrong from ep 2 |

Collapsed vowels in the baseline over epochs 0-3: 5.4%, 3.8%, 3.2%, 3.2%; one vote per symbol: 10.2%, 6.1%, 4.9%,
5.2%, 4.4%. Collapsed vowels for the new scheme use a different definition (see Tools) and cannot be compared with
the old 3%; I suspect most are not errors (the shared separators may hold much of a short vowel's sound), unverified.
The new scheme's duration loss is averaged over different symbols, so it is not comparable either; the prior loss is
averaged over the same mel frames and is roughly comparable (about 15% worse at this stage, expected with 28% fewer
symbols covering the same audio).

Word placements to recognise the errors quickly (seconds, ground truth file):
- 4/1423 ground truth: "or other" ends 1.34, pause 1.38-1.48, "we" from 1.50. The typical error: "form" runs to
  ~1.35-1.40, the pause lands on the gap after "form", "or other we" squeezed into 1.39-1.70. One vote per symbol at
  epoch 4: form 0.56-0.85, or 0.86-0.99, other 0.99-1.34, we 1.34-1.71 (correct up to "other", pause inside "we").
- 4/1428 ground truth: pause 1.19-1.42 after "Tuesday". Correct looks like "Tuesday 0.53-1.21 | _ 1.21-1.41 | we 1.41-".
  The typical error: "Tuesday 0.52-1.47 | _ 1.47-1.47 | we 1.47-1.55" (Tuesday swallows the pause).

Conclusions:
- Noise at the VITS2 starting scale (0.01) costs about 35% on the duration loss at epoch 3, does not make the
  alignment noticeably more fluid, and broke 4/1428 at epoch 1. The user judged it unhelpful; the code was reverted.
- One vote per symbol from epoch 0 does NOT reduce vowel collapse early (about 1.6x the baseline). Durations learn as
  well as the baseline in 5 epochs. The alignment moves twice as much per epoch and does not settle, which fits the
  user's experience: with one vote per symbol from the start the model sounded fine for 40-50 epochs, then drifted
  into gibberish (stuttering, "pig latin", language-like but unintelligible).
- The new phonemization alone fixes neither file in 5 epochs; the same two errors appear, 4/1428 earlier than with
  the old scheme.
- Correction to finding 4: the errors do not all come from early anchors locking in late. 4/1423 is wrong after the
  first epoch in every setup; 4/1428 starts right and drifts wrong (epoch 1-2 in the noise and new-scheme runs,
  somewhere between epoch 3 and 124 in the baseline). In every setup the model briefly finds the right placement and
  then moves the unwritten pause to whichever nearby gap or symbol is easiest to fit.
- Common factor of both files in every setup: a pause that is not written in the text. Next candidate I proposed
  (user has not answered yet): pause anchors, i.e. silences of 100 ms or more, below about -50 dB, may only be
  assigned to word gaps or punctuation. Works with either scheme, testable in 5 epochs. Before implementing: measure
  on existing checkpoints how often such silences fall outside word gaps and punctuation and how often next to a stop
  closure. Other candidates in my order: soft alignment / forward-sum loss (ticket 2), noise at 0.003, curriculum
  (ticket 3, less promising since one vote per symbol does not help early).

## Alignment noise: measurements and implementation notes (kept for reference, code reverted)
- VITS2 recipe verified in the vits2_pytorch implementation (models.py): noise = `torch.std(neg_cent) * randn *
  current_mas_noise_scale`, added to the score matrix before MAS, start 0.01, minus 2e-6 per step (gone after 5000
  steps, ~7 epochs for us, far too short). Their std includes padding.
- Our score matrix = minus half the squared distance, summed over 100 mel bins. Its spread (std) is about 102 in every
  checkpoint (124, 279, 789, 269b), dominated by far-away symbols, so it is effectively a constant. A frame's own
  symbol scores only about 0.27 better than its best neighbour (median). So scale 0.01 means noise ~1.0 per cell.
- Effect on 280 train files (20 per speaker, speaker 11 excluded), same in all four checkpoints:

  | Noise scale | Boundaries moved ≥1 frame | Moved >2 frames | Mean shift | Collapsed vowels |
  |---|---|---|---|---|
  | none | – | – | – | 0.35% |
  | 0.003 | 45% | 4% | 3.5 ms | 0.5% |
  | 0.01 | 64% | 15% | 7 ms | 1.1% |
  | 0.03 | 77% | 31% | 12 ms | 2.3% |
  | 0.1 | 86% | 53% | 23 ms | 4% |
  | 0.3 | 93% | 74% | 64 ms | 5% |

  During training at 0.01 the noise moved 30% of the boundaries by more than 2 frames at epoch 0, falling to 23% at
  epoch 3 as the Encoder sharpened.
- Pinned-path test (forcing the known pause onto the right word gap, then comparing path scores): the wrong
  alignment scores about 1000-1800 better than the pinned one at 124, 279, 789 and 269b (10-15 per differing frame
  against a 0.27 neighbour gap). Noise cannot flip that without scrambling everything (only scale ~0.3 would). So noise
  cannot correct a settled model; resuming from 124 with noise was not tried for that reason.
- The noise lengthened the shortest symbols: symbols with 1-2 frames without noise got about 2 frames more on average
  with noise, and the Duration Predictor learned that (error 0.24 against the alignment without noise, 0.19 against the
  noisy one). If noise is ever tried again: train the Duration Predictor on a second alignment search without noise
  (costs ~6 ms per step, about 1%), and let only the prior loss and the Decoder input use the noisy one.
- `super_monotonic_align.maximum_path` OVERWRITES the score tensor it is given (it stores running path scores in
  it). Harmless in training today, but anything that needs the scores after the call (noise, a second search,
  diagnostics) must pass a copy or run that search last.
- The user asked whether the noise is like the existing duration jitter: same mechanism (change the alignment before
  assembling a mel), but the jitter is audio-blind, happens after the prior and duration losses and only feeds the
  Decoder; the noise is on the scores, so MAS still follows the audio, and it trains the Encoder. Neither replaces the
  other.

## 13 November: alignment search (MAS) or Duration Predictor (DP)?
- The user listens to the `--debug` Encoder mel, which is built from DP durations (no MAS at inference). He hears
  both sentences right up to about epoch 19-24 of the "c" run, then the error appears and becomes stable.
- Male voices with errors: Brian (4), Lewis (11). No errors: Kai (0), Arthur (5), Michael (12), all female voices.
  Unwritten pause counts do not predict this (Michael, Bella, Nicole have as many as Brian). The user listened to
  whole books without hearing the problem; about 49 validation texts per speaker also appear in training (same wav
  for 0, 4, 5; a different take for the others), so the validation sentences are not new text.
- `inspect_alignment.py` now renders, for every checkpoint, the Encoder mel with MAS durations and with DP durations
  (`<utt>_epochNNN_encoder_mas_durations.wav` / `_encoder_dp_durations.wav`) and prints a per-word
  table. Run on 004c-154c, wavs and tables (`inspect_4_142{3,8}.txt`) in `mas_vs_dp/` in the project root.
- Result: MAS is constant from epoch 4 to 154 ("or other" end: ˈʌ|ð|ə at 1 frame each; "Tuesday": j 50, ˈ 41,
  z 1). DP gives those symbols 5-7 frames up to 039c, then 1 frame from 064c on; in 4/1428 DP moves z/d into the
  pause from 064c on. So the DP converges onto the wrong MAS between epochs 39 and 64; MAS does not drift.
  Not yet confirmed by ear (MAS renders) and the timing differs from the user's 19-24.
- Confirmed by ear (user): the MAS-duration render is broken from the first checkpoints, the DP-duration render
  sounds great up to epoch 19. The DP "fixes the mel" with what it learned from other sentences, then learns this
  sentence's wrong MAS durations. The Encoder sounds are fine early; the audible error is the durations.
- Prevalence (temporary script, lost with /tmp; 150 train files per speaker): files with a run of >=4 / >=8
  consecutive one-frame symbols: epoch 19: 309 / 34 of 2250 (speaker 11: 38/19, 0: 44/6, 4: 14/2); epoch 154:
  107 / 9 (4: 7/2, 12: 14/0, 0: 16/3). Most runs disappear with training; a few stay.
- Why MAS is fooled (temporary script, lost with /tmp; 019c): locally the right Encoder symbol usually matches best
  (ɔ of "form" at 0.70-0.83, ð at the dip 1.11-1.13, ə 1.17-1.26), but voiced symbols' Encoder frames are close to each
  other (0.1-0.3 per-bin RMS), so a wrong voiced symbol costs only ~0.5-1.5 per frame. A voiced symbol on a silent
  frame costs 15-25 per frame. So the path is decided by which symbols have silence-like Encoder frames, and the
  Encoder can only guess that from the text. 4/1423: the silence-like symbols are the gap after "form" and its
  separators, not the gap after "other". 4/1428: the most silence-like reachable symbol is the separator between z
  and d (the d closure), not the gap after "Tuesday"; MAS puts the pause there and shifts "uːz" onto j, "deɪ" onto
  the stress mark. Proposed next: test a capped (robust) MAS score on the CPU.

## 13 November: why MAS misplaces a pause (mechanism)
- The user rejects changing the corpus to work around a model flaw (commas only where a comma belongs; Brian's
  pauses are often emphasis, part of the voice). The Whisper-time based comma detector was unusable (Whisper word
  times 0.2-0.4 s off) and is abandoned; `find_missing_commas.py` was deleted; `metadata.4_review*` in the corpus are obsolete.
- Test with `measure_pause_placement.py` (deleted 17 Nov): pauses = >= 80 ms below -50 dB inside the speech (matcha wavs). Reference position
  = MAS on the mel with all pauses cut out (no silence left to mislead it).
- With pauses cut out, MAS places every word of 4/1423 and 4/1428 CORRECTLY at 004c and 019c (form 0.56-0.84, or
  0.85-0.98, other 0.99-1.32, we 1.34; Tuesday 0.52-1.16, we 1.17). At 154c even the cut version is wrong: by then
  the Encoder has learned the wrong alignment (prior loss on the chosen path).
- 019c, 150 train files per speaker, 734 pauses at a space without punctuation: 89% on their gap, 3% on the
  neighbouring phoneme, 8% displaced (pauses at written punctuation: 96% on their gap). Per-frame cost of putting the
  pause frames on the gap: 1.5 when it lands right, 31 when displaced; the chosen competitor costs ~3.7, a median 7
  symbols away, mostly a separator. So: when the Encoder's guess for the space is "transition", the pause goes to the
  nearest silence-like symbol and the symbols in between are squeezed.
- Displaced per speaker (on gap / neighbour / displaced): 11: 63/12/27, 4: 29/2/7, 0: 31/0/6, 13: 4/0/4;
  3 (Bella): 131/1/4, 6: 250/5/5, 12: 125 with 1 displaced. Matches what the user hears (Brian, Lewis).
- 848 pauses are "inside a word" by the reference (stop closures or reference errors), not analysed yet.
- With the correct comma, at 019c AND 154c (never trained with it), MAS places 4/1423 and 4/1428 right; the pause goes
  mostly to the separator after the comma (8-17 frames), the comma symbol itself gets 2-5. 4/0219 (pauses after "of"
  and "bright") is NOT fooled: both gaps already look silence-like (cost 1.5-2.1 per frame).
- Explaining the mechanism to the user: the space IS where a pause belongs and the Encoder can model it as a pause
  (he corrected me sharply on "the pause has no symbol"). Say instead: most spaces carry no silence (a transition),
  so the Encoder guesses per space from context; when it guesses "transition" for the space at a real pause, the
  pause goes to the nearest silence-like symbol. The user rejects changing the corpus to work around this model flaw;
  commas only where a comma truly belongs (Brian's emphasis pauses, e.g. before "pink", are the voice's character).

## Ideas for an alignment fix (not tested, the user has not chosen)
- Pause-cut MAS in training: run MAS on the mel without pauses, then give each pause to the space at the cut point
  (needs a better cut-point rule, see the 11/1111 weakness).
- Cheap silence for every space: in the MAS score, any space (and its separators) may take silent frames at low cost,
  whatever its Encoder frame looks like.
- Capped (robust) per-frame MAS score, so a silent frame cannot outweigh many phoneme frames.
- Older tickets still open: soft alignment / forward-sum loss, curriculum, separate RAD-TTS-style aligner.

## Corpus fixes (corpus project)
Corpus project: `E:\Programare\Corpus` (WSL `/mnt/e/Programare/Corpus`), Windows venv `.venv/Scripts/python.exe`.
Speakers are synthetic voices (Polly, Azure, Kokoro); `Training data/metadata.N.csv` per speaker. Read files there
with the shell; the IDE file reader hung on that drive once.
- `find_pauses.py`: the missing comma detector. Pauses come from the wav alone (5.3 ms frames, quieter than
  `--silence_db`, at least `--min_pause_ms`, inside the speech); the transcript is force-aligned with Meta's MMS model
  (`facebook/mms-1b-all`, CTC, one adapter per language), and each pause goes to the word gap it overlaps most.
  Writes `<stem>_pauses_generic.txt` and `<stem>_commas_generic.csv` (identical to the input except the proposed
  commas, for IntelliJ's diff). Its header explains how to choose the two values. Its `--matcha_list` option compares
  with the list of this project's old detector, which no longer exists.
- `validate_corpus.py` (Whisper large-v3) and `asr_review.py` with the rules in `ASR_diff_review.md`: checks that the
  voice said what the transcript says.
- `sync_splits.py`: brings the edits of `metadata.N.csv` into the train / validate splits.
- `bookmark.md` holds the user's prompts for each of these; follow them (they say when `metadata.N.csv` may be
  edited, and only after a backup). Remind the user to rerun `validate_corpus_ipa` here afterwards.
- The user's rule for commas (decided on Bella, 14 Nov): a comma wherever the voice really pauses, so the model learns
  "comma = pause" and a voice's quirk pauses do not leak into plain spaces at inference. Proposals are sorted into
  expected and unexpected ones as described in the bookmark prompt.
- Speaker audio, useful for choosing the values (pause level inside speech / values used):

  | Speaker | Pauses inside speech | Length / floor used |
  |---|---|---|
  | 11 Lewis | digital zero (noise gate); comma pauses 170-200 ms | 110 ms / -90 dB |
  | 4 Brian | ±1-4 LSB noise, -93..-105 dB RMS; at 110 ms half the hits were stop closures | 110 ms / -90 dB |
  | 3 Bella | starts with a -67..-72 dB tail, decays to -80..-90 dB; comma pauses median 203 ms | 140 ms / -65 dB |
  | 7 Andrei | -72..-94 dB; often does not pause at commas | 110 ms / -64 dB |
  | 13 Isabella | -60..-80 dB; Italian double consonants look like pauses below ~150 ms | 200 ms / -75 dB |
  | 2 Aria | -55..-70 dB noise floor, peaks 20-80 LSB | 120 ms / -60 dB, nothing found |
  | 5 Arthur | constant -57..-61 dB noise floor, comma pauses median 299 ms | 110 ms / -50 dB, only closures |
  | 6 Nicole | edges -50..-55 dB, middle -85..-90 dB; pauses at phrase boundaries in nearly every sentence | 200 ms / -50 dB (find_pauses) |

  Below about 110 ms stop closures (before t, k, p, d, b, g) start to look like pauses.
- eSpeak drops spaced dashes ("slowly - very slowly -" has no punctuation id), so the model sees no mark there.
- Speaker 7 (Andrei) audio anomalies (noise after the speech, laughter, hard cuts) were found with a temporary scan
  and fixed by the user on 16 Nov; stats, mels and CSVs regenerated then (mel_mean -4.669398, mel_std 6.485294).

## Extra notes for the assistant (not yet elsewhere)
- Speaker names (`matcha/inference.py`): 0 Kai, 1 Jane, 2 Aria, 3 Bella, 4 Brian, 5 Arthur, 6 Nicole, 7 Emil,
  8 Denise, 9 Henri, 10 Matthew, 11 Lewis, 12 Michael, 13 Isabella, 14 Marcello. Lewis is en-us there.
- Languages per speaker: en-us 0, 1, 2, 3, 6, 10, 12; en-gb 4, 5, 11; ro 7; fr-fr 8, 9; it 13, 14.
- Annotation inventory (train.csv): en stress 128k, length 55k, secondary stress 22k, syllabic 291; fr stress 23k,
  nasal tilde 11k; ro stress 13k, secondary 6k, palatalization 2k; it stress 16k, length 2.7k (mostly on consonants,
  double consonants; `ː` before stress 654).
- Word boundaries, epoch 789, 50 train files per speaker: flowing boundaries (3692, no silent frame) give closing
  separator / gap / opening separator 16 ms median each; gap at 1 frame only 5%. Boundaries with >= 100 ms silence
  (94): silent frames on gap 26%, separators 37%, inside the neighbouring phoneme groups 35%; the gap holds most of
  the pause in only 22%. So pauses are spread, and a third of them sits inside words: the main target for noise or
  pause anchors, independent of the scheme.
- Claims I made and corrected, do not repeat: "the three symbols at a word gap are almost identical" (I had only
  compared them with silence; they differ like any neighbours); "the length mark takes over the long vowel's steady
  part" (refuted: ~20%); "carry stress as a vowel feature" (against the user's principle); "the errors form before
  epoch 124 and lock in, so noise from epoch 0 should prevent them" (4/1423 is wrong after epoch 0, and noise did not
  help); "the first v24 run is the baseline for early epochs" (different batch size and loss settings, see above).
- CPU MAS + Encoder on 2100 files takes ~70 s on 8 threads, safe while training holds the GPU.

## 15 November: first run with the comma-corrected corpus (speakers 0, 1, 2, 4, 11, 13)
- Run `logs/train/v24/runs/2026-11-15_09-34-08/`, new scheme, from epoch 0, checked at epoch 4.
- The user reran the leading/trailing silence normalization on 15 Nov; times in this section are ~0.1 s later than in
  the older sections (4/1423 pause now 1.47-1.60 s, 4/1428 pause 1.29-1.52 s). Speaker 7 is a new voice (Andrei).
- 4/1423 and 4/1428: every word placed right by the alignment search (MAS), identical to the pause-cut reference,
  the pause goes to the comma and its separator. In the comma-free new-scheme test (12 Nov) they were wrong from
  epoch 1 and 2. Still to check on later checkpoints (the Duration Predictor drifted between epochs 39 and 64 before).
  Renders and tables: `mas_vs_dp_commas/`.
- 11/1111 (Lewis): the pause tool calls the pause after "degree," displaced (silence 1.29-1.45 s held by the end of
  ɹ, the stress mark, i, ː and the comma, with i, ː, comma at 1 frame each). The user hears the MAS-duration render
  as correct, so the Encoder gives the "ee" sound to ɹ and the stress mark; judging it wrong from the symbol table
  assumed the sound belongs to i, which it need not. Not an error. The pause after "University," is right.
- Collapsed vowels (new definition, 30 validation files per speaker): 35.3% (12 Nov comma-free run: 29% at epoch 3).
  Not a usable comma measure at epoch 4: many "collapsed" vowels have their sound in a neighbouring separator or
  stress mark ("form": ɔ 1 frame, stress mark 18).
- Pauses that move words (150 train files per speaker): 13 of 2126. This run gives word-gap silence mostly to the
  separators next to the gap, so the pause tool's on gap / neighbour split is not a quality measure; words moved is.
- Bella (3) has no commas yet in the CSVs: 180 space-only pauses in 150 files.

## 15 November evening: run 2026-11-15_13-26-57 (commas, Andrei as speaker 7, new Romanian), epochs 4-29
- Renders and tables for 4/1423, 4/1428, 11/1111 at 004-029: `mas_vs_dp_029/` (project root, untracked).
- 4/1423, 4/1428, 11/1111: MAS word placement right and stable at every checkpoint 4-29; the pause goes to the comma,
  the gap, the separator after it and the first frames of "we" (w holds ~90 ms of the 4/1428 pause). DP word lengths
  within a few frames of MAS (except the final silence, which DP makes shorter).
- Files with runs of consecutive one-frame symbols (MAS), first 150 train files per speaker (temporary script,
  deleted): >=4 / >=8: epoch 4 539/31, 19 352/14, 29 307/9 (comma-free 13 Nov run: 19 309/34, 154 107/9; samples
  not identical). Speaker 11 >=8: 1 (was 19). DP >=4 runs: 0, 1, 8 at 4, 19, 29 (first sign of DP copying MAS).
- Still broken: 0/0109 (from epoch 4, words after "opinion" placed late, "strong" 1.5 s; silences 3.33-3.78 and an
  unpunctuated one 4.01-4.18 s; maybe transcript mismatch, user to listen). 7/0035 "trebuie" 12 symbols in 107 ms.
  Speaker 7 has the most >=8 runs (5/150 at 29).
  User listened to 0/0109 at 029: the MAS-duration render is bad, the DP-duration render is right. He expects the DP
  to keep generalizing over this one wrong file. TRACK 0/0109 at later checkpoints (add it to `--track`, render DP).
- Pauses that move words, epoch 29, 150 files per speaker: 10 (9 by one word, 1 by two).

## 16 November: same run at epoch 134 (checkpoints 004-039, 094, 124, 134 now in `logs/train/v24/`)
- 4/1423, 4/1428, 11/1111: MAS word placement unchanged and right at 39, 94, 124, 134; DP within a few frames
  of MAS. Renders: `mas_vs_dp_134/`. Boundary movement per 100 epochs: 29→39 118%, 39→94 22%, 94→124 24%, 124→134 63%.
- One-frame runs (MAS), 150 train files per speaker, >=4 / >=8: 160 / 5 (epoch 29: 307 / 9). DP >=4 / >=8: 11 / 2,
  10 / 2 of them speaker 7. Every other speaker: at most 1 DP file.
- 0/0109: MAS still wrong (from 94 on the pause after "on" goes to the gap, "Terra." still after it). DP partly
  moving towards it: how 73→39 frames (MAS 19), strong 149→177 (MAS 284), you/know 23/29→17/17 (MAS 7/7).
- Speaker 7 (Andrei): the runs are mostly "trebuie"/"trebuit" (95 of 2085 train lines). In 7/0035 the audio really
  is short (t closure ~0.31-0.36 s, word over by ~0.43 s), so MAS squeezes b, u, j, e to 1 frame; DP copies it
  (7/0035, 7/0070 DP run 8). Real fast reduction, not a misplaced pause; user to listen.
- Pauses that move words: 9 (7 by one word, 2 by two); at 29: 10.

## 17 November: run "b" (all corpus fixes except speakers 10, 12, 14) against run "a" (only 4 and 11 clean)
- Run a = `2026-11-15_13-26-57` (`checkpoint_epoch={099a,134a}.ckpt`), run b = `2026-11-16_21-09-01`
  (`{064b,084b,099b}.ckpt`, still training). Same config except mel stats. Both measured on the CURRENT CSVs and mels
  (16 Nov 21:05), so run a is measured partly on text it never trained on (new commas, fixes) and maybe on different
  silence lengths: its Duration Predictor gives the leading silence ~0.05 s, the wavs have ~0.18-0.20 s (run b: 0.19).
- Numbers (validation 100 files/speaker for collapsed vowels, 50 for the rest; one-frame runs on 150 train/speaker):

  | | 099a | 134a | 064b | 084b | 099b |
  |---|---|---|---|---|---|
  | Collapsed vowels (new definition) | 30.9% | 29.8% | 37.2% | 36.4% | 36.1% |
  | Duration error (log scale) | 0.204 | 0.201 | 0.182 | 0.181 | 0.177 |
  | Files with ≥4 / ≥8 one-frame symbols, MAS | 154 / 6 | 173 / 6 | 60 / 7 | 60 / 8 | 56 / 7 |
  | Same, Duration Predictor | 14 / 2 | 10 / 2 | 3 / 0 | 6 / 0 | 7 / 1 |
  | Pauses that move words | – | 11 | – | – | 7 |

  Boundary movement per 100 epochs: a 99→134 20.9%; b 64→84 22.9%, 84→99 28.4%. One-frame runs per speaker, 134a →
  099b: Bella (3) 37 → 2, Kai (0) 31 → 13, Michael (12) 23 → 2, Andrei (7) 22 → 23 ("trebuie", real reduction).
  Pause reference (pause-cut MAS) lands at written punctuation 1165 times in b, 619 in a (2149 pauses).
- Run b's whole alignment sits LATER than run a's on the same mels: end of each symbol median +69 ms, 79-91% of
  boundaries later by >10 ms, every speaker; by position in the sentence (tenths) 48, 85, 96, 85, 85, 85, 75, 48, 5,
  0 ms. b's symbol j ends nearest to a's symbol j+3 (median 16 ms). Sibilants (English speakers, high-minus-low mel
  bands): run a gives the s sound to s and the separator after it, run b to the separator before s. 4/1423 at the
  acoustics (s 0.35-0.46, f 0.65-0.76, pause 1.47-1.59): run a's words start on time, run b's start one onset sound late
  ("some" holds the f) and catch up at the pause. Not judged as an error; the per-symbol metrics (collapsed vowels)
  change with it. Ears decide: renders in `mas_vs_dp_099b/` (134a and 099b).
- Tracked files at 099b: 4/1428 right (pause on the comma and gap, "we" at 1.52 s; run a lets "we" start at 1.32
  inside the pause). 11/1111: pause after "University," on the comma (run a: inside "in"). 4/1423 pause right in
  both. 0/0109 still wrong in both (MAS: "on" 0.73 s, "Terra." 8 frames, "You", "know" 7 each); run b's Duration
  Predictor follows MAS more than run a's (Terra 14, You 22, know 32 vs 25, 17, 17).
- Temporary scripts (deleted): one-frame runs, a-vs-b shift, sibilant placement.

## 18 November: run b at 264-309, the rising 100th percentile of the duration error
- The chart is the training-set largest duration error per batch (log scale), averaged over the epoch. Values: epoch 4
  3.04, 179 2.981, 249 3.001, 274 2.984, 309 2.996. It has been about 3.0 since epoch 4; the rise is 0.5%, within its
  usual wobble. Every other percentile keeps falling.
- The largest error in every batch is the trailing silence (0.77 s, about 144 frames): it is assigned to one of the last
  three symbols (last consonant, separator, final punctuation), and the Duration Predictor gives each of them ~3.7
  frames at every checkpoint (64 to 309). log(148) - log(7.7) = 2.96.
- Word alignment is not moving: 4/1423, 4/1428, 11/1111, 0/0109 identical at 264, 279, 294, 309 (0/0109 still wrong);
  boundaries moved >2 frames per 100 epochs 16.0%, 14.0%, 16.1% (64-134 was 20-28%); collapsed vowels 35.2% → 34.9%.
- Why the Duration Predictor never learns the trailing silence: in a training step MAS runs on the Encoder output with
  dropout on. Over 5 dropout draws on 300 train files (309, GPU kernel, bf16 autocast as in training) the silence
  went to the 3rd-last / 2nd-last / last symbol 40% / 31% / 29% of the time, and changed owner in 296 of 300 files. No
  symbol holds it more than half the time, and the Huber loss is in its linear region there (median-like), so the
  best prediction for each is "short". Trained alone on fixed (eval-mode) MAS targets, the Duration Predictor learns
  it within 100 steps (CPU test, fp32 and bf16 alike), so the model can learn it.
- Dropout during MAS is also a hidden alignment noise: between two draws 52% of the symbol durations change, 5.3% of
  the boundaries by >2 frames (about VITS2 noise scale 0.003-0.01, see the noise table). It has been in every run.
- bf16 autocast alone (the score's matmul runs in bf16) changes 26% of the symbol durations against fp32, mostly by
  one frame. So "CPU MAS identical to the GPU kernel" holds in fp32 and eval mode only; the diagnostic tools show the
  eval-mode alignment, not what a training step sees.
- Temporary scripts (deleted): largest-error symbols, last-symbol durations, Duration Predictor learnability test,
  GPU vs CPU MAS, dropout draws.
- Correction: the rise in the chart is a real trend (minimum ~2.981 around steps 100-180k, back to ~3.0 by 225k), not
  wobble. It is still the trailing silence's error; its exact cause is not measured.
- fp32 restored in `MatchaTTS.find_alignment` (the user's block from commit e4d9a5e, removed in bb34f3e on 1 Aug):
  autocast off, Encoder output and mel cast to fp32. With the Encoder output rounded to bf16 as in training, the
  durations now differ from pure fp32 for 0.6% of the symbols (was 26%). Dropout noise unchanged: 47.5% of the
  durations change between draws, 3.9% of the boundaries by >2 frames; the trailing silence still changes owner in
  288 of 300 files. Not yet trained with.
- Test runs, 18 Nov (all 5 epochs, `logs/train/v24/runs/`): A `2026-11-18_14-30-24` from 0, alignment without dropout;
  B `2026-11-18_15-12-39` from 0, fp32 only; C `2026-11-18_15-57-58` resume 309, both; D `2026-11-18_16-45-18`
  resume 309, fp32 only. A worse than B at epoch 4 (val prior 0.660 vs 0.609, collapsed 36.7% vs 31.3%); C broke at
  epoch 312 (train prior 0.39 → 1.07, never recovered); D stable and better than run b at 314 (train duration loss
  0.0393 vs 0.0434, val prior 0.468 vs 0.470). B at epoch 4 beats run b (bf16) at 4: val duration 0.057 vs 0.070.
  The user keeps dropout (it gives variation, like the duration jitter); alignment without dropout is dropped.
- fp32 history: MAS fp32 added 5ae33d2 (Dec 2025), removed b0adceb; re-added e4d9a5e (Apr 2026, "prior loss shoots
  up, duration loss +60% after ~100 epochs"), removed 23a013c (4 Jun, compilation move), comment deleted bb34f3e.
  Encoder in bf16 vs fp32 (eval, 314D): mel guess differs by 0.0017 on average (values ~0.63); the fp32 alignment on
  it differs for 2.9% of the durations, 0.46% of boundaries >2 frames. Duration Predictor: 0.0049 log, 0.056 frames.
- Who takes the edge silences (309, eval, fp32, 100 train files per speaker): leading silence always the first symbol
  (the space). Trailing: final punctuation 79%, separator before it 15%, last phoneme 5%; the owner takes all ~144
  frames, the other two 1 each. Worst: Aria (2) separator 62%, Brian (4) last phoneme 24%, Marcello (14) separator
  35%. The same text goes to different owners for different speakers (Harvard "...sentences.": 2 punctuation side, 4
  on z). Between 124 and 154 the majority owner moved from the separator to the punctuation.

## 18-19 November: new run from scratch with the alignment in fp32, and the Encoder precision test
- Run `logs/train/v24/runs/2026-11-18_17-57-39`: from epoch 0, alignment score in fp32, Encoder in bf16. At epoch 29
  against run b (bf16 alignment) at 29: validation duration loss 0.0293 vs 0.0585, prior 0.482 vs 0.502, Decoder
  0.082 vs 0.081. The Duration Predictor now predicts the trailing silence (~164 frames; run b ~4).
- Encoder precision test, all three from that run's checkpoint 19 to 29: the original run (Encoder bf16), a repeat
  `2026-11-18_22-49-14` (Encoder bf16, the noise reference), `2026-11-18_21-25-23` (Encoder fp32, option
  `model.encoder_fp32=true`). CPU numbers on 50 validation files per speaker.

  | Epoch 29 | bf16 original | bf16 repeat | Encoder fp32 |
  |---|---|---|---|
  | Validation prior loss | 0.4824 | 0.4780 | 0.4753 |
  | Validation duration loss | 0.0293 | 0.0295 | 0.0290 |
  | Validation Decoder loss | 0.0819 | 0.0829 | 0.0804 |
  | Duration error (CPU, log scale) | 0.1504 | 0.1524 | 0.1496 |
  | Collapsed vowels | 25.7% | 25.8% | 26.7% |
  | Files with >=4 / >=8 one-frame symbols, alignment | 69 / 6 | 77 / 9 | 70 / 5 |
  | Same, Duration Predictor | 3 / 0 | 12 / 0 | 3 / 0 |
  | Boundaries moved >2 frames, 19 to 29, per 100 epochs | 96% | 89% | 80% |
  | Seconds per epoch (no validation) | 392 | 391 | 417 |

  The two bf16 runs differ by 0.9% in prior loss; fp32 is 0.6% better than the better of them. Within noise or
  barely above it, costs 6.4% speed. Tracked files 4/1423, 4/1428, 0/0109, 0/2649, 0/0255 place every word the
  same in all three. 0/0109 still wrong ("Terra. You know how" squeezed); 0/2649 "have you got" squeezed
  (0.29-0.43 s).

## 19 November: utterance start and end symbols ("‹" replaces the leading space, "›" follows the final punctuation)
- Run `logs/train/v24/runs/2026-11-19_09-32-47` (149 symbols; the CPU tools now load only this scheme), compared
  with `2026-11-18_17-57-39` (leading space, no end symbol), same CSV texts, both with the fp32 alignment.
- Epoch 4: validation duration loss 0.0387 vs 0.0885, prior 0.578 vs 0.618; median duration error 0.144 vs 0.170.
  "‹" and "›" take the edge silences in 750 of 750 files from epoch 4 on.
- The old run catches up between 14 and 19 (its duration loss drops 0.083 → 0.031, when its trailing silence settles
  on the punctuation). At 29: validation duration 0.0286 vs 0.0293, prior 0.4843 vs 0.4824, Decoder 0.0788 vs
  0.0819; CPU (50 validation files per speaker): duration error 0.1481 vs 0.1504, collapsed vowels 27.3% vs 25.7%,
  files with >=4 / >=8 one-frame symbols, alignment 87/7 vs 69/6, Duration Predictor 0/0 vs 3/0; trailing silence
  predicted 158 frames (alignment 146). Boundaries moved >2 frames 19 → 29: 46% per 100 epochs (old run 96%).
- Word tables from the tools now include the edge silences in the first and the last word ("In 0.00-0.34"), because
  "‹" is not a word gap.
- 0/0109 changed: "Terra." 3.66-3.82 s (old run 3.76-3.81), "on" still runs into the pause. Not listened to yet.
