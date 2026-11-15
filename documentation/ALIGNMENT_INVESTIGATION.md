# Alignment investigation (v24/v25, November 2026): context for continuing in a new chat

Written for the assistant, not as user documentation. Read `.amazonq/rules/assistant-rules.md` and
`documentation/components.md` / `documentation/PROBLEMS.md` first.

KEY FINDING (13 Nov): the Duration Predictor (DP) starts fine, then learns the alignment search's (MAS) mistake as
training progresses. MAS is wrong for 4/1423 and 4/1428 from the first epoch and gives the same wrong durations at
every checkpoint (004c-154c). The Encoder mel built with the DP durations sounds right up to about epoch 19 (user's
ears), because the DP gives durations it learned from other sentences. As training goes on, the DP fits this
sentence's wrong MAS durations, and the error becomes audible and stable. MAS does not drift; the DP does. Details:
"13 November: alignment search (MAS) or Duration Predictor (DP)?".

WHERE WE STOPPED (13 Nov, 19:10; corpus work continued 14 Nov, see 'Missing comma detector: state'):
- New phonemization scheme KEPT (user: voices sound much better, timbre and intonation). Still uncommitted on branch
  `v24`; commit/branch decision open, ask the user.
- The user brought corrected transcripts for speakers 0, 4, 11 from the corpus project, removed speaker 15, deleted
  the old CSVs and regenerated `data/corpus-24k/train.csv` (34954 lines) and `validate.csv` (2623) with
  `validate_corpus_ipa` on the NEW scheme (max id 144, separator id 0, every line has a mel, speakers 0-14). Training
  now uses them directly: `python -m matcha.train +experiment=v24` (no `data=corpus-24k-v25`). Obsolete, deletion
  offered but not yet approved: `configs/data/corpus-24k-v25.yaml`, `train.v25.csv`, `validate.v25.csv`.
  `configs/experiment/v24.yaml` still checkpoints/validates every epoch (short-test setting).
- Training is stopped; the user wants to finish corpus fixes before starting the next run from epoch 0. Done today
  by him: commas in 13/0050 and 13/1146 (found by my pause list), Brian, Kai, Lewis via ASR, 11/1111 by ear.
- Next topic: fixing MAS itself (see "Ideas for an alignment fix" below), with Lewis (11) as the test case.
- Untracked review outputs in the project root: `mas_vs_dp/` (renders), `displaced_pauses.txt` (speakers 1-3,
  5-10, 12-14), `unpunctuated_pauses_11.txt` (Lewis). In the corpus folder `metadata.4_review*.csv/txt` are obsolete.

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
  the separator scheme is unique. Do not propose them again.
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
- All in `matcha/utils/`, CPU only unless stated, safe while training holds the GPU. Committed in ace1aa9 in their
  old-scheme form; `alignment_diagnostics.py` and `measure_collapsed_vowels.py` are now modified (uncommitted) for
  the NEW scheme:
  - `alignment_diagnostics.py`: shared helpers (load Encoder from a checkpoint, CPU MAS identical to the GPU
    kernel, word segments, Audacity labels). New scheme: `symbol_label` = the symbol itself, `phoneme_groups` = one
    segment per symbol (separators are shared, not merged), `vowel_spans` (replaces `vowel_core_positions`) = a vowel
    plus the length mark and diacritics written right after it; a stress mark before the vowel is left out.
  - `measure_collapsed_vowels.py -i <data yaml> -c <ckpt> [--split train] [--files_per_speaker N]`. New definition:
    a vowel span of 3 frames or fewer. NOT comparable with the old triple-based numbers.
  - `compare_alignment_over_checkpoints.py -i <data yaml> -c <ckpt...> --track 4/1423 4/1428`: collapse share,
    duration error, word placement of tracked files, boundary movement per 100 epochs between consecutive checkpoints.
    Works for the new scheme with `-i configs/data/corpus-24k.yaml`.
  - `measure_alignment_noise.py` (untracked, GPU): noise sweep and pinned-path test (`--pin 4/1423:4:1.38:1.48
    4/1428:1:1.19:1.42` = recording:word index:pause start:pause end; the word gap after that word is forced onto the
    pause). Imports `count_collapsed_vowels`, so it follows whatever scheme `symbols.py` has.
  - `inspect_alignment.py`: NEW scheme; per checkpoint, symbol and word tables of MAS vs DP durations, `--wav_dir`
    renders the Encoder mel with MAS and with DP durations (file names `_encoder_mas_durations` / `_encoder_dp_durations`;
    never call MAS output "ground truth"), `--labels_dir` writes `_mas_labels.txt` / `_dp_labels.txt`.
  - `measure_pause_placement.py`: where MAS puts each pause and what it costs; `-u` single files, or a corpus sample /
    all files with `--speakers`, `--list_displaced`, `--list_unpunctuated`.
  - `align_with_text.py -u <utt> --text "<alternative>" -c <ckpt...>`: MAS and DP word placement with a changed text.
  - `measure_alignment_by_symbol_class.py` still assumes the OLD scheme and fails to import.
  - To read old-scheme checkpoints again, the old `symbols.py` and the old versions of the two modified scripts are
    needed (e.g. `git stash`, or a worktree of commit aad0c56).
- `/tmp/print_tb.py <tensorboard dir...>` prints the per-epoch duration / prior losses of runs side by side (temporary,
  /tmp may be wiped; ~20 lines with tensorboard's EventAccumulator).
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

### State of the code (end of 12 November; the CSV parts are superseded by "WHERE WE STOPPED")
- The working tree is on branch `v24` (last commit aad0c56) with these UNCOMMITTED changes:
  - The new phonemization, brought over from branch `v25` by the user: `matcha/text/symbols.py`
    (single separator id 0, `annotation_ids`, `SEPARATOR_ID`, 147 symbols instead of 600), `matcha/text/phonemizers.py`
    (`insert_separators`), `tests/test_phonemizers.py` (98 pass). So the working tree now phonemizes with the NEW scheme,
    and `data/corpus-24k/train.csv` / `validate.csv` (old ids) no longer match the code.
  - `configs/experiment/v24.yaml`: checkpoints and validation every epoch (user's change for the short tests).
  - `matcha/utils/alignment_diagnostics.py` and `matcha/utils/measure_collapsed_vowels.py` rewritten for the new
    scheme (see Tools).
  - Untracked: `configs/data/corpus-24k-v25.yaml`, `matcha/utils/measure_alignment_noise.py`.
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

## 13 November: corpus fix first (missing commas, transcript errors)
- The user prefers fixing the corpus before changing MAS. Corpus project: `E:\Programare\Corpus` (WSL
  `/mnt/e/Programare/Corpus`), Windows venv `.venv/Scripts/python.exe` with faster-whisper. Speakers are synthetic
  voices (Polly, Azure, Kokoro); `metadata.N.csv` per speaker in `Training data/`.
- Pipeline: `validate_corpus.py` (Whisper large-v3, now also writes `<stem>_asr_words.json` with word timings) →
  `find_missing_commas.py` (audio-based pauses, per-speaker threshold, writes `<stem>_review.csv`) → assistant removes
  benign differences with a log → the user diffs `metadata.N.csv` against `_review.csv` and edits the corpus HIMSELF.
  Never modify `metadata.N.csv`. Pilot: speaker 4, check 4/1423 and 4/1428.
- FOUND: the corpus already has "In some form or other, we need fun." and "Next Tuesday, we must vote." (the user
  fixed Brian with ASR earlier), but `data/corpus-24k/*.csv` (and the v25 copies) hold the old text: 76 speaker-4
  lines differ (66 train, 10 validate; 35 of them only commas); 10 matcha train lines no longer exist in the corpus.
  So the model never trained on the Brian corrections. Regenerating the matcha CSVs is the user's decision.
- Whisper run for speaker 4 done (10.6 min): `metadata.4_asr.csv`, `metadata.4_asr_words.json`.
  `find_missing_commas.py` (Whisper word times) was later abandoned and deleted, see below.

## 13 November: why MAS misplaces a pause (mechanism)
- The user rejects changing the corpus to work around a model flaw (commas only where a comma belongs; Brian's
  pauses are often emphasis, part of the voice). The Whisper-time based comma detector was unusable (Whisper word
  times 0.2-0.4 s off) and is abandoned; `find_missing_commas.py` was deleted; `metadata.4_review*` in the corpus are obsolete.
- Test `matcha/utils/measure_pause_placement.py` (`-u` for single recordings): pauses = >= 80 ms below -50 dB inside the speech (matcha wavs). Reference position
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

## 13 November evening: full-corpus pause lists (019c, current CSVs)
- `measure_pause_placement.py --files_per_speaker 0 --speakers ... --list_displaced <file>` and
  `--list_unpunctuated <file>`. ~10 min for 12 speakers on the CPU.
- Speakers 1-3, 5-10, 12-14: 30883 pauses; 439 displaced. "Likely fine" (149): MAS gave the pause to a punctuated
  gap, the reference is one word late (e.g. "Max. I ⟨⟩ want"; mostly 1, 8, 9, 13). "Likely real" (290): 6 Nicole 172
  (41 >= 150 ms), 3 Bella 75 (28), 12: 18, 9: 11, 8: 7, 1: 3, 13: 3, 5: 1; most are 80-120 ms at stop closures.
  Isabella's 3: two real missing commas (user confirmed), 13/0482 "N A T O" spelled with a pause before O (no comma).
  Jane's 3: two right after a full stop (reference late), one t-closure; nothing to fix.
- Lewis (11): 81 pauses at punctuation (median 315 ms), 860 at plain spaces (226 >= 200 ms), 302 harmful (35%;
  Brian 18%, Bella/Nicole 2-3%). Most are where no comma belongs ("acclaim for ⟨⟩ her", "intake to ⟨⟩ at least").
  Options offered, none chosen: leave him (test case for a MAS fix), shorten unpunctuated pauses in his audio (on
  copies), or leave him out. Unverified suspicion: his gaps are digital zero (noise gate); check before deciding.
- Weakness of the reference position: when the words around a pause sound alike, the cut lands one word off. Example
  11/1111: real pauses after "degree" (1.19 s, 160 ms) and "University" (2.54 s, 400 ms); my list said "in ⟨⟩
  nineteen". In the lists the TIME is reliable, the words around ⟨⟩ can be off by one; tell the user to listen at
  the time. 11/1111 with the user's commas ("degree, from Yale University, in"): at 019c MAS still puts the 400 ms
  pause inside "in" (unlike Brian's sentences, the comma alone does not fix it in a model not trained on it).

## Ideas for an alignment fix (not tested, the user has not chosen)
- Pause-cut MAS in training: run MAS on the mel without pauses, then give each pause to the space at the cut point
  (needs a better cut-point rule, see the 11/1111 weakness).
- Cheap silence for every space: in the MAS score, any space (and its separators) may take silent frames at low cost,
  whatever its Encoder frame looks like.
- Capped (robust) per-frame MAS score, so a silent frame cannot outweigh many phoneme frames.
- Older tickets still open: soft alignment / forward-sum loss, curriculum, separate RAD-TTS-style aligner.

## Missing comma detector
Finds the places where a speaker pauses but the transcript has no punctuation. It is meant for voices whose pauses
should be marked with a comma so the model learns where this voice pauses, whether or not the comma is grammatical.
It is part of `matcha/utils/measure_pause_placement.py`, runs on the CPU (about 1 minute per 1500 recordings) and
never modifies any transcript.

How the user uses it: he asks the assistant to run it for a speaker number, a target pause length in ms and a
silence floor in dB. The assistant runs it, spot-checks the result (levels around a few listed pauses, odd entries,
skipped files) and reports anything suspicious, instead of the user running the command himself.

How it works:
- A pause is a stretch of at least the target length (`--min_pause_ms`) quieter than the silence floor
  (`--silence_db`) inside the speech, measured directly in the wav on 5.3 ms frames.
- Silence floor: the user measures pauses in Audacity as the visibly flat line with the waveform zoomed in, which is
  digital zero. -90 dB matches that (checked on 11/0003: flat line ~60 ms, below -90 dB 64 ms). -50 dB is far too
  generous: it adds the fade-out of the previous word (11/0003 read 139 ms).
- To find which word gap a pause belongs to, the alignment search (Monotonic Alignment Search, MAS) of a trained
  checkpoint is run on the mel twice: as in training, and with every pause cut out. The second one is the reference:
  without silence it follows the speech alone.
- A pause counts as covered when the reference or the normal MAS puts it at punctuation. Everything else is listed.

Command (reads the corpus `metadata.N.csv` directly and phonemizes it on the fly):
```
source .venv/bin/activate
python -m matcha.utils.measure_pause_placement -i configs/data/corpus-24k.yaml \
    -c logs/train/v24/checkpoint_epoch=019c.ckpt --files_per_speaker 0 \
    --transcript "/mnt/e/Programare/Corpus/Training data/metadata.11.csv" \
    --list_missing_commas missing_commas_11.txt --min_pause_ms 110 --silence_db -90 \
    --proposed_transcript "/mnt/e/Programare/Corpus/Training data/metadata.11_commas.csv"
```
`--proposed_transcript` writes a copy of the speaker's transcript that differs from it only by the proposed commas
(same lines, same order, empty lines and line endings kept), for IntelliJ's diff view: corpus file left, proposal
right, F7 jumps to the next proposal. A comma goes after the word before each listed pause, except for `[comma?]`
pauses and lines whose words cannot be matched to the phonemized words. eSpeak joins short words ("of the"), so
transcript words are matched to phonemized words by their phoneme counts. When the reference puts a pause inside a
word, the comma goes one word earlier: checked against the user's hand-edited Lewis transcript, the reference was
one word late in 262 of 297 such cases. On the uncorrected Lewis transcript, 862 of 925 proposed commas matched the
user's own edits.
To check the training filelist instead, leave out `--transcript` and add `--speakers <id>`.

Requirements: the recordings must have precomputed mels in `data/corpus-24k/mels` (lines without one are skipped
and listed), and the wavs are read from `data/corpus-24k/wav`, so all times refer to those wavs, not to the corpus
copies (their leading silence differs, by about 0.03 s for speaker 11). Any checkpoint of the current phonemization
scheme works; an early one (019c) is preferable, because a late one has learned the old misplacements.

Output: one entry per recording, in recording order, with its transcript, then one line per unexplained pause: start
time, length, a mark, and the words around the pause (⟨⟩ marks the pause). Marks:
- none: the pause is at a space without punctuation, the main case;
- `[word]`: the reference puts the pause inside a word, mostly a word gap it missed by one word;
- `[comma?]`: a comma sits just before ⟨⟩ or one word earlier, so it probably already holds the pause.
The time is reliable, the words around ⟨⟩ can be one word off (when neighbouring words sound alike, e.g. "University
in"), so listen at the given time in Audacity before editing.

### Missing comma detector: how the assistant runs it (instructions from the user)
- The user asks with three values: speaker number, target pause length (ms), silence floor (dB). Run it with
  `--transcript` on the corpus `metadata.N.csv`, `--list_missing_commas missing_commas_N.txt` in the matcha project
  root and `--proposed_transcript` next to the corpus file (`metadata.N_commas.csv`). Never modify `metadata.N.csv`.
- Before reporting, spot-check: a few listed pauses against the level profile of the wav (is the pause the flat
  digital-zero line, or a word's fade-out / a stop closure?), lines skipped for missing mels or unmatched words, and
  anything odd. Report the counts, a few example proposals and what to watch for.
- Useful context for choosing the length: the speaker's own pause at written commas (Lewis: real commas 170-200 ms
  at -90 dB). Below ~110 ms stop closures (before d, t, k, p, b, g) start to show up as pauses.
- The user liked the detector as it is. When he asks for a different PRESENTATION, change only the presentation;
  do not touch which pauses are detected without asking (he worried I had changed detection when I only changed
  the word placement of the commas).
- The diff file is the point: the user reviews `metadata.N.csv` (left) against `metadata.N_commas.csv` (right) in
  IntelliJ and jumps with F7. It must stay byte-identical except for the proposed commas.

### Missing comma detector: state (14 Nov)
- Lewis (11) is DONE. The user added commas by ear wherever Lewis pauses, then with the detector at 110 ms / -90 dB.
  At 90 ms the detector proposed 39 more lines; the user judged those as prosody (his real commas are 170-200 ms)
  and left them. `metadata.11_commas.csv` and `missing_commas_11.txt` were regenerated at 110 ms (no proposals).
- Speakers 0 and 4 were reviewed with Whisper ASR earlier; 13 got two commas (13/0050, 13/1146). Other speakers
  not yet run through the detector. Candidates by displaced pauses: 6 Nicole, 3 Bella, 12 Michael (see the
  13 November evening section).
- Brian (4), 14 Nov, 110 ms / -90 dB, 019c, corpus file of 14 Nov 12:37: 225 pauses >= 80 ms, 33 listed in 29
  recordings, 27 lines proposed, no skipped lines. All spot-checked pauses are digital near-zero (-95 to -105 dB),
  but about half are stop closures (before k/t/p/b/d/tʃ: case, debt, taste, pasta, clean-up, bowl, keeper,
  Edinburgh, try, looks), 112-139 ms, so for Brian 110 ms does not separate closures from phrase pauses; the next
  sound does. Most `[word]` entries are closures inside the word, not missed gaps, so the one-word-earlier comma is
  wrong there. eSpeak drops spaced dashes (4/0829 "slowly - very slowly -" has no punctuation id), so the model sees
  no mark at those pauses. Awaiting the user's review.
  CORRECTION (user, 4/0085 has no 110 ms silence): the detector uses the per-frame RMS level. Brian's closures and
  pauses are not digital zero: about 1 sample in 6 is ±1-4 LSB (±2 = -84 dB peak), which averages to -93..-105 dB
  RMS but is not the flat line in Audacity (±1 LSB = -90.3 dB). Lewis is noise-gated, so RMS and peak agreed there.
  With "every sample within ±1 LSB" only 6 of the 33 stay >= 110 ms (0425, 0612, 1448, 0829 at 3.80, 0970, 1016);
  4/0085 reads 107 ms. Switching the detector to a peak criterion = a detection change, ask first.
- Aria (2), 14 Nov, 120 ms / -60 dB, 019c, corpus file of 14 Nov 15:54 (2217 lines, all with mels): 73 pauses,
  63 at punctuation, 10 inside a word (all covered by MAS at punctuation), 0 listed, `metadata.2_commas.csv`
  identical to the corpus file. Her pauses inside speech are NOT digital zero: a -55..-70 dB RMS floor, peaks
  20-80 LSB, so at -60 dB a comma pause breaks into short pieces. Cross-check (to /tmp, deleted): -50 dB also lists
  0; -45 dB lists 19, nearly all stop closures (act, talked, fact, verdict, architect). Nothing to fix for Aria.
  Cosmetic: the summary line says "pauses of 80 ms or more" but counts with `--min_pause_ms`.
- Isabella (13), 14 Nov, 200 ms / -75 dB, corpus file of 14 Nov 13:29 (1207 recordings, the other 6 lines are
  markers/empty): 346 pauses, 330 at punctuation, 16 inside a word (covered), 0 listed, `metadata.13_commas.csv`
  identical. Her pauses sit around -60..-80 dB (not digital zero), so -75 dB splits many of them; at -60 dB / 200 ms
  916 pauses, still 0 listed. Cross-checks (to /tmp, deleted): 150 ms at -75 / -60 dB: 0; 100 ms / -60 dB: 7;
  100 ms / -50 dB: 74, almost all Italian double-consonant closures (notte, detto, tutto, fatti, sette, attimo).
  Only space-only candidates below 200 ms: 13/0480 3.98 s "volta ⟨⟩ e li" ~120 ms, 13/0486 3.80 s "romane ⟨⟩ e
  alcuni" ~100 ms, 13/0532 5.84 s "zuppa ⟨⟩ e ne" ~100 ms. The two commas the user added earlier (13/0050 2.31 s
  ~128 ms, 13/1146 ~107 ms) are also well under 200 ms.
- Bella (3), 14 Nov, 200 ms / -75 dB, 019c, corpus file of 5 May (4298 lines, no skipped lines, 4.5 min):
  1353 pauses, 244 at punctuation, 1109 listed in 1026 recordings (310 `[word]`, 3 `[comma?]`), 1024 lines
  proposed; diff file verified identical except commas. Spot-checked 9: real silence (median -80..-86 dB RMS, peaks
  12-17 LSB, not digital zero, loud speech right after), not closures. Her unpunctuated pauses are as long as her
  comma pauses: lines with one comma, longest -75 dB silence median 203 ms; lines without inner punctuation median
  155 ms, 26% >= 200 ms. MAS already puts 800 of 805 space-only pauses on their gap (5 displaced: 3/1372, 4044,
  0853, 1669, 4204), so these pauses do not hurt her alignment. Spaced dashes get ", -" proposals (3/0794, 0804,
  0805). Awaiting the user's decision (wholesale commas would change a quarter of her lines).
  User's decision: add the commas anyway, so the model learns "comma = pause" and Bella's quirk pauses (her TTS
  engine's) do not leak into plain spaces at inference. Rerun at 140 ms / -75 dB (files overwritten): 3464 pauses,
  446 at punctuation, 3018 listed in 2240 recordings, 2237 lines proposed, 3/4129 unmatched (no comma). By length:
  140-160 ms 531, 160-180 726, 180-200 652, 200+ 1109; `[word]` 12-17% below 200 ms. 14 random 140-199 ms entries:
  all phrase-boundary pauses at the same level as the long ones (median -80..-84 dB, peaks 13-27 LSB), no closures;
  most common next word "is" (348), "and" (216). Odd ones: 3/1639 "ten, point five", 3/3756 "to make, decisions".
  13 displaced pauses (the 5 above plus 1078, 3738, 1174, 3965, 0855, 1120, 1922, 3509).
- The matcha CSVs (`data/corpus-24k/train.csv`, `validate.csv`) do NOT yet contain the user's latest Lewis commas
  (corpus file saved 14 Nov 12:00; CSVs from 13 Nov 17:27). Remind him to bring the corpus over and rerun
  `validate_corpus_ipa` before training.

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
- The alignment cache `/tmp/alignments_789_train.pkl` was pickled from `__main__`; to load it elsewhere set
  `__main__.AlignedUtterance` to the class from `measure_alignment_by_symbol_class` first. /tmp may be wiped.
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
