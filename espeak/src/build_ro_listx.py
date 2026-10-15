"""
Generate espeak/src/ro/ro_listx from the RoLEX lexicon.

For every RoLEX word form we compare what the current eSpeak rules (ro_rules + ro_list +
ro_emoji in espeak/src/ro) produce with what RoLEX says, both rendered to IPA by eSpeak
itself through phonemizer, exactly as matcha does at training time. Only words that
really differ are written to ro_listx.

Why ro_listx and not ro_list / ro_extra: espeak-ng compiles ro_listx first, then ro_list,
then ro_emoji, then ro_extra; a later entry wins. So:
    ro_listx  generated here, never edit by hand (it is overwritten)
    ro_list   eSpeak's hand-written entries (flags like $u, $1 ...) still win over RoLEX
    ro_extra  your manual overrides, win over everything (edit, then run compile_ro.sh)

Comparison: both sides are normalised before comparing (secondary stress, palatalisation
mark ʲ, ɾ/r, ŋ/n, ɪ/j, ʊ/w, ç/h are ignored) so that only stress position and real
segmental differences count. The emitted entries already follow eSpeak's conventions
(see rolex_to_espeak.py).

Homographs (same spelling, different RoLEX pronunciations, e.g. cópii / copíi): if the rules
already match one of the readings, the word is left alone; if the readings disagree with
the rules and with each other, the word is skipped and listed in reports/ro_ambiguous.tsv
for a manual decision in ro_extra.

Writes:
    espeak/src/ro/ro_listx
    espeak/src/reports/ro_patterns.txt        summary + most frequent difference patterns
    espeak/src/reports/ro_ambiguous.tsv       homographs needing a manual decision
    espeak/src/reports/ro_list_conflicts.tsv  ro_list words where RoLEX disagrees
    espeak/src/reports/ro_rejected.tsv        RoLEX rows the converter refused, with reason

Run through build_ro_listx.sh, which also downloads RoLEX and compiles espeak/bin/ro.
"""

import argparse
import collections
import difflib
import logging
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rolex_to_espeak import ConversionError, convert_row, to_cedilla  # noqa: E402

from phonemizer.backend import EspeakBackend  # noqa: E402
from phonemizer.backend.espeak.wrapper import EspeakWrapper  # noqa: E402

SRC_FILES = ("ro_rules", "ro_list", "ro_emoji")
SYSTEM_DATA_FILES = ("phontab", "phonindex", "phondata", "phondata-manifest", "intonations")

CANON = str.maketrans({"ˌ": None, "ʲ": None, "ɾ": "r", "ŋ": "n", "ɪ": "j", "ʊ": "w", "ç": "h"})
IPA_VOWELS = set("aeiouəɨɔy")

quiet_logger = logging.getLogger("build_ro_listx.phonemizer")
quiet_logger.setLevel(logging.ERROR)


def canon(ipa: str) -> str:
    return ipa.translate(CANON)


def system_data_dir() -> Path:
    line = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True,
                          check=True).stdout
    return Path(re.search(r"Data at: (.*)$", line.strip()).group(1))


def list_keys(path: Path) -> set[str]:
    """Single-word keys of an eSpeak *_list style file."""
    keys = set()
    if not path.exists():
        return keys
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("//", 1)[0].strip()
        if not line or line[0] in "_(":
            continue
        keys.add(to_cedilla(line.split()[0].lower()))
    return keys


def build_data_dir(work: Path, name: str, src_dir: Path, extra_files: dict[str, str]) -> Path:
    """Compile SRC_FILES (+ extra_files) into a fresh eSpeak data dir and return it."""
    sources = work / name / "src"
    data = work / name / "data"
    (data / "lang" / "roa").mkdir(parents=True)
    sources.mkdir(parents=True)
    system = system_data_dir()
    for f in SYSTEM_DATA_FILES:
        shutil.copy(system / f, data / f)
    shutil.copy(system / "lang" / "roa" / "ro", data / "lang" / "roa" / "ro")
    for f in SRC_FILES:
        shutil.copy(src_dir / f, sources / f)
    for f, text in extra_files.items():
        (sources / f).write_text(text, encoding="utf-8")
    result = subprocess.run(["espeak-ng", f"--path={data}", "--compile=ro"], cwd=sources,
                            capture_output=True, text=True)
    log = result.stdout + result.stderr
    errors = [l for l in log.splitlines() if re.match(r"\s*\d+: ", l)]
    if result.returncode != 0 or errors or not (data / "ro_dict").exists():
        raise RuntimeError(f"espeak-ng compile failed for {name}:\n" + "\n".join(errors[:20] or [log]))
    return data


def phonemize(data_dir: Path, words: list[str]) -> list[str]:
    """Same backend settings as matcha/text/phonemizers.py."""
    EspeakWrapper.set_data_path(str(data_dir))
    try:
        backend = EspeakBackend(language="ro", preserve_punctuation=True, with_stress=True,
                                language_switch="remove-flags", logger=quiet_logger)
    finally:
        EspeakWrapper.set_data_path(None)
    return backend.phonemize(words, strip=True, njobs=1)


def load_rolex(path: Path):
    """key -> ordered unique eSpeak variants; plus rejected rows."""
    variants = collections.defaultdict(list)
    rejected = []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for line in fh:
            row = line.rstrip("\r\n").split("\t")
            if len(row) != 6:
                rejected.append(("bad column count", row))
                continue
            form, lemma, _msd, syllables, stress, transcription = row
            try:
                phonemes = convert_row(form, lemma, syllables, stress, transcription)
            except ConversionError as e:
                rejected.append((str(e), row))
                continue
            key = to_cedilla(form.lower())
            if phonemes not in variants[key]:
                variants[key].append(phonemes)
    return variants, rejected


def stress_from_end(ipa: str) -> int | None:
    """Number of vowel letters after the primary stress mark (rough syllable position)."""
    if "ˈ" not in ipa:
        return None
    tail = ipa.split("ˈ", 1)[1]
    return sum(c in IPA_VOWELS for c in tail)


def segment_patterns(base: str, cand: str):
    """Edit operations between stress-free canonical IPA strings, with 1 char of context."""
    a, b = base.replace("ˈ", ""), cand.replace("ˈ", "")
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        left = a[i1 - 1] if i1 > 0 else "#"
        right = a[i2] if i2 < len(a) else "#"
        yield f"{a[i1:i2] or '∅'} → {b[j1:j2] or '∅'}", f"{left}_{right}"


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rolex", type=Path, required=True, help="rolex.v1.txt")
    ap.add_argument("--rolex-commit", default="unknown", help="for the ro_listx header")
    ap.add_argument("--src-dir", type=Path, default=here / "ro")
    ap.add_argument("--report-dir", type=Path, default=here / "reports")
    args = ap.parse_args()

    espeak_version = subprocess.run(["espeak-ng", "--version"], capture_output=True, text=True,
                                    check=True).stdout.split(":")[1].split()[0]

    print("Converting RoLEX ...")
    variants, rejected = load_rolex(args.rolex)
    ro_list = list_keys(args.src_dir / "ro_list")
    manual = list_keys(args.src_dir / "ro_extra")
    words = sorted(k for k in variants if len(k) > 1)
    print(f"  {len(words)} word forms, {len(rejected)} rows rejected, "
          f"{len(ro_list)} ro_list keys, {len(manual)} ro_extra (manual) keys")

    with tempfile.TemporaryDirectory(prefix="ro_listx_") as tmp:
        work = Path(tmp)
        print("Phonemizing with current rules ...")
        base_dir = build_data_dir(work, "baseline", args.src_dir, {})
        baseline = dict(zip(words, phonemize(base_dir, words)))

        # One compile per homograph variant index; ro_extra so candidates also beat ro_list.
        candidates = collections.defaultdict(list)
        for k in range(max(len(v) for v in variants.values())):
            subset = [w for w in words if len(variants[w]) > k]
            print(f"Phonemizing RoLEX variant #{k + 1} ({len(subset)} words) ...")
            extra = "".join(f"{w}\t{variants[w][k]}\n" for w in subset)
            cand_dir = build_data_dir(work, f"cand{k}", args.src_dir, {"ro_extra": extra})
            for w, ipa in zip(subset, phonemize(cand_dir, subset)):
                candidates[w].append(ipa)

        status = {}
        chosen = {}  # word -> index of the variant written to ro_listx
        for w in words:
            cb = canon(baseline[w])
            cc = [canon(c) for c in candidates[w]]
            if w in manual:
                status[w] = "manual"
            elif cb in cc:
                status[w] = "same"
            elif len(set(cc)) > 1:
                status[w] = "list_conflict" if w in ro_list else "ambiguous"
            else:
                status[w] = "list_conflict" if w in ro_list else "diff"
                if status[w] == "diff":
                    chosen[w] = 0

        # Adding a stem (e.g. "abdic") can change inflected forms that are NOT listed, because
        # ro_rules strips suffixes (the _S1 ... rules) and looks the stem up in the dictionary.
        # So verify the whole lexicon and pin every word whose output changed, until stable.
        pinned = set()
        ambiguous_now = {}
        for round_no in range(1, 11):
            listx_text = render_listx(chosen, variants, baseline, candidates, args.rolex_commit,
                                      espeak_version)
            print(f"Verifying round {round_no}: {len(chosen)} entries ...")
            final_dir = build_data_dir(work, f"final{round_no}", args.src_dir, {"ro_listx": listx_text})
            final = dict(zip(words, phonemize(final_dir, words)))
            mismatches, newly_pinned = [], 0
            for w in words:
                expected = candidates[w][chosen[w]] if w in chosen else baseline[w]
                if status[w] == "manual" or final[w] == expected:
                    continue
                if status[w] == "same" and w not in ro_list and w not in pinned:
                    cb = canon(baseline[w])
                    chosen[w] = next(i for i, c in enumerate(candidates[w]) if canon(c) == cb)
                    pinned.add(w)
                    newly_pinned += 1
                elif status[w] in ("ambiguous", "list_conflict") or (status[w] == "same" and w in ro_list):
                    # Can't pin: no single right reading / ro_list (flag-only entry) wins. Reported.
                    ambiguous_now[w] = final[w]
                else:
                    mismatches.append((w, status[w], expected, final[w]))
            if mismatches:
                for m in mismatches[:20]:
                    print("  MISMATCH", *m)
                sys.exit(f"{len(mismatches)} words do not phonemize as expected with the new ro_listx")
            print(f"  pinned {newly_pinned} more words whose output changed through a listed stem")
            if newly_pinned == 0:
                break
        else:
            sys.exit("ro_listx did not converge")

    (args.src_dir / "ro_listx").write_text(listx_text, encoding="utf-8")
    write_reports(args.report_dir, words, variants, baseline, candidates, status, chosen, pinned,
                  ambiguous_now, rejected, args.rolex_commit, espeak_version)
    print(f"Wrote {args.src_dir / 'ro_listx'} ({len(chosen)} entries, {len(pinned)} of them pinned) "
          f"and reports in {args.report_dir}")


def render_listx(chosen, variants, baseline, candidates, rolex_commit, espeak_version) -> str:
    lines = [
        "// GENERATED by espeak/src/build_ro_listx.sh - do not edit, it gets overwritten.",
        "// Put manual fixes in ro_extra instead (it overrides this file and ro_list).",
        f"// Source: RoLEX https://github.com/adrianastan/rolex commit {rolex_commit}",
        "//   Lőrincz, Irimia, Stan, Barbu Mititelu, Natural Language Engineering 29(3), 2022.",
        f"// Words where eSpeak-ng {espeak_version} rules disagree with RoLEX, plus words pinned",
        "// because a listed stem would otherwise change them. Comment: rules IPA -> entry IPA.",
        "",
    ]
    for w in sorted(chosen):
        lines.append(f"{w}\t{variants[w][chosen[w]]}\t// {baseline[w]} -> {candidates[w][chosen[w]]}")
    return "\n".join(lines) + "\n"


def write_reports(report_dir, words, variants, baseline, candidates, status, chosen, pinned,
                  ambiguous_now, rejected, rolex_commit, espeak_version):
    report_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "ro_ambiguous.tsv", "w", encoding="utf-8") as fh:
        fh.write("word\trules_ipa\tnew_dict_ipa\trolex_variants_ipa\trolex_variants_espeak\n")
        for w in words:
            if status[w] == "ambiguous":
                fh.write(f"{w}\t{baseline[w]}\t{ambiguous_now.get(w, baseline[w])}\t"
                         f"{' | '.join(candidates[w])}\t{' | '.join(variants[w])}\n")

    with open(report_dir / "ro_list_conflicts.tsv", "w", encoding="utf-8") as fh:
        fh.write("word\tro_list_ipa\tnew_dict_ipa\trolex_variants_ipa\trolex_variants_espeak\n")
        for w in words:
            if status[w] == "list_conflict" or (status[w] == "same" and w in ambiguous_now):
                fh.write(f"{w}\t{baseline[w]}\t{ambiguous_now.get(w, baseline[w])}\t"
                         f"{' | '.join(candidates[w])}\t{' | '.join(variants[w])}\n")

    with open(report_dir / "ro_rejected.tsv", "w", encoding="utf-8") as fh:
        fh.write("reason\tform\tlemma\tmsd\tsyllables\tstress\ttranscription\n")
        for reason, row in rejected:
            fh.write(reason + "\t" + "\t".join(row) + "\n")

    # Pattern analysis over the entries written to ro_listx.
    stress_only, segmental = [], []
    for w in chosen:
        if w in pinned:
            continue
        b, c = canon(baseline[w]), canon(candidates[w][chosen[w]])
        (stress_only if b.replace("ˈ", "") == c.replace("ˈ", "") else segmental).append(w)

    seg_counts = collections.Counter()
    seg_ctx = collections.defaultdict(collections.Counter)
    seg_examples = collections.defaultdict(list)
    for w in segmental:
        seen = set()
        for pattern, ctx in segment_patterns(canon(baseline[w]), canon(candidates[w][chosen[w]])):
            if pattern in seen:
                continue
            seen.add(pattern)
            seg_counts[pattern] += 1
            seg_ctx[pattern][ctx] += 1
            if len(seg_examples[pattern]) < 4:
                seg_examples[pattern].append(f"{w} {baseline[w]}→{candidates[w][chosen[w]]}")

    shift_counts = collections.Counter()
    shift_suffixes = collections.defaultdict(collections.Counter)
    shift_examples = collections.defaultdict(list)
    for w in stress_only:
        b, c = canon(baseline[w]), canon(candidates[w][chosen[w]])
        key = f"{stress_from_end(b)} → {stress_from_end(c)}"
        shift_counts[key] += 1
        shift_suffixes[key][w[-3:]] += 1
        if len(shift_examples[key]) < 4:
            shift_examples[key].append(f"{w} {baseline[w]}→{candidates[w][chosen[w]]}")

    st = collections.Counter(status.values())
    reasons = collections.Counter(r for r, _ in rejected)
    out = [
        f"RoLEX ({rolex_commit}) vs eSpeak-ng {espeak_version} Romanian rules",
        "",
        f"Word forms compared:             {len(words)}",
        f"  rules already agree:           {st['same']}",
        f"  RoLEX differs, in ro_listx:    {st['diff']}",
        f"    stress position only:        {len(stress_only)}",
        f"    segmental (maybe + stress):  {len(segmental)}",
        f"  ambiguous homographs, skipped: {st['ambiguous']}  (ro_ambiguous.tsv)",
        f"  in ro_list, RoLEX disagrees:   {st['list_conflict']}  (ro_list_conflicts.tsv)",
        f"  manual overrides in ro_extra:  {st['manual']}",
        f"Pinned in ro_listx (agree, but a listed stem would change them): {len(pinned)}",
        f"Ambiguous / ro_list words changed through a listed stem:        {len(ambiguous_now)}",
        f"RoLEX rows rejected:             {len(rejected)}  (ro_rejected.tsv)",
    ]
    out += [f"  {n:7d}  {r}" for r, n in reasons.most_common()]
    out += ["", "Stress shifts (vowels after the stress mark: rules → RoLEX; 0 = final syllable)", ""]
    for key, n in shift_counts.most_common(15):
        suffixes = ", ".join(f"-{s}:{c}" for s, c in shift_suffixes[key].most_common(12))
        out.append(f"{n:7d}  {key}   top endings: {suffixes}")
        out += [f"           {e}" for e in shift_examples[key]]
    out += ["", "Segmental patterns (rules → RoLEX, canonical IPA; contexts as left_right, # = word edge)", ""]
    for pattern, n in seg_counts.most_common(80):
        ctx = ", ".join(f"{c}:{k}" for c, k in seg_ctx[pattern].most_common(6))
        out.append(f"{n:7d}  {pattern:14}  contexts {ctx}")
        out += [f"           {e}" for e in seg_examples[pattern]]
    (report_dir / "ro_patterns.txt").write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
