import argparse
from pathlib import Path
from pymcd.mcd import Calculate_MCD

"""
Script mcd_generate.sh generates speech using phrases from the validation set.
Script mcd_validate.sh runs this python file to compare them against the originals

Here are the results:
======================================================================
utterance_001_speaker_000                MCD:   3.86 dB
utterance_001_speaker_001                MCD:   2.49 dB
utterance_001_speaker_002                MCD:   2.65 dB
utterance_001_speaker_003                MCD:   2.06 dB
utterance_001_speaker_004                MCD:   8.39 dB
utterance_001_speaker_005                MCD:   2.89 dB
utterance_001_speaker_006                MCD:   2.81 dB
utterance_001_speaker_007                MCD:   4.55 dB
utterance_001_speaker_008                MCD:   3.65 dB
utterance_001_speaker_009                MCD:   2.61 dB
======================================================================
Average                                  MCD:   3.60 dB


For phrases found in the training set, the values were:
======================================================================
utterance_3823_speaker_000               MCD:   3.48 dB
utterance_3901_speaker_001               MCD:   2.19 dB
utterance_0700_speaker_002               MCD:   2.23 dB
utterance_0700_speaker_003               MCD:   1.66 dB
utterance_0700_speaker_004               MCD:   4.11 dB
utterance_0700_speaker_005               MCD:   2.61 dB
utterance_0700_speaker_006               MCD:   2.66 dB
======================================================================
Average                                  MCD:   2.70 dB
"""

def main():
    parser = argparse.ArgumentParser(description="Compare generated mp3 files to original wav files using MCD")
    parser.add_argument("folder", type=str, help="Folder containing .wav and .mp3 files")
    args = parser.parse_args()
    
    folder = Path(args.folder)
    
    wav_files = sorted(folder.glob("utterance_*.wav"))
    if not wav_files:
        print(f"Error: No utterance_*.wav files found in {folder}")
        return
    
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    
    print(f"\nComparing {len(wav_files)} mp3/wav pairs\n")
    print("=" * 70)
    
    results = []
    for wav_file in wav_files:
        mp3_file = folder / f"{wav_file.stem}.mp3"
        
        if not mp3_file.exists():
            print(f"{wav_file.name:40s} -> SKIPPED (no matching mp3)")
            continue
        
        mcd_value = mcd_toolbox.calculate_mcd(str(mp3_file), str(wav_file))
        results.append((wav_file.stem, mcd_value))
        print(f"{wav_file.stem:40s} MCD: {mcd_value:6.2f} dB")
    
    print("=" * 70)
    if results:
        avg_mcd = sum(r[1] for r in results) / len(results)
        print(f"{'Average':40s} MCD: {avg_mcd:6.2f} dB\n")
    else:
        print("\nNo matching pairs found\n")
    
    if results:
        print("Interpretation:")
        print("  < 5.0 dB  = Excellent (very close to original)")
        print("  5-7 dB    = Good (noticeable but acceptable)")
        print("  7-10 dB   = Fair (clear acoustic differences)")
        print("  > 10 dB   = Poor (significant timbre/quality issues)")
        print("\nLower MCD = better acoustic match to ground truth")


if __name__ == "__main__":
    main()
