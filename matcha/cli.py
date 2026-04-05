import argparse
import os
import time
import warnings
from pathlib import Path
# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

import soundfile as sf
import torch

from matcha.inference import load_matcha, load_vocoder, pipeline, convert_to_mp3, SAMPLE_RATE, HIGH_RES_HOP_LENGTH, ODE_SOLVER, VOICES

VOCODERS = { "vocos" }


def save_to_folder(filename: str, waveform: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    wav_path = folder / f"{filename}.wav"
    sf.write(wav_path, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return wav_path


def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: a modern non-autoregressive TTS."
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        required=True,
        help="Path to the custom model checkpoint",
    )
    parser.add_argument(
        "--vocoder",
        type=str,
        default="vocos",
        help="Vocoder to use (default: Vocos)",
        choices=VOCODERS,
    )
    parser.add_argument("--text", 
        type=str, 
        default=None,
        required=True,
        help="Text to synthesize"
    )
    parser.add_argument("--spk", 
        type=str, 
        default=None,
        required=True,
        help="Speaker ID or comma-separated list of IDs (e.g., 0 or 0,1,2)"
    )
    parser.add_argument(
        "--solver",
        type=str,
        default=ODE_SOLVER,
        help="ODE solver to use (default: midpoint)",
    )
    parser.add_argument(
        "--length_scale",
        type=float,
        default=None,
        help="How much longer each phoneme duration should be compared to speaker's default.",
    )
    parser.add_argument("--steps", 
        type=int, 
        default=20, 
        help="Number of ODE steps  (default: 20)")
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Save encoder mel wav too, and a phoneme durations file.",
    )

    args = parser.parse_args()

    args.spk = [int(s.strip()) for s in args.spk.split(",")]
    
    print(f"[🍵] Loading custom model from {args.checkpoint_path}")
    args.model = "custom_model"

    model = load_matcha(args.model, args.checkpoint_path)
    model.decoder.solver = args.solver

    vocoder = load_vocoder(args.vocoder)

    print_config(args)

    spk_list = args.spk if args.spk[0] is not None else [None]
    for spk_id in spk_list:
        speak(args, model, vocoder, args.text, spk_id)


def speak(args, model, vocoder, text, speaker=0):
    base_name = f"speaker_{speaker:03d}"
    print("".join(["="] * 100))

    t = time.perf_counter()
    voice = next((v for v in VOICES if v["id"] == str(speaker)), VOICES[0])
    language = voice["lang"]
    scale_correction = voice["scale_correction"]
    if args.length_scale is not None:
        length_scale = args.length_scale
    else:
        length_scale = 1.0

    if args.debug:
        decoder_wav, encoder_wav, phoneme_dur_pairs = pipeline(
            model, vocoder, text.strip(), language, speaker or 0, None, args.steps, scale_correction, length_scale, debug=True
        )
        elapsed = time.perf_counter() - t
        audio_duration = decoder_wav.shape[-1] / SAMPLE_RATE
        print(f"[🍵] Total time: {elapsed:.2f}s | RTF: {elapsed / audio_duration:.4f}")

        save_to_folder(base_name, decoder_wav.cpu().numpy(), args.output_folder)
        save_to_folder(f"{base_name}_encoder", encoder_wav.cpu().numpy(), args.output_folder)

        dur_path = Path(args.output_folder) / f"{base_name}_durations.txt"
        # Columns: phoneme | raw duration (frames) | enforced duration (frames) | start time (seconds)
        # Raw duration is the duration predictor output before the monotonic enforcement in synthesise().
        frame_seconds = HIGH_RES_HOP_LENGTH / SAMPLE_RATE
        lines = []
        cumulative_frames = 0
        for symbol, raw, dur in phoneme_dur_pairs:
            start_seconds = cumulative_frames * frame_seconds
            lines.append(f"{symbol}\t{raw:.2f}\t{int(dur)}\t{start_seconds:.3f}")
            cumulative_frames += int(dur)
        dur_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[🍵] Encoder wav and durations saved to {args.output_folder}")
    else:
        waveform = pipeline(model, vocoder, text.strip(), language, speaker or 0, None, args.steps, scale_correction, length_scale)
        elapsed = time.perf_counter() - t
        audio_duration = waveform.shape[-1] / SAMPLE_RATE
        print(f"[🍵] Total time: {elapsed:.2f}s | RTF: {elapsed / audio_duration:.4f}")

        save_to_folder(base_name, waveform.cpu().numpy(), args.output_folder)
        Path(args.output_folder, f"{base_name}.mp3").write_bytes(convert_to_mp3(waveform))

    print("".join(["="] * 100))


def print_config(args):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Length scale: {args.length_scale}")
    print(f"\t- ODE steps: {args.steps}")
    print(f"\t- ODE Solver: {args.solver}")
    print(f"\t- Speaker: {args.spk}")


if __name__ == "__main__":
    cli()
