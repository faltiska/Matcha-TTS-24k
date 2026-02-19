import argparse
import datetime as dt
import os
import warnings
from pathlib import Path
# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

import soundfile as sf
import torch

from matcha.inference import load_matcha, load_vocoder, synthesise, convert_to_mp3, convert_to_opus_ogg, SAMPLE_RATE, ODE_SOLVER

VOCODERS = { "vocos", "bigvgan" }

def save_to_folder(filename: str, waveform: dict, folder: str):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    wav_path = folder / f"{filename}.wav"
    sf.write(wav_path, waveform, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return wav_path


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.steps > 0, "Number of ODE steps must be greater than 0"

    if args.checkpoint_path is None:
        # When using pretrained models
        if args.model in SINGLESPEAKER_MODEL:
            args = validate_args_for_single_speaker_model(args)

        if args.model in MULTISPEAKER_MODEL:
            args = validate_args_for_multispeaker_model(args)
    else:
        # When using a custom model
        if args.vocoder is None:
            args.vocoder = "vocos"
            warn_ = f"[-] Using custom model checkpoint, but no vocoder specified, defaulting to {args.vocoder}."
            warnings.warn(warn_, UserWarning)
        if args.speaking_rate is None:
            args.speaking_rate = 1.0
        if args.spk is not None:
            args.spk = [int(s.strip()) for s in args.spk.split(",")]
        else:
            args.spk = [None]

    assert args.speaking_rate > 0, "Speaking rate must be greater than 0"

    return args


def validate_args_for_multispeaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != MULTISPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {MULTISPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = MULTISPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = MULTISPEAKER_MODEL[args.model]["speaking_rate"]

    spk_range = MULTISPEAKER_MODEL[args.model]["spk_range"]
    if args.spk is not None:
        spk_list = [int(s.strip()) for s in args.spk.split(",")]
        for spk_id in spk_list:
            assert (
                spk_id >= spk_range[0] and spk_id <= spk_range[-1]
            ), f"Speaker ID {spk_id} must be between {spk_range} for this model."
        args.spk = spk_list
    else:
        available_spk_id = MULTISPEAKER_MODEL[args.model]["spk"]
        warn_ = f"[!] Speaker ID not provided! Using speaker ID {available_spk_id}"
        warnings.warn(warn_, UserWarning)
        args.spk = [available_spk_id]

    return args


def validate_args_for_single_speaker_model(args):
    if args.vocoder is not None:
        if args.vocoder != SINGLESPEAKER_MODEL[args.model]["vocoder"]:
            warn_ = f"[-] Using {args.model} model! I would suggest passing --vocoder {SINGLESPEAKER_MODEL[args.model]['vocoder']}"
            warnings.warn(warn_, UserWarning)
    else:
        args.vocoder = SINGLESPEAKER_MODEL[args.model]["vocoder"]

    if args.speaking_rate is None:
        args.speaking_rate = SINGLESPEAKER_MODEL[args.model]["speaking_rate"]

    if args.spk is not None:
        warn_ = f"[-] Ignoring speaker id {args.spk} for {args.model}"
        warnings.warn(warn_, UserWarning)
    args.spk = [SINGLESPEAKER_MODEL[args.model]["spk"]]

    return args


@torch.inference_mode()
def cli():
    parser = argparse.ArgumentParser(
        description=" 🍵 Matcha-TTS: A fast TTS architecture with conditional flow matching"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the custom model checkpoint",
    )

    parser.add_argument(
        "--vocoder",
        type=str,
        default=None,
        help="Vocoder to use (default: will use the one suggested with the pretrained model))",
        choices=VOCODERS,
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., en-us, en-gb, ro)")
    parser.add_argument("--spk", type=str, default=None, help="Speaker ID or comma-separated list (e.g., 0 or 0,1,2)")
    parser.add_argument(
        "--solver",
        type=str,
        default=ODE_SOLVER,
        help="ODE solver to use (default: midpoint)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=20, help="Number of ODE steps  (default: 20)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for inference (default: use GPU if available)")
    parser.add_argument(
        "--denoiser_strength",
        type=float,
        default=0.00025,
        help="Strength of the vocoder bias denoiser (default: 0.00025)",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default=os.getcwd(),
        help="Output folder to save results (default: current dir)",
    )

    args = parser.parse_args()

    args = validate_args(args)
    device = get_device(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        args.model = "custom_model"

    model = load_matcha(args.model, args.checkpoint_path, device)
    model.decoder.solver = args.solver
    
    vocoder = load_vocoder(args.vocoder, device)

    print_config(args, model)

    spk_list = args.spk if args.spk[0] is not None else [None]
    for spk_id in spk_list:
        speak(args, model, vocoder, args.text, spk_id)


def speak(args, model, vocoder, text, spk_id=0):
    base_name = f"speaker_{spk_id:03d}"
    print("".join(["="] * 100))

    start_t = dt.datetime.now()
    waveform, rtf = synthesise(model, vocoder, text.strip(), args.language, spk_id or 0, None, args.steps, args.speaking_rate)
    t = (dt.datetime.now() - start_t).total_seconds()
    rtf_w = t * SAMPLE_RATE / waveform.shape[-1]
    print(f"[🍵] Inference time: {t:.2f}s, RTF: {rtf_w:.2f}")

    save_to_folder(base_name, waveform.cpu().numpy(), args.output_folder)
    Path(args.output_folder, f"{base_name}.mp3").write_bytes(convert_to_mp3(waveform))
    Path(args.output_folder, f"{base_name}.ogg").write_bytes(convert_to_opus_ogg(waveform))

    print("".join(["="] * 100))
    print(f"[🍵] RTF: {rtf:.4f}, RTF+vocoder: {rtf_w:.4f}")


def print_config(args, model):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- ODE steps: {args.steps}")
    print(f"\t- ODE Solver: {args.solver}")
    print(f"\t- Speaker: {args.spk}")

def get_device(args):
    if torch.cuda.is_available() and not args.cpu:
        print("[+] GPU Available! Using GPU")
        device = torch.device("cuda")
    else:
        print("[-] GPU not available or forced CPU run! Using CPU")
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    cli()
