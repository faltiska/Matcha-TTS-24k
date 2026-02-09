import argparse
import datetime as dt
import io
import os
import subprocess
import time
import warnings
from pathlib import Path
import numpy as np

# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

import soundfile as sf
import torch

from matcha.inference import load_matcha, load_vocoder, process_text, to_waveform
from matcha.utils.utils import assert_model_downloaded, get_user_data_dir

MATCHA_URLS = {
    "matcha_ljspeech": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_ljspeech.ckpt",
    "matcha_vctk": "https://github.com/shivammehta25/Matcha-TTS-checkpoints/releases/download/v1.0/matcha_vctk.ckpt",
}

VOCODER_URLS = {
    "vocos": "https://huggingface.co/charactr/vocos-mel-24khz",
}

MULTISPEAKER_MODEL = {
    "matcha_vctk": {"vocoder": "vocos", "speaking_rate": 1.0, "spk": 0, "spk_range": (0, 107)}
}

SINGLESPEAKER_MODEL = {"matcha_ljspeech": {"vocoder": "vocos", "speaking_rate": 1.0, "spk": None}}


def get_texts(args):
    if args.text:
        texts = [args.text]
    else:
        with open(args.file, encoding="utf-8") as f:
            texts = f.readlines()
    return texts


def assert_required_models_available(args):
    save_dir = get_user_data_dir()
    if not hasattr(args, "checkpoint_path") and args.checkpoint_path is None:
        model_path = args.checkpoint_path
    else:
        model_path = save_dir / f"{args.model}.ckpt"
        assert_model_downloaded(model_path, MATCHA_URLS[args.model])

    vocoder_path = save_dir / f"{args.vocoder}"
    assert_model_downloaded(vocoder_path, VOCODER_URLS[args.vocoder])
    return {"matcha": model_path, "vocoder": vocoder_path}


def save_to_folder(filename: str, waveform: dict, folder: str, sample_rate: int = 22050):
    folder = Path(folder)
    folder.mkdir(exist_ok=True, parents=True)
    
    # Convert to MP3
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    
    process = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-ab", "192k", "pipe:1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    mp3_data, _ = process.communicate(input=wav_buffer.read())
    
    mp3_path = folder / f"{filename}.mp3"
    mp3_path.write_bytes(mp3_data)
    return mp3_path


def validate_args(args):
    assert (
        args.text or args.file
    ), "Either text or file must be provided Matcha-T(ea)TTS need sometext to whisk the waveforms."
    assert args.temperature >= 0, "Sampling temperature cannot be negative"
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
        "--model",
        type=str,
        default="matcha_ljspeech",
        help="Model to use",
        choices=MATCHA_URLS.keys(),
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
        choices=VOCODER_URLS.keys(),
    )
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize")
    parser.add_argument("--file", type=str, default=None, help="Text file to synthesize")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., en-us, en-gb, ro)")
    parser.add_argument("--spk", type=str, default=None, help="Speaker ID or comma-separated list (e.g., 0 or 0,1,2)")
    parser.add_argument(
        "--solver",
        type=str,
        default="midpoint",
        help="ODE solver to use (default: midpoint)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Variance of the x0 noise (default: 0.8)",
    )
    parser.add_argument(
        "--speaking_rate",
        type=float,
        default=None,
        help="change the speaking rate, a higher value means slower speaking rate (default: 1.0)",
    )
    parser.add_argument("--steps", type=int, default=15, help="Number of ODE steps  (default: 20)")
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
    paths = assert_required_models_available(args)

    if args.checkpoint_path is not None:
        print(f"[🍵] Loading custom model from {args.checkpoint_path}")
        paths["matcha"] = args.checkpoint_path
        args.model = "custom_model"

    model = load_matcha(args.model, paths["matcha"], device)
    model.decoder.solver = args.solver
    
    # Set audio params if not present (for old checkpoints)
    if not hasattr(model, "sample_rate"):
        model.sample_rate = 24000
    if not hasattr(model, "hop_length"):
        model.hop_length = 256
    
    vocoder, denoiser = load_vocoder(args.vocoder, paths["vocoder"], device)

    print_config(args, model)

    texts = get_texts(args)

    spk_list = args.spk if args.spk[0] is not None else [None]
    for spk_id in spk_list:
        synthesis(args, device, model, vocoder, denoiser, texts, spk_id)


def synthesis(args, device, model, vocoder, denoiser, texts, spk_id):
    total_rtf = []
    total_rtf_w = []
    sample_rate = getattr(model, "sample_rate")
    for i, text in enumerate(texts):
        i = i + 1
        base_name = f"utterance_{i:03d}_speaker_{spk_id:03d}" if spk_id is not None else f"utterance_{i:03d}"

        print("".join(["="] * 100))
        text = text.strip()
        text_processed = process_text(i, text, args.language, device)

        print(f"[🍵] Whisking Matcha-T(ea)TS for: {i}")
        start_t = dt.datetime.now()
        output = model.synthesise(
            text_processed["x"],
            text_processed["x_lengths"],
            n_timesteps=args.steps,
            temperature=args.temperature,
            spks=spk_id if spk_id is not None else 0,
            length_scale=args.speaking_rate,
        )
        waveform = to_waveform(output["mel"], vocoder, denoiser, args.denoiser_strength)
        # RTF with vocoder
        t = (dt.datetime.now() - start_t).total_seconds()
        rtf_w = t * sample_rate / (waveform.shape[-1])
        print(f"[🍵-{i}] Inference time: {t:.2f}s, RTF: {rtf_w:.2f}")
        total_rtf.append(output["rtf"])
        total_rtf_w.append(rtf_w)

        location = save_to_folder(base_name, waveform.cpu().numpy(), args.output_folder, sample_rate)
        print(f"[+] Waveform saved: {location}")

    print("".join(["="] * 100))
    print(f"[🍵] Average Matcha-TTS RTF: {np.mean(total_rtf):.4f} ± {np.std(total_rtf)}")
    print(f"[🍵] Average Matcha-TTS + VOCODER RTF: {np.mean(total_rtf_w):.4f} ± {np.std(total_rtf_w)}")
    print("[🍵] Enjoy the freshly whisked 🍵 Matcha-TTS!")


def print_config(args, model):
    print("[!] Configurations: ")
    print(f"\t- Model: {args.model}")
    print(f"\t- Vocoder: {args.vocoder}")
    print(f"\t- Temperature: {args.temperature}")
    print(f"\t- Speaking rate: {args.speaking_rate}")
    print(f"\t- Number of ODE steps: {args.steps}")
    print(f"\t- ODE Solver: {args.solver}")
    print(f"\t- Speaker: {args.spk}")
    print(f"\t- Sample rate: {getattr(model, 'sample_rate')}")
    print(f"\t- Hop length: {getattr(model, 'hop_length')}")

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
