""" from https://github.com/jik876/hifi-gan """

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

from .xutils import get_padding, init_weights

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.h = h
        self.convs1 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )
        self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.h = h
        self.convs = nn.ModuleList(
            [
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
            ]
        )
        self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Generator(torch.nn.Module):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(80, h.upsample_initial_channel, 7, 1, padding=3))
        resblock = ResBlock1 if h.resblock == "1" else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        h.upsample_initial_channel // (2**i),
                        h.upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, 1, 7, 1, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if use_spectral_norm is False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList([AvgPool1d(4, 2, padding=2), AvgPool1d(4, 2, padding=2)])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses



if __name__ == "__main__":
    # CLI entrypoint to test HiFiGAN vocoder quality on a given WAV or a mel (.npy), fully YAML-driven.
    import argparse
    import os

    import numpy as np
    import torch
    import torch.backends.cudnn as cudnn
    from scipy.io.wavfile import write as wav_write

    try:
        import librosa
        from librosa.util import normalize as librosa_normalize
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "librosa is required for this test script. Please install it (pip install librosa)."
        ) from e

    from types import SimpleNamespace
    from matcha.hifigan.denoiser import Denoiser
    from matcha.utils.audio import MAX_WAV_VALUE
    from matcha.mel.extractors import get_mel_extractor
    from matcha.cli import assert_required_models_available, VOCODER_URLS

    parser = argparse.ArgumentParser(description="Test HiFiGAN vocoder: use corpus YAML, accept wav or mel.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wav", type=str, help="Input WAV file to extract mel from (corpus config used for all params)")
    group.add_argument("--mel", type=str, help="Input precomputed mel numpy file (.npy) from precompute_corpus (corpus config used for all params)")
    parser.add_argument("--vocoder-id", type=str, required=True, choices=list(VOCODER_URLS.keys()), help="Pretrained vocoder ID (will be downloaded to user data dir if needed)")
    parser.add_argument("--data-config", "-c", required=True, type=str, help="Path to corpus data YAML (as used by precompute_corpus). All params (mel_mean, mel_std, sample_rate, n_feats, n_fft, hop_length, win_length, f_min, f_max) ARE TAKEN FROM THIS FILE.")
    default_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vocoder-test.wav"))
    parser.add_argument("--out", type=str, default=default_out, help="Output WAV path (default: project_root/vocoder-test.wav)")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="Device for inference (default: auto)")
    parser.add_argument("--remove-weight-norm", action="store_true", help="Remove weight norm from generator for inference speed")
    parser.add_argument("--denoise", action="store_true", help="Apply HiFi-GAN denoiser (bias removal) after generation")
    parser.add_argument("--denoise-strength", type=float, default=0.0005, help="Denoiser strength (default: 5e-4)")

    args = parser.parse_args()

    # Seed & cuDNN flags for reproducibility
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    # Load and display corpus config
    import yaml
    from pathlib import Path
    cfg_path = Path(args.data_config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[hifigan-test] Loaded corpus data config: {cfg_path}")

    # Extract all params ONLY from YAML
    try:
        mel_mean = float(config["data_statistics"]["mel_mean"])
        mel_std = float(config["data_statistics"]["mel_std"])
        sample_rate = int(config["sample_rate"])
        n_feats = int(config["n_feats"])
        n_fft = int(config["n_fft"])
        hop_length = int(config["hop_length"])
        win_length = int(config["win_length"])
        f_min = float(config["f_min"])
        f_max = float(config["f_max"])
    except Exception as e:
        raise RuntimeError(f"Config file missing required fields for HiFiGAN test. Required: sample_rate, n_feats, n_fft, hop_length, win_length, f_min, f_max, data_statistics.mel_mean, data_statistics.mel_std\n{e}")

    print(f"[hifigan-test] Effective params: sr={sample_rate}, n_feats={n_feats}, mel_mean={mel_mean}, mel_std={mel_std}, n_fft={n_fft}, hop_length={hop_length}, win_length={win_length}, f_min={f_min}, f_max={f_max}")

    # Device selection
    if args.device == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hifigan-test] device={device}")

    # Load vocoder weights from user data dir if needed
    _ns = SimpleNamespace(checkpoint_path="", model="matcha_ljspeech", vocoder=args.vocoder_id)
    paths = assert_required_models_available(_ns)
    ckpt_path = str(paths["vocoder"])
    print(f"[hifigan-test] using vocoder_id '{args.vocoder_id}' -> {ckpt_path}")

    # Build generator
    from matcha.hifigan.config import v1 as _cfg_v1
    h = SimpleNamespace(**_cfg_v1)
    h.sampling_rate = sample_rate
    h.num_mels = n_feats
    h.n_fft = n_fft
    h.hop_size = hop_length
    h.win_size = win_length
    h.fmin = f_min
    h.fmax = f_max
    generator = Generator(h).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("generator", ckpt.get("state_dict", ckpt))
    missing, unexpected = generator.load_state_dict(state, strict=False)
    if missing:
        print(f"[hifigan-test] Missing keys when loading state_dict: {missing}")
    if unexpected:
        print(f"[hifigan-test] Unexpected keys when loading state_dict: {unexpected}")

    generator.eval()
    if args.remove_weight_norm:
        print("[hifigan-test] Removing weight norm for inference")
        generator.remove_weight_norm()

    # Load or extract mel, apply normalization and denormalization logic as for vocos24k
    if args.mel is not None:
        if not os.path.exists(args.mel):
            raise FileNotFoundError(f"Specified mel file does not exist: {args.mel}")
        print(f"[hifigan-test] Loading mel from npy: {args.mel}")
        mel = np.load(args.mel)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]
        # Denormalize, as wrapper.py
        print(f"[hifigan-test] Denormalizing mel using mean={mel_mean}, std={mel_std}")
        mel = mel * mel_std + mel_mean
        mel = torch.from_numpy(mel).float().to(device)
        print(f"[hifigan-test] Loaded mel shape: {tuple(mel.shape)}")
    elif args.wav is not None:
        y, _ = librosa.load(args.wav, sr=sample_rate)
        y = librosa_normalize(y) * 0.95
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(device)
        mel_extractor = get_mel_extractor(
            "hifigan",
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_feats,
            f_min=f_min,
            f_max=f_max,
        )
        mel = mel_extractor(y_t).cpu().numpy()
        print(f"[hifigan-test] Extracted mel (factory: hifigan) shape: {tuple(mel.shape)}")
        # Normalize and then denormalize (testing full train/infer symmetry)
        print(f"[hifigan-test] Normalizing mel with mean={mel_mean}, std={mel_std}")
        mel = (mel - mel_mean) / mel_std
        print(f"[hifigan-test] Denormalizing mel using mean={mel_mean}, std={mel_std} (restores to vocoder expected domain)")
        mel = mel * mel_std + mel_mean
        mel = torch.from_numpy(mel).float().to(device)
        print(f"[hifigan-test] Final (denormalized) mel shape: {tuple(mel.shape)}")
    else:
        raise ValueError("You must provide either --wav or --mel")

    # Synthesize waveform
    with torch.no_grad():
        y_hat = generator(mel).squeeze(0).squeeze(0)  # [T]

    if args.denoise:
        print(f"[hifigan-test] Applying denoiser (strength={args.denoise_strength})")
        denoiser = Denoiser(generator)
        y_hat = denoiser(y_hat.unsqueeze(0), strength=args.denoise_strength).squeeze(0)

    # Save output wav (16-bit PCM)
    y_np = y_hat.detach().float().cpu().numpy()
    y_np = np.clip(y_np, -1.0, 1.0)
    wav_int16 = (y_np * MAX_WAV_VALUE).astype(np.int16)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wav_write(args.out, sample_rate, wav_int16)
    print(f"[hifigan-test] Reconstituted file saved to: {args.out}")
