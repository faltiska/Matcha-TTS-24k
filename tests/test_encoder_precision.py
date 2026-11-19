from types import SimpleNamespace

import pytest
import torch

from matcha.models.matcha_tts import MatchaTTS


class ConvEncoder(torch.nn.Module):
    """Stands in for the TextEncoder: one convolution, which bf16 autocast runs in bf16."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(4, 4, 3, padding=1)

    def forward(self, x, x_lengths, speaker_embedding_enc, speaker_embedding_dur):
        return self.conv(x), self.conv(x), None


@pytest.mark.parametrize(("encoder_fp32", "expected_dtype"), [(False, torch.bfloat16), (True, torch.float32)])
def test_encoder_runs_in_fp32_only_when_asked(encoder_fp32, expected_dtype):
    model = SimpleNamespace(encoder=ConvEncoder(), hparams=SimpleNamespace(encoder_fp32=encoder_fp32))
    x = torch.randn(2, 4, 7)
    speaker_embedding = torch.randn(2, 8)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        mu_x, logw, _ = MatchaTTS.run_encoder(model, x, None, speaker_embedding, speaker_embedding)

    assert mu_x.dtype == expected_dtype
    assert logw.dtype == expected_dtype
