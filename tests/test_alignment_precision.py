import torch

import matcha.models.matcha_tts as matcha_tts
from matcha.models.matcha_tts import MatchaTTS


def test_alignment_score_is_exact_under_bf16_autocast(monkeypatch):
    """
    Training runs under bf16 autocast. The alignment score must still be computed in fp32: the scores are in the
    hundreds, where bf16 can only represent steps of 1 or 2, while the choice between neighbouring symbols depends
    on differences of a fraction of a point.
    """
    torch.manual_seed(0)
    # The Encoder output arrives already rounded to bf16, as in training.
    mu_x = (torch.randn(1, 100, 5) * 3).bfloat16()
    y_fine = torch.randn(1, 100, 40) * 3
    scores_seen_by_mas = []

    def record_scores(scores, mask, dtype):
        scores_seen_by_mas.append(scores.clone())
        return torch.zeros_like(scores)

    monkeypatch.setattr(matcha_tts, "maximum_path_gpu", record_scores)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        MatchaTTS.find_alignment(None, torch.ones(1, 1, 5, 40), mu_x, y_fine)

    mu, y = mu_x.double(), y_fine.double()
    expected = (
        -0.5 * (y ** 2).sum(1, keepdim=True)
        + mu.transpose(1, 2) @ y
        - 0.5 * (mu ** 2).sum(1, keepdim=True).transpose(1, 2)
    )
    scores = scores_seen_by_mas[0]
    assert scores.dtype == torch.float32
    # A bf16 matrix multiplication is off by about 1 here, far more than this tolerance.
    torch.testing.assert_close(scores.double(), expected, atol=1e-2, rtol=0)
