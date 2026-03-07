import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleEncoder(nn.Module):
    """
    Encodes one or more reference mel spectrograms into the three speaker embedding vectors
    used by MatchaTTS (encoder, duration predictor, decoder).

    Architecture:
        Conv stack  →  GRU  →  attention pooling over N references  →  3 projection heads
    """

    def __init__(self, n_feats: int, spk_emb_dim_enc: int, spk_emb_dim_dur: int, spk_emb_dim_dec: int, n_channels: int = 128):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv1d(n_feats, n_channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(n_channels, n_channels, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(n_channels, n_channels, batch_first=True, bidirectional=True)
        gru_out_dim = n_channels * 2

        self.attention = nn.Linear(gru_out_dim, 1)
        self.clip_attention = nn.Linear(gru_out_dim, 1)

        self.head_enc = nn.Linear(gru_out_dim, spk_emb_dim_enc)
        self.head_dur = nn.Linear(gru_out_dim, spk_emb_dim_dur)
        self.head_dec = nn.Linear(gru_out_dim, spk_emb_dim_dec)

    def forward(self, mels: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode a batch of reference mels and return the three speaker embeddings.

        Args:
            mels:    (N, n_feats, T_max) — padded batch of reference clips
            lengths: (N,)               — actual length of each clip
        Returns:
            (enc_emb, dur_emb, dec_emb), each shape (1, spk_emb_dim_*)
        """
        N, _, T = mels.shape
        x = self.convs(mels)                         # (N, hidden, T)
        x = x.transpose(1, 2)                        # (N, T, hidden)
        x, _ = self.gru(x)                           # (N, T, gru_out_dim)

        # Mask padding before frame-level attention
        time_idx = torch.arange(T, device=mels.device).unsqueeze(0)  # (1, T)
        pad_mask = time_idx >= lengths.unsqueeze(1)                   # (N, T)
        scores = self.attention(x).squeeze(-1)                        # (N, T)
        scores = scores.masked_fill(pad_mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)              # (N, T, 1)
        clip_vecs = (x * weights).sum(dim=1)                          # (N, gru_out_dim)

        # Clip-level attention pool → single speaker vector
        clip_vecs = clip_vecs.unsqueeze(0)                            # (1, N, gru_out_dim)
        clip_scores = self.clip_attention(clip_vecs)                  # (1, N, 1)
        clip_weights = F.softmax(clip_scores, dim=1)                  # (1, N, 1)
        z = (clip_vecs * clip_weights).sum(dim=1)                     # (1, gru_out_dim)

        return self.head_enc(z), self.head_dur(z), self.head_dec(z)
