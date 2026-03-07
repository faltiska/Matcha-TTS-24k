import torch
import torch.nn.functional as F
from lightning import LightningModule


class EncoderLightning(LightningModule):
    """
    LightningModule that trains StyleEncoder to predict the three speaker
    embedding vectors of a frozen MatchaTTS model from reference mel spectrograms.
    """

    def __init__(self, matcha_ckpt: str, n_channels: int = 128, optimizer=None):
        super().__init__()
        self.save_hyperparameters(logger=False)

        ckpt = torch.load(matcha_ckpt, map_location="cpu", weights_only=False)
        hparams = ckpt["hyper_parameters"]
        n_feats        = hparams["n_feats"]
        spk_emb_dim_enc = hparams.get("spk_emb_dim_enc") or hparams["spk_emb_dim"]
        spk_emb_dim_dur = hparams.get("spk_emb_dim_dur") or hparams["spk_emb_dim"]
        spk_emb_dim_dec = hparams.get("spk_emb_dim_dec") or hparams["spk_emb_dim"]

        from matcha.models.style_encoder import StyleEncoder
        self.encoder = StyleEncoder(
            n_feats, spk_emb_dim_enc, spk_emb_dim_dur, spk_emb_dim_dec, n_channels=n_channels
        )

        sd = ckpt["state_dict"]
        enc_emb = sd["encoder_speaker_embeddings.weight"].detach()
        dur_emb = sd["duration_speaker_embeddings.weight"].detach()
        dec_emb = sd["decoder_speaker_embeddings.weight"].detach()
        self.register_buffer("enc_targets", enc_emb)
        self.register_buffer("dur_targets", dur_emb)
        self.register_buffer("dec_targets", dec_emb)

    def configure_optimizers(self):
        return self.hparams.optimizer(params=self.parameters())

    def training_step(self, batch, batch_idx):
        spk_id, mels, lengths = batch

        pred_enc, pred_dur, pred_dec = self.encoder(mels, lengths)

        gt_enc = self.enc_targets[spk_id].unsqueeze(0)
        gt_dur = self.dur_targets[spk_id].unsqueeze(0)
        gt_dec = self.dec_targets[spk_id].unsqueeze(0)

        loss = (
            self._embedding_loss(pred_enc, gt_enc)
            + self._embedding_loss(pred_dur, gt_dur)
            + self._embedding_loss(pred_dec, gt_dec)
        )

        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def _embedding_loss(self, pred, target):
        mse = F.mse_loss(pred, target)
        cosine = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
        return mse + 0.1 * cosine
