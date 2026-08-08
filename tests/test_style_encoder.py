import torch

from matcha.models.style_encoder import (
    StyleEncoder,
    StyleEncoderLightningModule,
    effective_embedding_error,
    masked_smooth_l1_loss,
    masked_stats_pool,
)


def _speaker_projection_that_ignores_its_second_input():
    """A stand-in for the main model's speaker projection, deliberately blind to one input direction."""
    projection = torch.nn.Linear(2, 2)
    with torch.no_grad():
        projection.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        projection.bias.copy_(torch.tensor([5.0, -3.0]))
    return projection


def test_embedding_error_in_an_ignored_direction_reads_as_no_error():
    projection = _speaker_projection_that_ignores_its_second_input()
    error_the_model_cannot_see = torch.tensor([[0.0, 1.0]])

    reading = effective_embedding_error(error_the_model_cannot_see, projection)

    torch.testing.assert_close(reading, torch.tensor([0.0]))


def test_embedding_error_in_an_influential_direction_reads_as_a_real_error():
    projection = _speaker_projection_that_ignores_its_second_input()
    error_the_model_reacts_to = torch.tensor([[1.0, 0.0]])

    reading = effective_embedding_error(error_the_model_reacts_to, projection)

    # The projection turns this into [1.0, 0.0], whose root mean square is sqrt(0.5).
    torch.testing.assert_close(reading, torch.tensor([0.5]).sqrt())


def test_embedding_error_ignores_the_projection_bias_because_it_cancels_in_a_difference():
    projection = _speaker_projection_that_ignores_its_second_input()
    error = torch.tensor([[1.0, 0.0]])

    reading_with_bias = effective_embedding_error(error, projection)
    with torch.no_grad():
        projection.bias.zero_()
    reading_without_bias = effective_embedding_error(error, projection)

    torch.testing.assert_close(reading_with_bias, reading_without_bias)


def test_error_quantiles_can_be_logged_from_bfloat16_errors():
    """Training runs in bf16-mixed precision, and torch.quantile() rejects bfloat16 inputs."""
    logged_metrics = {}

    class QuantileLoggerStub:
        QUANTILE_LABELS = StyleEncoderLightningModule.QUANTILE_LABELS
        _quantile_probs = torch.tensor(StyleEncoderLightningModule.QUANTILE_PROBS)
        _log_quantiles = StyleEncoderLightningModule._log_quantiles

        def log(self, name, value, **kwargs):
            logged_metrics[name] = value

    errors = torch.arange(101, dtype=torch.bfloat16)
    QuantileLoggerStub()._log_quantiles("acoustic", errors, batch_size=1)

    assert logged_metrics["error_quantiles/acoustic_p25"] == 25
    assert logged_metrics["error_quantiles/acoustic_p50"] == 50
    assert logged_metrics["error_quantiles/acoustic_p75"] == 75
    assert logged_metrics["error_quantiles/acoustic_p90"] == 90
    assert logged_metrics["error_quantiles/acoustic_p95"] == 95
    assert logged_metrics["error_quantiles/acoustic_p99"] == 99


def test_the_largest_absolute_error_is_logged_because_thresholds_are_tuned_to_it():
    logged_metrics = {}

    class QuantileLoggerStub:
        QUANTILE_LABELS = StyleEncoderLightningModule.QUANTILE_LABELS
        _quantile_probs = torch.tensor(StyleEncoderLightningModule.QUANTILE_PROBS)
        _log_quantiles = StyleEncoderLightningModule._log_quantiles

        def log(self, name, value, **kwargs):
            logged_metrics[name] = value

    errors = torch.tensor([0.1, 7.5, 0.3, 0.2])
    QuantileLoggerStub()._log_quantiles("rhythm", errors, batch_size=1)

    assert logged_metrics["error_quantiles/rhythm_p100"] == errors.max()


def test_error_quantiles_are_skipped_when_every_token_is_padding():
    class QuantileLoggerStub:
        QUANTILE_LABELS = StyleEncoderLightningModule.QUANTILE_LABELS
        _quantile_probs = torch.tensor(StyleEncoderLightningModule.QUANTILE_PROBS)
        _log_quantiles = StyleEncoderLightningModule._log_quantiles

        def log(self, name, value, **kwargs):
            raise AssertionError(f"nothing should be logged for an empty error tensor, got {name}")

    QuantileLoggerStub()._log_quantiles("acoustic", torch.empty(0), batch_size=1)


def test_masked_stats_pool_ignores_padded_frames():
    features = torch.tensor([[[1.0, 3.0, 100.0], [2.0, 6.0, 100.0]]])
    valid_frames = torch.tensor([[[1.0, 1.0, 0.0]]])

    pooled = masked_stats_pool(features, valid_frames)

    expected = torch.tensor([[2.0, 4.0, 1.0, 2.0]])
    torch.testing.assert_close(pooled, expected)


def test_style_encoder_returns_separate_embeddings_for_each_conditioning_path():
    encoder = StyleEncoder(n_feats=3, hidden_channels=4, n_layers=2, spk_emb_dim=5)
    mel = torch.randn(2, 3, 7)
    valid_frames = torch.ones(2, 1, 7)

    acoustic_embedding, rhythm_embedding = encoder(mel, valid_frames)

    assert acoustic_embedding.shape == (2, 5)
    assert rhythm_embedding.shape == (2, 5)


def test_the_two_predictors_share_no_weights_so_neither_loss_can_reach_the_other():
    encoder = StyleEncoder(n_feats=3, hidden_channels=4, n_layers=2, spk_emb_dim=5)
    mel = torch.randn(2, 3, 7)
    valid_frames = torch.ones(2, 1, 7)

    acoustic_embedding, _ = encoder(mel, valid_frames)
    acoustic_embedding.sum().backward()

    acoustic_params = list(encoder.acoustic_predictor.parameters())
    rhythm_params = list(encoder.rhythm_predictor.parameters())
    assert all(p.grad is not None for p in acoustic_params)
    assert all(p.grad is None for p in rhythm_params)
    shared_parameters = {id(p) for p in acoustic_params} & {id(p) for p in rhythm_params}
    assert not shared_parameters


def test_masked_smooth_l1_loss_is_independent_of_acoustic_feature_count():
    single_feature_prediction = torch.ones(1, 1, 2)
    single_feature_target = torch.zeros_like(single_feature_prediction)
    hundred_feature_prediction = single_feature_prediction.repeat(1, 100, 1)
    hundred_feature_target = torch.zeros_like(hundred_feature_prediction)
    valid_tokens = torch.ones(1, 1, 2)

    single_feature_loss = masked_smooth_l1_loss(
        single_feature_prediction,
        single_feature_target,
        valid_tokens,
        beta=1.0,
    )
    hundred_feature_loss = masked_smooth_l1_loss(
        hundred_feature_prediction,
        hundred_feature_target,
        valid_tokens,
        beta=1.0,
    )

    torch.testing.assert_close(single_feature_loss, torch.tensor(0.5))
    torch.testing.assert_close(hundred_feature_loss, single_feature_loss)


def test_masked_smooth_l1_loss_ignores_padded_tokens():
    prediction = torch.tensor([[[1.0, 100.0]]])
    target = torch.zeros_like(prediction)
    valid_tokens = torch.tensor([[[1.0, 0.0]]])

    loss = masked_smooth_l1_loss(prediction, target, valid_tokens, beta=1.0)

    torch.testing.assert_close(loss, torch.tensor(0.5))
