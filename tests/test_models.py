"""Model shape, loss and save/reload checks. Skipped if PyTorch is not installed."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from solarcast.losses import pseudo_huber, quantile_loss, weighted_mse
from solarcast.models import build_model
from solarcast.predict import Forecaster

MCFG = {"hidden": [16, 8], "dropout": 0.2, "attention_dim": 8}


@pytest.mark.parametrize("name", ["mlp", "attention"])
def test_output_shape(name):
    model = build_model(name, MCFG)
    assert model(torch.randn(5, 14)).shape == (5, 7)


@pytest.mark.parametrize("loss", [weighted_mse, pseudo_huber, quantile_loss])
def test_losses_are_scalar_nonnegative_and_differentiable(loss):
    pred = torch.randn(4, 7, requires_grad=True)
    value = loss(pred, torch.randn(4, 7))
    assert value.ndim == 0 and value.item() >= 0
    value.backward()
    assert pred.grad is not None


def make_bundle(name="mlp"):
    torch.manual_seed(0)
    model = build_model(name, MCFG)
    return {
        "architecture": name, "model_cfg": MCFG, "state_dict": model.state_dict(),
        "feature_names": [f"f{i}" for i in range(14)], "horizons": 7,
        "x_mean": np.zeros(14), "x_std": np.ones(14),
        "y_mean": np.full(7, 100.0), "y_std": np.full(7, 50.0),
        "interval": 0.8, "calibrated_half_width": np.full(7, 30.0),
    }


def test_saved_model_gives_identical_predictions(tmp_path):
    bundle = make_bundle()
    X = np.random.default_rng(0).normal(size=(6, 14))
    before = Forecaster(bundle).predict(X)
    torch.save(bundle, tmp_path / "m.pt")
    after = Forecaster.load(tmp_path / "m.pt").predict(X)
    np.testing.assert_allclose(before, after)


@pytest.mark.parametrize("method", ["mc_dropout", "calibrated"])
def test_interval_is_ordered_and_nonnegative(method):
    fc = Forecaster(make_bundle())
    X = np.random.default_rng(1).normal(size=(6, 14))
    mean, lower, upper = fc.predict_interval(X, method, n_samples=20)
    assert mean.shape == lower.shape == upper.shape == (6, 7)
    assert (lower <= upper).all() and (lower >= 0).all()
