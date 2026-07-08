import pytest
import torch

from openfold3.core.data.pipelines.featurization.loss_weights import (
    set_loss_weights,
    set_loss_weights_for_disordered_set,
)

_DUMMY_LOSS_SETTINGS = {
    "confidence_loss_names": ["plddt", "pae"],
    "diffusion_loss_names": ["diffusion"],
    "loss_weights": {"plddt": 1.0, "pae": 1.0, "diffusion": 2.0},
    "min_resolution": 0.0,
    "max_resolution": 9.0,
}


@pytest.mark.parametrize(
    "resolution, expected_confidence_weight",
    [
        pytest.param(3.0, 1.0, id="in-range-keeps-confidence"),
        pytest.param(0.0, 1.0, id="min-boundary-keeps-confidence"),
        pytest.param(9.0, 1.0, id="max-boundary-keeps-confidence"),
        pytest.param(None, 0.0, id="none-zeros-confidence"),
        pytest.param(9.1, 0.0, id="above-max-zeros-confidence"),
        pytest.param(-0.1, 0.0, id="below-min-zeros-confidence"),
    ],
)
def test_set_loss_weights(resolution, expected_confidence_weight):
    result = set_loss_weights(_DUMMY_LOSS_SETTINGS, resolution)

    assert result["plddt"] == torch.tensor([expected_confidence_weight])
    assert result["pae"] == torch.tensor([expected_confidence_weight])
    # diffusion weight should always be preserved
    assert result["diffusion"] == torch.tensor([2.0])


@pytest.mark.parametrize(
    "resolution, disable_non_protein, expected_confidence_weight",
    [
        pytest.param(3.0, False, 1.0, id="in-range-keeps-confidence"),
        pytest.param(3.0, True, 1.0, id="in-range-disable-non-protein"),
        pytest.param(None, False, 0.0, id="none-zeros-confidence"),
        pytest.param(9.1, True, 0.0, id="above-max-zeros-confidence"),
    ],
)
def test_set_loss_weights_for_disordered_set(
    resolution, disable_non_protein, expected_confidence_weight
):
    result = set_loss_weights_for_disordered_set(
        _DUMMY_LOSS_SETTINGS, resolution, disable_non_protein
    )

    assert result["plddt"] == torch.tensor([expected_confidence_weight])
    assert result["pae"] == torch.tensor([expected_confidence_weight])
    assert result["diffusion"] == torch.tensor([2.0])
    assert result["disable_non_protein_diffusion_weights"] == torch.tensor(
        [disable_non_protein]
    )
