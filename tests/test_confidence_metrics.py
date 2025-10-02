import math
import unittest
import torch

from openfold3.core.metrics.confidence import (
    get_bin_centers,
    probs_to_expected_error,
    compute_global_predicted_distance_error,
)
from openfold3.projects.of3_all_atom.project_entry import OF3ProjectEntry

class TestConfidenceMetrics(unittest.TestCase):
    def test_get_bin_centers_basic(self):
        # 0..1 split into 5 bins 
        centers = get_bin_centers(0.0, 1.0, 5)
        expected = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        self.assertTrue(torch.allclose(centers, expected, atol=1e-6))

    def test_probs_to_expected_error_uniform(self):
        # Uniform distribution across bins should give the midpoint (0.5) for [0,1]
        no_bins = 10
        probs = torch.ones(no_bins) / no_bins
        exp_err = probs_to_expected_error(probs, 0.0, 1.0, no_bins)
        self.assertTrue(torch.allclose(exp_err, torch.tensor(0.5), atol=1e-6))

    def test_shape(self):
        batch_size, num_samples, n_atom = 1, 5, 16
        proj_entry = OF3ProjectEntry()
        config = proj_entry.get_model_config_with_presets()
        logits = torch.randn((batch_size, num_samples, n_atom, config.confidence.plddt.no_bins))
        probs = torch.softmax(logits, dim=-1)
        error = probs_to_expected_error(probs, **config.confidence.plddt)
        self.assertTrue(error.shape == (batch_size, num_samples, n_atom))


if __name__ == "__main__":
    unittest.main()
