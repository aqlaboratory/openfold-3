import pandas as pd
import pytest

from openfold3.core.data.tools.colabfold_msa_server import (
    fetch_author_to_label_chain_ids,
    remap_colabfold_template_chain_ids,
)


def _make_m8_dataframe(template_ids: list[str], m_index: int = 101) -> pd.DataFrame:
    """Build a minimal m8-format DataFrame for testing."""
    n = len(template_ids)
    return pd.DataFrame(
        {
            0: [m_index] * n,
            1: template_ids,
            2: [0.98] * n,
            3: [100] * n,
            4: [1] * n,
            5: [0] * n,
            6: [1] * n,
            7: [100] * n,
            8: [1] * n,
            9: [100] * n,
            10: [1e-10] * n,
            11: [100] * n,
            12: ["100M"] * n,
        }
    )


class TestFetchAuthorToLabelChainIds:
    """Tests for fetch_author_to_label_chain_ids (hits real RCSB API)."""

    def test_1rnb_author_a_maps_to_label_b(self):
        """1RNB: author chain A (protein) -> label chain B."""
        result = fetch_author_to_label_chain_ids({"1rnb"})

        assert "1rnb" in result
        a2l = result["1rnb"]
        assert a2l["A"] == ["B"]
        assert a2l["C"] == ["A"]

    def test_identity_mapping(self):
        """4PQX: author chain IDs match label chain IDs."""
        result = fetch_author_to_label_chain_ids({"4pqx"})

        assert "4pqx" in result
        assert result["4pqx"]["A"] == ["A"]

    def test_batch_query(self):
        """Multiple PDB IDs are fetched in a single request."""
        result = fetch_author_to_label_chain_ids({"1rnb", "4pqx"})

        assert "1rnb" in result
        assert "4pqx" in result

    def test_empty_set(self):
        """Empty input returns empty dict without API call."""
        assert fetch_author_to_label_chain_ids(set()) == {}


class TestRemapColabfoldTemplateChainIds:
    """Tests for remap_colabfold_template_chain_ids."""

    def test_remap_author_to_label(self):
        """1rnb_A (author) should be remapped to 1rnb_B (label)."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_A", "4pqx_A"]),
            m_with_templates={101},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert "rep1" in result
        remapped_ids = result["rep1"][1].tolist()
        assert remapped_ids[0] == "1rnb_B"
        assert remapped_ids[1] == "4pqx_A"

    def test_unknown_author_chain_raises(self):
        """When the author chain ID isn't in the API response, raise."""
        with pytest.raises(RuntimeError, match="Author chain Z not found in 1rnb"):
            remap_colabfold_template_chain_ids(
                template_alignments=_make_m8_dataframe(["1rnb_Z"]),
                m_with_templates={101},
                rep_ids=["rep1"],
                rep_id_to_m={"rep1": 101},
            )

    def test_skips_rep_without_templates(self):
        """Rep IDs not in m_with_templates should be skipped."""
        result = remap_colabfold_template_chain_ids(
            template_alignments=_make_m8_dataframe(["1rnb_A"]),
            m_with_templates={999},
            rep_ids=["rep1"],
            rep_id_to_m={"rep1": 101},
        )

        assert len(result) == 0
