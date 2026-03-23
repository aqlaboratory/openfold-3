from openfold3.core.data.tools.rscb import (
    fetch_label_to_author_chain_ids,
    get_model_ranking_fit,
)


class TestFetchLabelToAuthorChainIds:
    """Tests for fetch_label_to_author_chain_ids (hits real RCSB API)."""

    def test_1rnb_label_to_author(self):
        """1RNB: label chain B -> author chain A (protein)."""
        result = fetch_label_to_author_chain_ids({"1rnb"})

        assert "1rnb" in result
        l2a = result["1rnb"]
        assert l2a["B"] == "A"
        assert l2a["A"] == "C"

    def test_identity_mapping(self):
        """4PQX: label chain IDs match author chain IDs."""
        result = fetch_label_to_author_chain_ids({"4pqx"})

        assert "4pqx" in result
        assert result["4pqx"]["A"] == "A"

    def test_batch_query(self):
        """Multiple PDB IDs are fetched in a single request."""
        result = fetch_label_to_author_chain_ids({"1rnb", "4pqx"})

        assert "1rnb" in result
        assert "4pqx" in result

    def test_empty_set(self):
        """Empty input returns empty dict without API call."""
        assert fetch_label_to_author_chain_ids(set()) == {}


class TestGetModelRankingFit:
    """Tests for get_model_ranking_fit (hits real RCSB API)."""

    def test_entry_with_ligands(self):
        """4PQX has ligands with ranking_model_fit scores."""
        result = get_model_ranking_fit("4pqx")

        assert isinstance(result, dict)
        assert len(result) > 0
        for rcsb_id, score in result.items():
            assert rcsb_id.startswith("4PQX.")
            assert isinstance(score, (int, float))

    def test_entry_without_ligands(self):
        """1RNB (protein-only) returns empty dict."""
        result = get_model_ranking_fit("1rnb")

        assert result == {}

    def test_nonexistent_entry(self):
        """Invalid PDB ID returns empty dict without raising."""
        result = get_model_ranking_fit("0000")

        assert result == {}
