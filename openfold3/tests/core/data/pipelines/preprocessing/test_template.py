from pathlib import Path

from openfold3.core.data.io.sequence.template import (
    A3mParser,
    parse_template_alignment,
)
from openfold3.core.data.io.structure.cif import _load_ciffile
from openfold3.core.data.primitives.structure.metadata import (
    get_asym_id_to_canonical_seq_dict,
    get_author_to_label_chain_ids,
    get_label_to_author_chain_id_dict,
    resolve_author_to_label_chain_id,
)

_TEST_DATA_DIR = Path(__file__).parent


class TestTemplatePreprocessor:
    def test_template_has_author_chain_id(self):
        """Verify author->label chain ID resolution for 1RNB.

        https://github.com/aqlaboratory/openfold-3/issues/101

        In 1RNB, author chain "A" is label chain "B" (the protein barnase).
        The ColabFold alignment reports "1rnb_A" which must be resolved to
        label chain "B" before the sequence can be looked up.
        """
        alignment_file = _TEST_DATA_DIR / "colabfold_template.m8"
        query_seq_str = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"
        templates = parse_template_alignment(
            aln_path=Path(alignment_file),
            query_seq_str=query_seq_str,
            max_sequences=200,
        )

        # find the offending "1rnb_A"
        template = templates[16]
        assert template.chain_id == "A" and template.entry_id == "1rnb"

        template_structure_file = _TEST_DATA_DIR / f"{template.entry_id}.cif"
        cif_file = _load_ciffile(template_structure_file)

        chain_id_seq_map = get_asym_id_to_canonical_seq_dict(cif_file)
        label_to_author = get_label_to_author_chain_id_dict(cif_file)
        author_to_label_chain_ids = get_author_to_label_chain_ids(label_to_author)
        label_chain_id = resolve_author_to_label_chain_id(
            author_to_label_chain_ids[template.chain_id],
            chain_id_seq_map=chain_id_seq_map,
        )

        # Author "A" -> label "B" (the protein chain)
        assert label_chain_id == "B"

        template_sequence = chain_id_seq_map.get(label_chain_id)

        parser = A3mParser(max_sequences=None)
        parsed = parser(
            (
                f">query_X/1-{len(query_seq_str)}\n"
                f"{query_seq_str}\n"
                f">{template.entry_id}_{label_chain_id}/{1}-{len(template_sequence)}\n"
                f"{template_sequence}\n"
            ),
            query_seq_str,
            realign=True,
        )

        assert len(parsed) == 2
        assert parsed[0].seq_id == 1
        assert parsed[1].seq_id < 1
