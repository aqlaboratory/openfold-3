"""
Contains functions around clustering sequences or molecules or finding representatives.
"""


def map_chain_to_representative(
    query_seq_dict: dict[str, str], repr_seq_dict: dict[str, str]
) -> dict[str, str]:
    """Maps chains to their representative chains."""

    # Convert to seq -> chain mapping for easier lookup
    repr_seq_to_chain = {seq: chain for chain, seq in repr_seq_dict.items()}

    query_to_repr = {}

    # Map each query chain to its representative
    for query_chain, query_seq in query_seq_dict.items():
        repr_chain = repr_seq_to_chain.get(query_seq)

        query_to_repr[query_chain] = repr_chain

    return query_to_repr
