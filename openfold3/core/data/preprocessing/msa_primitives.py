
import pandas as pd

from openfold3.core.data.preprocessing.io import Msa


def process_uniprot_headers(msa: Msa) -> None:
    """Reformats the headers of an Msa object parsed from a UniProt MSA.

    The list of headers is converted into a DataFrame containing the uniprot_id,
    species_id, chain_start and chain_end columns. If the Msa only contains the
    query sequence, an empty DataFrame is returned.

    Args:
        msa (Msa): parsed Msa object

    Returns:
        None
    """

    # Embed into DataFrame
    if len(msa.headers) == 1:
        # Empty DataFrame for UniProt MSAs that only contain the query sequence
        headers = pd.DataFrame({'raw': msa.headers})
        headers = pd.DataFrame()
    else:
        headers = pd.DataFrame({'raw': msa.headers[1:]})
        headers = headers['raw'].str.split(r'[|_/:-]', expand=True)
        headers = headers[[1, 3, 4, 5]]
        headers.columns = ['uniprot_id', 'species_id', 'chain_start', 'chain_end']

    msa.headers = headers


def pair_msas():

    return