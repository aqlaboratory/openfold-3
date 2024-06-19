# TODO add license

# Some biotite examples
from tempfile import gettempdir
import biotite.database.rcsb as rcsb
import biotite.structure.io.pdbx as pdbx

cif_file_path = rcsb.fetch("1l2y", "cif", gettempdir())
cif_file = pdbx.PDBxFile.read(cif_file_path)