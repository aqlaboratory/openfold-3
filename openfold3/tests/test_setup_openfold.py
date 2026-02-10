from openfold3.setup_openfold import setup_biotite_ccd


def test_setup_biotite_ccd(tmp_path):
    ccd_path = tmp_path / "test_ccd.cif"
    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert ccd_path.exists()
    assert has_downloaded

    has_downloaded = setup_biotite_ccd(ccd_path=ccd_path, force_download=False)
    assert not has_downloaded
