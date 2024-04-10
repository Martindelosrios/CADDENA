import h5py
from importlib_resources import files


def importtestset():
    ref = files("BATMAN") / "dataset/"
    data_path = str(ref)

    with h5py.File(data_path + "/testset.h5", "r") as data:
        pars_norm = data["pars_norm"][()]

    return pars_norm


def test_importtestset():
    pars_norm = importtestset()
    assert pars_norm[0, 0] == 0.3719487741506609
