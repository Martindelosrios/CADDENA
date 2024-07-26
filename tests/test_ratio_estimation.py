from CADDENA import caddena, models
import numpy as np
import h5py
from importlib_resources import files


def estimate_ratios():
    ref = files("CADDENA") / "dataset/"
    DATA_PATH = str(ref)
    with h5py.File(DATA_PATH + "/testset.h5", "r") as data:
        pars_testset = data["pars_testset"][()]
        rate_testset = data["rate_testset"][()]
        drate_testset = data["drate_testset"][()]
        s1s2_testset = data["s1s2_testset"][()]
        pars_min = data.attrs["pars_min"]
        pars_max = data.attrs["pars_max"]
        x_min_rate = data.attrs["x_min_rate"]
        x_max_rate = data.attrs["x_max_rate"]
        x_min_drate = data.attrs["x_min_drate"]
        x_max_drate = data.attrs["x_max_drate"]
        x_max_s1s2 = data.attrs["x_max_s1s2"]
    
    pars_norm = (pars_testset - pars_min) / (pars_max - pars_min)
    
    x_norm_rate = np.log10(rate_testset)
    #x_norm_rate = (x_norm_rate - x_min_rate) / (x_max_rate - x_min_rate)
    x_norm_rate = x_norm_rate / x_max_rate
    x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)
 
    x_norm_drate = np.log10(drate_testset)
    x_norm_drate = (x_norm_drate - x_min_drate) / (x_max_drate - x_min_drate)
    
    x_norm_s1s2 = (s1s2_testset)
    x_norm_s1s2 = x_norm_s1s2 / x_max_s1s2

    np.random.seed(28890)
    i = np.random.randint(len(pars_norm))
    print(i)
    
    pars_true   = pars_norm[i,:]
    x_obs_rate  = x_norm_rate[i].reshape(1)
    x_obs_drate = x_norm_drate[i,:]
    x_obs_s1s2  = x_norm_s1s2[i,:].reshape(1,96,96)
    
    pars_prior = np.random.uniform(low = 0, high = 1, size = (1_000, 3))
    
    models.XENONnT_O1_rate.load_weights()
    models.XENONnT_O1_drate.load_weights()
    models.XENONnT_O1_s1s2.load_weights()

    logratios1D, logratios2D = caddena.ratio_estimation([x_obs_rate, x_obs_drate, x_obs_s1s2], pars_prior, [models.XENONnT_O1_rate, models.XENONnT_O1_drate, models.XENONnT_O1_s1s2])

    return logratios1D, logratios2D


def test_estimate_ratios():
    logratios1D, logratios2D = estimate_ratios()

    assert np.allclose(logratios1D[0][0,:], np.array([-0.16239598,  0.20058311, -0.06536658]), atol = 1e-5)
    assert np.allclose(logratios1D[1][0,:], np.array([-0.08454858,  0.55239703, -0.21410568]), atol = 1e-5)
    assert np.allclose(logratios1D[2][0,:], np.array([-0.19126089,  0.38586499, -0.06993887]), atol = 1e-5)
    assert np.allclose(logratios2D[0][0,:], np.array([ 0.06297819, -0.25379332,  0.31567388]), atol = 1e-5)
    assert np.allclose(logratios2D[1][0,:], np.array([ 0.99333211, -0.19759012,  1.06851521]), atol = 1e-5)
    assert np.allclose(logratios2D[2][0,:], np.array([ 0.3337919 , -0.36944396,  0.12885156]), atol = 1e-5)
