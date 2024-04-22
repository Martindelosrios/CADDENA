from BATMAN import batman, models
import numpy as np
import h5py
from importlib_resources import files


def estimate_ratios():
    ref = files("BATMAN") / "dataset/"
    DATA_PATH = str(ref)
    with h5py.File(DATA_PATH + "/testset.h5", "r") as data:
      x_norm_rate  = data['x_norm_rate'][()]
      x_norm_drate = data['x_norm_drate'][()]
      x_norm_s1s2  = data['x_norm_s1s2'][()]
      pars_norm    = data['pars_norm'][()]
      pars_min = data.attrs["pars_min"]
      pars_max = data.attrs["pars_max"]
      x_min_rate = data.attrs["x_min_rate"]
      x_max_rate = data.attrs["x_max_rate"]
      x_min_drate = data.attrs["x_min_drate"]
      x_max_drate = data.attrs["x_max_drate"]

    np.random.seed(28890)
    i = np.random.randint(len(pars_norm))
    print(i)
    
    pars_true   = pars_norm[i,:]
    x_obs_rate  = x_norm_rate[i,:]
    x_obs_drate = x_norm_drate[i,:]
    x_obs_s1s2  = x_norm_s1s2[i,:].reshape(1,96,96)
    
    pars_prior = np.random.uniform(low = 0, high = 1, size = (1_000, 3))
    
    models.XENONnT_O1_rate.train_model()
    models.XENONnT_O1_drate.train_model()
    models.XENONnT_O1_s1s2.train_model()

    logratios1D, logratios2D = batman.ratio_estimation([x_obs_rate, x_obs_drate, x_obs_s1s2], pars_prior, [models.XENONnT_O1_rate, models.XENONnT_O1_drate, models.XENONnT_O1_s1s2])

    return logratios1D, logratios2D


def test_estimate_ratios():
    logratios1D, logratios2D = estimate_ratios()

    assert np.allclose(logratios1D[0][0,:], np.array([-0.17116392,  0.08299124, -0.06267583]), atol = 1e-5)
    assert np.allclose(logratios1D[1][0,:], np.array([-0.1066963 ,  0.18183607, -0.01303009]), atol = 1e-5)
    assert np.allclose(logratios1D[2][0,:], np.array([-0.24844716,  0.60588804, -0.1671143 ]), atol = 1e-5)
    assert np.allclose(logratios2D[0][0,:], np.array([-0.55746773, -0.17068126,  0.09961353]), atol = 1e-5)
    assert np.allclose(logratios2D[1][0,:], np.array([0.23994138 , 0.02681413 ,  0.49061815]), atol = 1e-5)
    assert np.allclose(logratios2D[2][0,:], np.array([ 0.63054098, -0.26630078,  0.74245657]), atol = 1e-5)
