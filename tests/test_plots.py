from BATMAN import batman, models
import numpy as np
import h5py
from importlib_resources import files
import matplotlib.pyplot as plt


def doplots():
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

    models.XENONnT_O1_rate.load_weights()
    models.XENONnT_O1_drate.load_weights()
    models.XENONnT_O1_s1s2.load_weights()

    logratios1D, logratios2D = batman.ratio_estimation([x_obs_rate, x_obs_drate, x_obs_s1s2], pars_prior, [models.XENONnT_O1_rate, models.XENONnT_O1_drate, models.XENONnT_O1_s1s2])

    ax1 = batman.plot1d([logratios1D[0]], pars_prior, pars_true, par = 0)
    ax2 = batman.plot2d(logratios2D, pars_prior, pars_true, fill = True, line = True, linestyle = 'solid', color = 'black')
    
    return ax1, ax2

def test_doplots():
    ax1, ax2 = doplots()

    assert ax1.xaxis.get_label().get_text() == '$\\log_{10}(\\sigma)$'
    assert ax2.xaxis.get_label().get_text() == '$M_{DM}$ [GeV]'
