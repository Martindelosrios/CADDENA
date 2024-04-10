import numpy as np
import swyft
import h5py

from scipy.integrate import simps, trapezoid
from scipy.interpolate import CloughTocher2DInterpolator
from importlib_resources import files


class Model:
    """
    Class to save trained models

    Parameters
    ----------

    network: swyft.network object that contains the network
             that would be saved.

    trainer: swyft.trainer object that contains the trainer
             of the trained network.

    comments: str. with all the information of the trained network.

    """

    def __init__(self, network, trainer, comments="No added comments"):
        self.network = network
        self.trainer = trainer
        self.comments = comments

    def __repr__(self):
        output = self.comments
        return output


def ratio_estimation(obs, prior, models):
    """
    Function that computes the likelihood-to-evidence ratio
    for the given observation, using the all the
    listed models.

    Parameters
    ----------

    obs: List of observations that will be analyzed. The type of data
         will depend on the listed models.

    prior: Np array with the model parameters that will be analyzed
           sampled from the desires prior.

    models: List of models to be used to analyze obs.

    Returns
    -------

    List of Likelihood-to-Evidence ratio of the parameters given
    the observation for all the listed models.
    """
    if len(obs) != len(models):
        "The number of observations does not match the number of models"

    prior_sample = swyft.Samples(z=prior)

    logratios1d = []
    logratios2d = []
    for imodel, model in enumerate(models):
        print(imodel)
        obs_sample = swyft.Sample(x=obs[imodel])
        output = model.trainer.infer(model.network, obs_sample, prior_sample)
        logratios1d.append(np.asarray(output[0].logratios))
        logratios2d.append(np.asarray(output[1].logratios))

    return logratios1d, logratios2d

# This data is needed only for the plots
#  so should not be here!!!
# I have to tidy a bit the code and put this
# data inside the plotting function
ref = files("BATMAN") / "dataset/"
DATA_PATH = str(ref)
with h5py.File(DATA_PATH + "/testset.h5", "r") as data:
    pars_min = data.attrs["pars_min"]
    pars_max = data.attrs["pars_max"]
    x_min_rate = data.attrs["x_min_rate"]
    x_max_rate = data.attrs["x_max_rate"]
    x_min_drate = data.attrs["x_min_drate"]
    x_max_drate = data.attrs["x_max_drate"]



def plot1d(
    ax,
    predictions,
    pars_prior,
    pars_true,
    par=1,
    xlabel=r"$\log_{10}(\sigma)$",
    ylabel=r"$P(\sigma|x)\ /\ P(\sigma)$",
    flip=False,
    fill=True,
    linestyle="solid",
    color="black",
    fac=1,
):
    # Let's put the results in arrays
    parameter = pars_prior[:, par] * (pars_max[par] - pars_min[par]) + pars_min[par]
    ratios = np.zeros_like(predictions[0][:, par])
    for pred in predictions:
        ratios = ratios + np.asarray(pred[:, par])
    ratios = np.exp(ratios)

    ind_sort = np.argsort(parameter)
    ratios = ratios[ind_sort]
    parameter = parameter[ind_sort]

    # Let's compute the integrated probability for different threshold
    cuts = np.linspace(np.min(ratios), np.max(ratios), 100)
    integrals = []
    for c in cuts:
        ratios0 = np.copy(ratios)
        ratios0[np.where(ratios < c)[0]] = 0
        integrals.append(trapezoid(ratios0, parameter) / trapezoid(ratios, parameter))

    integrals = np.asarray(integrals)

    # Let's compute the thresholds corresponding to 0.9
    #  and 0.95 integrated prob
    cut90 = cuts[np.argmin(np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin(np.abs(integrals - 0.95))]

    if not flip:
        ax.plot(10**parameter, fac * ratios, c=color, linestyle=linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_between(
                10 ** parameter[ind],
                fac * ratios[ind],
                [0] * len(ind),
                color="darkcyan",
                alpha=0.3,
            )
            ind = np.where(ratios > cut95)[0]
            ax.fill_between(
                10 ** parameter[ind],
                fac * ratios[ind],
                [0] * len(ind),
                color="darkcyan",
                alpha=0.5,
            )
        ax.axvline(
            x=10 ** (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]),
            color="black",
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
    else:
        ax.plot(fac * ratios, 10**parameter, c=color, linestyle=linestyle)
        if fill:
            ind = np.where(ratios > cut90)[0]
            ax.fill_betweenx(
                10 ** parameter[ind],
                [0] * len(ind),
                fac * ratios[ind],
                color="darkcyan",
                alpha=0.3,
            )
            ind = np.where(ratios > cut95)[0]
            ax.fill_betweenx(
                10 ** parameter[ind],
                [0] * len(ind),
                fac * ratios[ind],
                color="darkcyan",
                alpha=0.5,
            )
        ax.axhline(
            y=10 ** (pars_true[par] * (pars_max[par] - pars_min[par]) + pars_min[par]),
            color="black",
        )
        ax.set_xlabel(ylabel)
        ax.set_ylabel(xlabel)
        # ax.set_xlim(-0.1,8)
        ax.set_ylim(1e-50, 1e-42)
        ax.set_yscale("log")

    return ax


def plot2d(
    ax,
    predictions,
    pars_prior,
    pars_true,
    fill=True,
    line=False,
    linestyle="solid",
    color="black",
):

    # results_pars = np.asarray(predictions[0][1].params)
    results_pars = pars_prior  # * (pars_max - pars_min) + pars_min

    results = np.zeros_like(predictions[0][:, 0])
    for pred in predictions:
        results = results + np.asarray(pred[:, 0])

    # Let's make an interpolation function
    interp = CloughTocher2DInterpolator(results_pars[:, 0:2], np.exp(results))

    def interpol(log_m, log_sigma):
        m_norm = (log_m - pars_min[0]) / (pars_max[0] - pars_min[0])
        sigma_norm = (log_sigma - pars_min[1]) / (pars_max[1] - pars_min[1])
        return interp(m_norm, sigma_norm)

    # Let's estimate the value of the posterior in a grid
    nvals = 20
    m_values = np.logspace(0.8, 2.99, nvals)
    s_values = np.logspace(-49.0, -43.1, nvals)
    m_grid, s_grid = np.meshgrid(m_values, s_values)

    ds = np.log10(s_values[1]) - np.log10(s_values[0])
    dm = np.log10(m_values[1]) - np.log10(m_values[0])

    res = np.zeros((nvals, nvals))
    for m in range(nvals):
        for s in range(nvals):
            res[m, s] = interpol(np.log10(m_values[m]), np.log10(s_values[s]))
    res[np.isnan(res)] = 0
    # Let's compute the integral
    norm = simps(simps(res, dx=dm, axis=1), dx=ds)

    # Let's look for the 0.9 probability threshold
    cuts = np.linspace(np.min(res), np.max(res), 100)
    integrals = []
    for c in cuts:
        res0 = np.copy(res)
        res0[np.where(res < c)[0], np.where(res < c)[1]] = 0
        integrals.append(simps(simps(res0, dx=dm, axis=1), dx=ds) / norm)
    integrals = np.asarray(integrals)

    cut90 = cuts[np.argmin(np.abs(integrals - 0.9))]
    cut95 = cuts[np.argmin(np.abs(integrals - 0.95))]
    if fill:
        ax.contourf(
            m_values,
            s_values,
            res.T,
            levels=[0, cut90, np.max(res)],
            colors=["white", "darkcyan"],
            alpha=0.3,
            linestyles=["solid"],
        )
        ax.contourf(
            m_values,
            s_values,
            res.T,
            levels=[0, cut95, np.max(res)],
            colors=["white", "darkcyan"],
            alpha=0.5,
            linestyles=["solid"],
        )
    if line:
        ax.contour(
            m_values,
            s_values,
            res.T,
            levels=[0, cut90],
            colors=[color],
            linestyles=["solid"],
        )
        ax.contour(
            m_values,
            s_values,
            res.T,
            levels=[0, cut95],
            colors=[color],
            linestyles=["--"],
        )

    ax.axvline(
        x=10 ** (pars_true[0] * (pars_max[0] - pars_min[0]) + pars_min[0]),
        color="black",
    )
    ax.axhline(
        y=10 ** (pars_true[1] * (pars_max[1] - pars_min[1]) + pars_min[1]),
        color="black",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_{DM}$ [GeV]")
    ax.set_ylabel(r"$\sigma$ $[cm^{2}]$")

    return ax
