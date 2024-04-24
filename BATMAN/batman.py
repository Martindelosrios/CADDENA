import warnings

import matplotlib.pyplot as plt
import numpy as np
import swyft
from scipy.integrate import simps, trapezoid
from scipy.interpolate import CloughTocher2DInterpolator


class Model:
    """
    Class to save trained models

    Parameters
    ----------

    network: SwyftModule object that contains the network
             that would be saved.

    trainer: SwyftTrainer object that contains the trainer
             of the trained network.

    path_to_weigths: Path to pre-trained weights.

    trained_flag: Boolean that marks if the model is already trained.

    comments: str. with all the information of the trained network.

    test_data: swyft.lightning.data.SwyftDataModule object with the data that will be used for testing the model.
    """

    def __init__(
        self,
        network,
        trainer,
        path_to_weights=None,
        trained_flag=False,
        comments="No added comments",
        test_data=None,
    ):
        self.network = network
        self.trainer = trainer
        self.path_to_weights = path_to_weights
        self.trained_flag = trained_flag
        self.test_data = test_data
        self.comments = comments

    def __repr__(self):
        output = self.comments
        if self.trained_flag is False:
            output = output + "\n NOT TRAINED \n"
        else:
            output = output + "\n READY TO USE :) \n"
        return output

    def load_weights(self, path_to_weights=None, test_data=None):
        """
        Method for loading pre-saved weights.

        Parameters
        ----------

        path_to_weights: Path to the pre-saved weights.

        test_data: swyft.lightning.data.SwyftDataModule object with the data that will be used to test the model.
        """
        print("Training model...")
        if path_to_weights is None:
            path_to_weights = self.path_to_weights
        print("Reading weights from \n")
        print(path_to_weights)
        if test_data is None:
            test_data = self.test_data

        self.trainer.test(self.network, test_data, ckpt_path=path_to_weights)
        self.trained_flag = True
        self.path_to_weights = path_to_weights
        return None


def ratio_estimation(obs: list, prior: np.array, models: list) -> list:
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
        raise ValueError(
            "The number of observations does not match the number of models"
        )

    for i in models:
        if not i.trained_flag:
            warnings.warn("You are using a model that is not trained yet.")

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


def plot1d(
    predictions,
    pars_prior,
    pars_true,
    ax=None,
    par=1,
    xlabel=r"$\log_{10}(\sigma)$",
    ylabel=r"$P(\sigma|x)\ /\ P(\sigma)$",
    flip=False,
    fill=True,
    linestyle="solid",
    color="black",
    fac=1,
    pars_min=np.array([1.0, 1.0, 1.0]),
    pars_max=np.array([1.0, 1.0, 1.0]),
):
    """
    Function to plot the 1D marginal posterior of a given parameter.

    Parameters
    ----------

    predictions: list of predictions made with batman.ratio_estimation()

    pars_prior: np.array with parameters sampled from a chosen prior.

    pars_true: np.array with the real parameter of the model.

    ax: matplotlib axis where the plot will be saved.
        If None, an axis will be created and returned. Default = None

    par: Integer for indicating the parameters that will be plotted.
         Default = 1.

    xlabel: Str. with the xlabel of the plot.

    ylabel: Str. with the ylabel of the plot.

    flip: Bool indicating if the axis will be flipped or not. Default = False

    fill: Bool indicating if the contours should be filled or not. Default = True

    linestyle: Linestyle for the contours. Default = 'solid'

    color: Color for the contours. Defatul = 'black'

    fac: Factor for improving the distribution visualization. Default = 1.

    pars_min: np.array with the minimum values of the analised parameters.
    Needed for re-normalizing to physical values. Default = [1,1,1]

    pars_max: np.array with the maximum values of the analised parameters.
    Needed for re-normalizing to physical values. Default = [1,1,1]

    Returns
    -------

    matplotlib axis containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
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
    predictions,
    pars_prior,
    pars_true,
    ax=None,
    fill=True,
    line=False,
    linestyle="solid",
    color="black",
    pars_min=np.array([1.0, 1.0, 1.0]),
    pars_max=np.array([1.0, 1.0, 1.0]),
):
    """
    Function to plot the 2D marginal posterior of the first and second parameters.

    Parameters
    ----------

    predictions: list of predictions made with batman.ratio_estimation()

    pars_prior: np.array with parameters sampled from a chosen prior.

    pars_true: np.array with the real parameter of the model.

    ax: matplotlib axis where the plot will be saved.
        If None, an axis will be created and returned. Default = None

    fill: Bool indicating if the contours should be filled or not. Default = True

    line: bool indicating if a line should be plotted for each contour. Default = False

    linestyle: Linestyle for the contours. Default = 'solid'

    color: Color for the contours. Defatul = 'black'

    pars_min: np.array with the minimum values of the analised parameters.
    Needed for re-normalizing to physical values. Default = [1,1,1]

    pars_max: np.array with the maximum values of the analised parameters.
    Needed for re-normalizing to physical values. Default = [1,1,1]

    Returns
    -------

    matplotlib axis containing the plot.
    """

    if ax is None:
        fig, ax = plt.subplots(1, 1)
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
