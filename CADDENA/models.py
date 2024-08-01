import h5py
import numpy as np
import swyft
import torch
from importlib_resources import files

from CADDENA.caddena import Model

# Check if gpu is available
if torch.cuda.is_available():
    device = "gpu"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

# Dataset for testing and loading previously trained models

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
x_norm_rate = (x_norm_rate - x_min_rate) / (x_max_rate - x_min_rate)
x_norm_rate = x_norm_rate.reshape(len(x_norm_rate), 1)

x_norm_drate = np.log10(drate_testset)
x_norm_drate = (x_norm_drate - x_min_drate) / (x_max_drate - x_min_drate)

x_norm_s1s2 = s1s2_testset
x_norm_s1s2 = x_norm_s1s2 / x_max_s1s2

samples_test_rate = swyft.Samples(x=x_norm_rate, z=pars_norm)
dm_test_rate = swyft.SwyftDataModule(
    samples_test_rate, fractions=[0.0, 0.0, 1], batch_size=32
)

samples_test_drate = swyft.Samples(x=x_norm_drate, z=pars_norm)
dm_test_drate = swyft.SwyftDataModule(
    samples_test_drate, fractions=[0.0, 0.0, 1], batch_size=32
)

samples_test_s1s2 = swyft.Samples(x=x_norm_s1s2, z=pars_norm)
dm_test_s1s2 = swyft.SwyftDataModule(
    samples_test_s1s2, fractions=[0.0, 0.0, 1], batch_size=32
)


# Creating model for XENON nT with rate
class NetworkRate(swyft.SwyftModule):
    def __init__(self):
        super().__init__()
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=1, num_params=3, varnames="pars_norm"
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=1, marginals=marginals, varnames="pars_norm"
        )

    def forward(self, a, b):
        logratios1 = self.logratios1(a["x"], b["z"])
        logratios2 = self.logratios2(a["x"], b["z"])
        return logratios1, logratios2


trainer_rate = swyft.SwyftTrainer(
    accelerator=device, devices=1, max_epochs=2000, precision=64
)
network_rate = NetworkRate()

comments = r"""
    XENONnT_O1_rate:

    This model was trained using synthetic data of the expected total
    rate in xenonNT for dark matter particles with O(1)
    with :math:`M_{dm} \in [6-1000] GeV`, :math:`\sigma = [10^{-50}-10^{-43}] cm^{2}`
    and :math:`\theta = [-\pi/2-\pi/2]`.
    In order to analyse a new data x it must be a np.array with shape (n,1),
    where n is the number of observations to be analysed.

    You can extract the 1D marginal posteriors of each parameter or the 2D
    marginal posteriors of combination of parameters.
"""
XENONnT_O1_rate = Model(
    network_rate,
    trainer_rate,
    path_to_weights=DATA_PATH + "/O1_rate.ckpt",
    test_data=dm_test_rate,
    comments=comments,
)

# Creating drate model


class NetworkDrate(swyft.SwyftModule):
    def __init__(self, lr=1e-3, gamma=1.0):
        super().__init__()
        self.optimizer_init = swyft.OptimizerInit(
            torch.optim.Adam,
            dict(lr=lr, weight_decay=1e-5),
            torch.optim.lr_scheduler.ExponentialLR,
            dict(gamma=gamma),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Linear(58, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 1000),
            torch.nn.ReLU(),
            torch.nn.Linear(1000, 500),
            torch.nn.ReLU(),
            torch.nn.Linear(500, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 5),
        )
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=5, num_params=3, varnames="pars_norm"
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=5, marginals=marginals, varnames="pars_norm"
        )

    def forward(self, a, b):
        img = torch.tensor(a["x"])
        f = self.net(img)
        logratios1 = self.logratios1(f, b["z"])
        logratios2 = self.logratios2(f, b["z"])
        return logratios1, logratios2


trainer_drate = swyft.SwyftTrainer(
    accelerator=device, devices=1, max_epochs=2000, precision=64
)
network_drate = NetworkDrate()

comments = """
    XENONnT_O1_drate:

    This model was trained using synthetic data of the expected differential
    rate in xenonNT for dark matter particles with O(1)
    with :math:`M_{dm} in [6-1000] GeV`, :math:`sigma = [10^{-50}-10^{-43}] cm^{2}`
    and :math:`theta = [-pi/2-pi/2]`.
    In order to analyse a new data x it must be a np.array with shape (n,59),
    where n is the number of observations to be analysed.

    You can extract the 1D marginal posteriors of each parameter or the 2D
    marginal posteriors of combination of parameters.
"""
XENONnT_O1_drate = Model(
    network_drate,
    trainer_drate,
    path_to_weights=DATA_PATH + "/O1_drate.ckpt",
    test_data=dm_test_drate,
    comments=comments,
)

# Let's create the S1S2 model


class NetworkS1s2(swyft.SwyftModule):
    def __init__(self, lr=1e-3, gamma=1.0):
        super().__init__()
        self.optimizer_init = swyft.OptimizerInit(
            torch.optim.Adam,
            dict(lr=lr, weight_decay=1e-5),
            torch.optim.lr_scheduler.ExponentialLR,
            dict(gamma=gamma),
        )
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Flatten(),
            torch.nn.Linear(10580, 50),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(50, 10),
        )
        marginals = ((0, 1), (0, 2), (1, 2))
        self.logratios1 = swyft.LogRatioEstimator_1dim(
            num_features=10, num_params=3, varnames="pars_norm"
        )
        self.logratios2 = swyft.LogRatioEstimator_Ndim(
            num_features=10, marginals=marginals, varnames="pars_norm"
        )

    def forward(self, a, b):
        img = torch.tensor(a["x"])
        f = self.net(img)
        logratios1 = self.logratios1(f, b["z"])
        logratios2 = self.logratios2(f, b["z"])
        return logratios1, logratios2


trainer_s1s2 = swyft.SwyftTrainer(
    accelerator=device, devices=1, max_epochs=2500, precision=64
)
network_s1s2 = NetworkS1s2()

comments = """
    XENONnT_O1_s1s2:

    This model was trained using synthetic data of the expected S1S2
    signal in xenonNT for dark matter particles with O(1)
    with :math:`M_{dm} in [6-1000] GeV`, :math:`sigma = [10^{-50}-10^{-43}] cm^{2}`
    and :math:`\theta = [-pi/2-pi/2]`.
    In order to analyse a new data x it must be a np.array with shape (n,1,97,70),
    where n is the number of observations to be analysed.

    You can extract the 1D marginal posteriors of each parameter or the 2D
    marginal posteriors of combination of parameters.
"""
XENONnT_O1_s1s2 = Model(
    network_s1s2,
    trainer_s1s2,
    path_to_weights=DATA_PATH + "/O1_s1s2.ckpt",
    test_data=dm_test_s1s2,
    comments=comments,
)
