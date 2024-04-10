import h5py
import swyft
import torch

from BATMAN.batman import Model
from importlib_resources import files


# Check if gpu is available
if torch.cuda.is_available():
    device = "gpu"
    print("Using GPU")
else:
    device = "cpu"
    print("Using CPU")

# Dataset for testing and loading previously trained models

ref = files("BATMAN") / "dataset/"
DATA_PATH = str(ref)
with h5py.File(DATA_PATH + "testset.h5", "r") as data:
    x_norm_rate = data["rate_norm"][()]
    x_norm_drate = data["drate_norm"][()]
    x_norm_s1s2 = data["s1s2_norm"][()]
    pars_norm = data["pars_norm"][()]

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

# ckpt_path = swyft.best_from_yaml(DATA_PATH + "O1_rate.yaml")
trainer_rate.test(network_rate, dm_test_rate, ckpt_path=DATA_PATH + "O1_rate.ckpt")

comments = """
This model was trained with simulations of data expected in XENON nT with 
an eft O1 dark matter model, varying the dark matter mass, the
scattering amplitude and the isospin angle.

You can extract the 1D marginal posteriors of each parameter or the 2D
marginal posteriors of combination of parameters.
"""
XENONnT_O1_rate = Model(network_rate, trainer_rate, comments=comments)

# Creating drate


# Now let's define a network that estimates all the 1D
#  and 2D marginal posteriors
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
            # torch.nn.Dropout(0.2),
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
        # z   = torch.tensor(B['z'])
        f = self.net(img)
        logratios1 = self.logratios1(f, b["z"])
        logratios2 = self.logratios2(f, b["z"])
        return logratios1, logratios2


trainer_drate = swyft.SwyftTrainer(
    accelerator=device, devices=1, max_epochs=2000, precision=64
)
network_drate = NetworkDrate()

# ckpt_path = swyft.best_from_yaml(DATA_PATH + "O1_drate.yaml")
trainer_drate.test(
    network_drate,
    dm_test_drate,
    ckpt_path=DATA_PATH + "O1_drate_epoch=22_val_loss=-1.51_train_loss=-1.42.ckpt",
)

XENONnT_O1_drate = Model(network_drate, trainer_drate, comments=comments)

# S1S2


# Now let's define a network that estimates all the 1D
#   and 2D marginal posteriors
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
        # z   = torch.tensor(b['z'])
        f = self.net(img)
        logratios1 = self.logratios1(f, b["z"])
        logratios2 = self.logratios2(f, b["z"])
        return logratios1, logratios2


trainer_s1s2 = swyft.SwyftTrainer(
    accelerator=device, devices=1, max_epochs=2500, precision=64
)
network_s1s2 = NetworkS1s2()

# ckpt_path = swyft.best_from_yaml(DATA_PATH + "O1_s1s2.yaml")
trainer_s1s2.test(
    network_s1s2,
    dm_test_s1s2,
    ckpt_path=DATA_PATH + "O1_s1s2_epoch=4_val_loss=-1.59_train_loss=-1.79-v2.ckpt",
)

XENONnT_O1_s1s2 = Model(network_s1s2, trainer_s1s2, comments=comments)
