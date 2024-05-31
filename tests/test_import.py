import h5py
import swyft
import torch
from importlib_resources import files

from CADDENA.caddena import Model


def importtestset():
    ref = files("CADDENA") / "dataset/"
    data_path = str(ref)

    with h5py.File(data_path + "/testset.h5", "r") as data:
        x_norm_rate = data["x_norm_rate"][()]
        x_norm_drate = data["x_norm_drate"][()]
        x_norm_s1s2 = data["x_norm_s1s2"][()]
        pars_norm = data["pars_norm"][()]
        pars_min = data.attrs["pars_min"]
        pars_max = data.attrs["pars_max"]
        x_min_rate = data.attrs["x_min_rate"]
        x_max_rate = data.attrs["x_max_rate"]
        x_min_drate = data.attrs["x_min_drate"]
        x_max_drate = data.attrs["x_max_drate"]

    return x_norm_rate, x_norm_drate, x_norm_s1s2, pars_norm, pars_min, pars_max, x_min_rate, x_max_rate, x_min_drate, x_max_drate


def test_importtestset():
    x_norm_rate, x_norm_drate, x_norm_s1s2, pars_norm, pars_min, pars_max, x_min_rate, x_max_rate, x_min_drate, x_max_drate = importtestset()
    assert x_norm_rate[0,0] == 0.24924694017490479
    assert x_norm_drate[0, 0] == 0.0142018779342723
    assert x_norm_s1s2[0,0,21,47] == 6
    assert pars_norm[0, 0] == 0.3719487741506609
    assert pars_min[0] == 0.77828586
    assert pars_max[0] == 2.99964286
    assert x_min_rate == 3.423245873936808
    assert x_max_rate == 5.563303059369412
    assert x_min_drate[0] == 5.0 
    assert x_max_drate[0] == 34085.0


def createmodel():
    # Check if gpu is available
    if torch.cuda.is_available():
        device = "gpu"
        print("Using GPU")
    else:
        device = "cpu"
        print("Using CPU")

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
    This model was trained with simulations of s1-s2 data expected in XENON nT with
    an eft O1 dark matter model, varying the dark matter mass, the
    scattering amplitude and the isospin angle in the ranges [], [],
    and [] respectively.
    
    You can extract the 1D marginal posteriors of each parameter or the 2D
    marginal posteriors of combination of parameters.
    """
    XENONnT_O1_s1s2 = Model(network_s1s2, trainer_s1s2, comments=comments)
    return XENONnT_O1_s1s2


def test_createmodel():
    comments = """
    This model was trained with simulations of s1-s2 data expected in XENON nT with
    an eft O1 dark matter model, varying the dark matter mass, the
    scattering amplitude and the isospin angle in the ranges [], [],
    and [] respectively.
    
    You can extract the 1D marginal posteriors of each parameter or the 2D
    marginal posteriors of combination of parameters.
    """

    XENONnT_O1_s1s2 = createmodel()
    assert XENONnT_O1_s1s2.comments == comments
