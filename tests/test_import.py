import h5py
from importlib_resources import files
import swyft
import torch
#from BATMAN.batman import Model


def importtestset():
    ref = files("BATMAN") / "dataset/"
    data_path = str(ref)

    with h5py.File(data_path + "/testset.h5", "r") as data:
        pars_norm = data["pars_norm"][()]

    return pars_norm


def test_importtestset():
    pars_norm = importtestset()
    assert pars_norm[0, 0] == 0.3719487741506609

#def createModel():
#    # Check if gpu is available
#    if torch.cuda.is_available():
#        device = "gpu"
#        print("Using GPU")
#    else:
#        device = "cpu"
#        print("Using CPU")
#
#
#    class NetworkS1s2(swyft.SwyftModule):
#        def __init__(self, lr=1e-3, gamma=1.0):
#            super().__init__()
#            self.optimizer_init = swyft.OptimizerInit(
#                torch.optim.Adam,
#                dict(lr=lr, weight_decay=1e-5),
#                torch.optim.lr_scheduler.ExponentialLR,
#                dict(gamma=gamma),
#            )
#            self.net = torch.nn.Sequential(
#                torch.nn.Conv2d(1, 10, kernel_size=5),
#                torch.nn.MaxPool2d(2),
#                torch.nn.ReLU(),
#                torch.nn.Dropout(0.2),
#                torch.nn.Conv2d(10, 20, kernel_size=5, padding=2),
#                torch.nn.MaxPool2d(2),
#                torch.nn.ReLU(),
#                torch.nn.Dropout(0.2),
#                torch.nn.Flatten(),
#                torch.nn.Linear(10580, 50),
#                torch.nn.ReLU(),
#                torch.nn.Dropout(0.2),
#                torch.nn.Linear(50, 10),
#            )
#            marginals = ((0, 1), (0, 2), (1, 2))
#            self.logratios1 = swyft.LogRatioEstimator_1dim(
#                num_features=10, num_params=3, varnames="pars_norm"
#            )
#            self.logratios2 = swyft.LogRatioEstimator_Ndim(
#                num_features=10, marginals=marginals, varnames="pars_norm"
#            )
#
#        def forward(self, a, b):
#            img = torch.tensor(a["x"])
#            f = self.net(img)
#            logratios1 = self.logratios1(f, b["z"])
#            logratios2 = self.logratios2(f, b["z"])
#            return logratios1, logratios2
#
#
#    trainer_s1s2 = swyft.SwyftTrainer(
#        accelerator=device, devices=1, max_epochs=2500, precision=64
#    )
#    network_s1s2 = NetworkS1s2()
#    
#    comments = """
#    This model was trained with simulations of s1-s2 data expected in XENON nT with 
#    an eft O1 dark matter model, varying the dark matter mass, the
#    scattering amplitude and the isospin angle in the ranges [], [],
#    and [] respectively.
#    
#    You can extract the 1D marginal posteriors of each parameter or the 2D
#    marginal posteriors of combination of parameters.
#    """
#    XENONnT_O1_s1s2 = Model(network_s1s2, trainer_s1s2, comments=comments) 
#    return XENONnt_O1_s1s2
#
#def test_createModel():
#    comments = """
#    This model was trained with simulations of s1-s2 data expected in XENON nT with 
#    an eft O1 dark matter model, varying the dark matter mass, the
#    scattering amplitude and the isospin angle in the ranges [], [],
#    and [] respectively.
#    
#    You can extract the 1D marginal posteriors of each parameter or the 2D
#    marginal posteriors of combination of parameters.
#    """
#
#    XENONnt_O1_s1s2 = createModel()
#    assert XENONnt_O1_s1s2.comments == comments 
