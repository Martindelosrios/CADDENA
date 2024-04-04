import swyft
import numpy as np

class Model:
    '''
    Class to save trained models

    Attributes
    ----------

    Methods
    -------
    '''
    def __init__(self, network, trainer, comments = 'No added comments'):
        self.network = network
        self.trainer = trainer
        self.comments = comments

    def __repr__(self):
        output = self.comments
        return output


def ratio_estimation(obs, prior, models):
    '''
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

    List of Likelihood-to-Evidence ratio of the parameters given the observation for
    all the listed models.
    '''
    if len(obs) != len(models): 'The number of observations does not match the number of models'
    
    prior_sample = swyft.Samples(z = prior)
    
    logratios1D = []
    logratios2D = []
    for imodel, model in enumerate(models):
        print(imodel)
        obs_sample = swyft.Sample(x = obs[imodel])
        output = model.trainer.infer(model.network, obs_sample, prior_sample)
        logratios1D.append( np.asarray(output[0].logratios) )
        logratios2D.append( np.asarray(output[1].logratios) )

    return logratios1D, logratios2D
