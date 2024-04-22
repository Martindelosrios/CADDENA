Usage
=====


.. _installation:
Installation
------------

To use BATMAN, first install it from github using pip:

.. code-block:: console

   (.venv) $ git clone https://github.com/Martindelosrios/BATMAN.git


Loading pre-trained models
--------------------------

To pre-trained models are saved in ``batman.models`` module that
can be loaded doing:

.. code-block:: python

    from BATMAN import batman, models

In order to load the weights obtained in the training published on
arxiv XXXX.XXXX you have to use .

.. code-block:: python

    models.XENONnT_O1_rate.load_weights()


Once the model trained (or equivantely the pre-trained weights loaded)
you can compute the likelihood-to-evidence ratios doing:

.. code-block:: python

    batman.ratio_estimation([x_obs_rate], pars_prior, [models.XENONnT_O1_rate])
    

where ``pars_prior`` is a np.array with parameters sampled from a choosen prior.

