Usage
=====


.. _installation:
Installation
------------

To use CADDENA, we first need to install it from github using pip:

.. code-block:: console

   $ git clone https://github.com/Martindelosrios/CADDENA.git
   $ cd CADDENA
   (.venv) $ pip install .

Loading pre-trained models
--------------------------

Some pre-trained models are saved in ``caddena.models`` 
(you can check the list of available models in :ref:`Models`).
These models can be loaded doing:

.. code-block:: python

    from CADDENA import caddena, models

Before using them you have to load the weights obtained in the
training published on arxiv XXXX.XXXX. For a given model you can do:

.. code-block:: python

    model.load_weights()


Once the model is trained (or equivantely the pre-trained weights loaded)
you can compute the likelihood-to-evidence ratios doing:

.. code-block:: python

    caddena.ratio_estimation([x], pars_prior, [model])
    

where ``x`` is the observtion to be analysed (that must have the shape for which the model was trained)
and ``pars_prior`` is a np.array with parameters sampled from a choosen prior.

You can also estimate the ratios for a list of observations and models at the same time!!


.. code-block:: python

    caddena.ratio_estimation([x1, x2, ...], pars_prior, [model1, model2, ...])
