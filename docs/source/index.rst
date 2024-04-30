BATMAN (BAyesian Toolkit for Machine learning ANalysis) pacakage
----------------------------------------------------------------

.. figure:: ../../.badges/LOGO.png

.. figure:: ../../.badges/coverage.svg
.. figure:: ../../.badges/python.svg
    :target: https://www.python.org/downloads/release/python-380/
.. figure:: https://readthedocs.org/projects/batman-rtd/badge/?version=latest
    :target: https://batman-rtd.readthedocs.io/en/latest/?badge=latest


Welcome to BATMAN! 

This is a code for analysing dark matter direct detection experiment
with Bayesian statistic using Machine Learning tools.

This code is presented in arxiv XXXX.XXXX and is build on top of `SWYFT <https://github.com/undark-lab/swyft>`_.

The main goal of BATMAN is to estimate the full posterior distribution
of some interesting parameter given new data :math:`P(\Theta|X)`!

This is done using the `TMNRE <https://arxiv.org/abs/2111.08030>`_ 
(Truncated Marginal Neural Ratio Estimation) method which computes the likelihood-to-evidence
ratio (:math:`r(X,\Theta) = \frac{P(X|\Theta)}{P(X)}`) using neural networks.
This can then be translated to the posterior just multiplying by a prior :math:`P(\Theta)`.

This is a modular library, so if you have several observations :math:`X_{i}`
(that can be of different types) coming from the same physical model 
(with some :math:`\Theta` parameters), you can analyse each observations with a
given TMNRE model specifically trained for it and compute
individual ratios :math:`r_{i}(X_{i}, \Theta)`. 
But then you can combine all the results just assuming that the prior for a
given analysis is the posterior of the previous analysis which leads to
:math:`P(\Theta|X_{n}) = \prod_{i}^{n}r_{i}(X_{i}, \Theta)P(\Theta)`

.. note::

   This project is under active development.


Contents
--------

.. toctree::
   Home <self>

.. toctree::
    Getting started <usage>

.. toctree::
    Available models <models>

.. toctree::
    Examples <examples>
