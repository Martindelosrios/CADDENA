Usage
=====

.. _installation:

Installation
------------

To use BATMAN, first install it using pip:

.. code-block:: console

   (.venv) $ pip install batman

Creating recipes
----------------

To retrieve a list of random ingredients,
you can use the ``batman.Model()`` function:

.. autofunction:: BATMAN.batman.Model

The ``kind`` parameter should be either ``"meat"``, ``"fish"``,
or ``"veggies"``. Otherwise, :py:func:`lumache.get_random_ingredients`
will raise an exception.

.. autoexception:: lumache.InvalidKindError

For example:

>>> import lumache
>>> lumache.get_random_ingredients()
['shells', 'gorgonzola', 'parsley']

