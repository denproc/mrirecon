
MRI Reconstruction Toolbox in PyTorch
_____________________________________

MRI Reconstruction Toolbox offers a collection of the most common MRI reconstruction algorithms implemented in PyTorch.
The Toolbox facilitates the usage of reconstruction methods as a part of Deep Learning pipelines in medical domain.
Our package takes advantage of batched matrix operations, which can be performed on both CPU and GPU devices.

Available Algorithms
--------------------

- SENSE reconstruction for parallel imaging (`pdf <https://onlinelibrary.wiley.com/doi/pdfdirect/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S>`_)
- kt-SENSE reconstruction for dynamic magnetic resonance imaging (MRI) (`pdf <https://onlinelibrary.wiley.com/doi/pdf/10.1002/mrm.10611>`_)


.. installation-section-start

Installation
------------
`MRI Reconstruction Toolbox in PyTorch  <https://github.com/denproc/mrirecon>`_ can be installed using ``pip`` or ``git``.


If you use ``pip``, you can install it with:

.. code-block:: sh

    $ pip install mrirecon

If you want to use the latest features straight from the master, clone `MRI Reconstruction repo <https://github.com/denproc/mrirecon>`_:

.. code-block:: sh

   git clone https://github.com/denproc/mrirecon.git
   cd mrirecon
   python setup.py install

.. installation-section-end

.. citation-section-start

Citation
--------
If you use  in your project, please, cite it as follows.

.. code-block:: tex

   @misc{mrirecon,
     title={MRI Reconstruction Toolbox in PyTorch},
     url={https://github.com/denproc/mrirecon},
     note={Open-source software available at https://github.com/denproc/mrirecon},
     author={Denis Prokopenko},
     year={2022},
   }

.. citation-section-end

.. contacts-section-start

Contacts
--------
**Denis Prokopenko** - `@denproc <https://github.com/denproc>`_ - ``d.prokopenko@outlook.com``

.. contacts-section-end
