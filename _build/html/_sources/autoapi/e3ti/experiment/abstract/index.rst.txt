e3ti.experiment.abstract
========================

.. py:module:: e3ti.experiment.abstract




Module Contents
---------------

.. py:class:: Experiment(cfg)

   Bases: :py:obj:`abc.ABC`


   Abstract base experiment class.

   All experiments must
   1: Save their configuration
   2: Contain a "run"
   3: Be able to give a descriptive print statement describing the config


   TODO: Comment me


   .. py:attribute:: cfg


   .. py:method:: run()
      :abstractmethod:



   .. py:method:: summarize_cfg()
      :abstractmethod:



   .. py:attribute:: __slots__
      :value: ()



