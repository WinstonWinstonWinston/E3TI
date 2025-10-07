e3ti.experiment.train
=====================

.. py:module:: e3ti.experiment.train






Module Contents
---------------

.. py:data:: logger

.. py:data:: logging_levels
   :value: ('debug', 'info', 'warning', 'error', 'exception', 'fatal', 'critical')


.. py:class:: Train(cfg)

   Bases: :py:obj:`e3ti.experiment.abstract.Experiment`


   TODO: Comment me


   .. py:attribute:: data_cfg


   .. py:attribute:: train_cfg


   .. py:attribute:: module_cfg


   .. py:attribute:: train_dataset
      :value: None



   .. py:attribute:: test_dataset
      :value: None



   .. py:attribute:: valid_dataset
      :value: None



   .. py:attribute:: datamodule
      :value: None



   .. py:attribute:: train_device_ids


   .. py:attribute:: module


   .. py:method:: run()


   .. py:method:: summarize_cfg()

      TODO: Add in train specific cfg summarizer



   .. py:attribute:: cfg


   .. py:attribute:: __slots__
      :value: ()



