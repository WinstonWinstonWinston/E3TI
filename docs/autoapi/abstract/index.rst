abstract
========

.. py:module:: abstract




Module Contents
---------------

.. py:class:: E3TIPrior

   Bases: :py:obj:`abc.ABC`


   Abstract interface for prior samplers used in E3TI workflows.


   .. py:method:: sample(batch, stratified)
      :abstractmethod:


      Draw samples from the prior and return the same batch object type with updated keys.

      :param batch:
          A torch batch of geometric data objects coming from a data loader.
      :type batch: torch_geometric.data.Data
      :param stratified:
          Whether to use stratified sampling over the time variable
      :type stratified: bool

      :return:
          The input batch type with modified keys (e.g., velocity, score, denoised point).
      :rtype: torch_geometric.data.Data



   .. py:method:: log_prob(batch)
      :abstractmethod:


      Compute log-probabilities under the prior for variables referenced by batch.

      :param batch:
          A torch batch of geometric data objects coming from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          Per-example log p(x) with a leading batch dimension.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()
      :abstractmethod:



   .. py:attribute:: __slots__
      :value: ()



