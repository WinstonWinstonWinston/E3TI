normal
======

.. py:module:: normal




Module Contents
---------------

.. py:class:: NormalPrior(mean, std)

   Bases: :py:obj:`e3ti.prior.abstract.E3TIPrior`


   Abstract interface for prior samplers used in E3TI workflows.

   :param cfg:
       Configuration object (dataclass/dict/omegaconf) holding prior hyperparameters.
   :type cfg: Any


   .. py:attribute:: std


   .. py:attribute:: mean


   .. py:method:: sample(batch, stratified)

      Draw samples from the prior and return the same batch object type with updated keys.

      :param batch:
          A torch batch of geometric data objects coming from a data loader.
      :type batch: torch_geometric.data.Data
      :param stratified:
          Whether to use stratified sampling over the time variable
      :type stratified: bool

      :return:
          The input batch type with modified keys.
      :rtype: torch_geometric.data.Data



   .. py:method:: log_prob(batch)

      Compute log-probabilities under the prior for variables referenced by batch.

      :param batch:
          A torch batch of geometric data objects coming from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          Per-example log p(x) with a leading batch dimension.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()


   .. py:attribute:: __slots__
      :value: ()



