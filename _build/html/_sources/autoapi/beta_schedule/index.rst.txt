beta_schedule
=============

.. py:module:: beta_schedule




Module Contents
---------------

.. py:class:: BetaSchedulerBase(diffusion_steps, beta_min=None, beta_max=None)

   .. py:attribute:: diffusion_steps


   .. py:attribute:: beta_min
      :value: None



   .. py:attribute:: beta_max
      :value: None



   .. py:method:: get_alpha_bars()


   .. py:method:: get_alphas()


   .. py:method:: get_snr_weight()


   .. py:method:: get_betas()
      :abstractmethod:



   .. py:method:: get_snr()


.. py:class:: LinearBetaScheduler(diffusion_steps, beta_min=None, beta_max=None)

   Bases: :py:obj:`BetaSchedulerBase`


   .. py:method:: get_betas()


   .. py:attribute:: diffusion_steps


   .. py:attribute:: beta_min
      :value: None



   .. py:attribute:: beta_max
      :value: None



   .. py:method:: get_alpha_bars()


   .. py:method:: get_alphas()


   .. py:method:: get_snr_weight()


   .. py:method:: get_snr()


.. py:class:: SigmoidalBetaScheduler(diffusion_steps, beta_min=None, beta_max=None)

   Bases: :py:obj:`BetaSchedulerBase`


   .. py:method:: get_betas()


   .. py:attribute:: diffusion_steps


   .. py:attribute:: beta_min
      :value: None



   .. py:attribute:: beta_max
      :value: None



   .. py:method:: get_alpha_bars()


   .. py:method:: get_alphas()


   .. py:method:: get_snr_weight()


   .. py:method:: get_snr()


