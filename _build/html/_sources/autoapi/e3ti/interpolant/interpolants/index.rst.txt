e3ti.interpolant.interpolants
=============================

.. py:module:: e3ti.interpolant.interpolants




Module Contents
---------------

.. py:class:: TemporallyLinearInterpolant(velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`e3ti.interpolant.abstract.LinearInterpolant`


   Linear interpolant I(t, x_0, x_1) = (1 - t) * x_0 + t * x_1 between points x_0 and x_1 from two distributions p_0
   and p_1 at times t.

   Construct linear interpolant.


   .. py:method:: alpha(t)

      Alpha function alpha(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: alpha_dot(t)

      Time derivative of the alpha function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta(t)

      Beta function beta(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta_dot(t)

      Time derivative of the beta function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: get_corrector()

      Get the corrector implied by the interpolant.

      :return:
          Identity corrector that does nothing.
      :rtype: Corrector



   .. py:method:: gamma(t)

      Gamma function gamma(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: gamma_dot(t)

      Time derivative of the gamma function in the stochastic interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



   .. py:method:: interpolate(t, x_0, x_1)

      Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

      In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
      :math:`x_1` and the alpha and beta functions.

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Points from p_0.
      :type x_0: torch.Tensor
      :param x_1: Points from p_1.
      :type x_1: torch.Tensor

      :returns: *tuple[torch.Tensor, torch.Tensor]* -- Interpolated value and the latent noise.



   .. py:method:: interpolate_derivative(t, x_0, x_1, z)

      Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
      :math:`t` with respect to time.

      In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
      the unwrapped :math:`x_1` and the alpha and beta functions.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor
      :param x_0: Points from :math:`p_0`.
      :type x_0: torch.Tensor
      :param x_1: Points from: math:`p_1`.
      :type x_1: torch.Tensor

      :returns: *torch.Tensor* -- Derivative of the interpolant.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: loss(t, x_0, x_1, z, b, eta=None)

      Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
      Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

      .. math::
      L_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\,\lVert b\rVert^2 - 2\, b \cdot \dot I\,\right] \\
      L_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\,\lVert \eta\rVert^2 - 2\, \eta \cdot z\,\right] \\
      L(\theta) = \text{velocity\_weight}\,L_{\text{velocity}}(\theta) + \text{denoiser\_weight}\,L_{\text{denoiser}}(\theta)

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Samples from the base distribution rho_0.
      :type x_0: torch.Tensor
      :param x_1: Samples from the data distribution rho_0.
      :type x_1: torch.Tensor
      :param z: Latent noise values :math:`z \sim \mathcal{N}(0, 1)`.
      :type z: torch.Tensor
      :param b: Predicted velocity values for :math:`x_t`.
      :type b: torch.Tensor
      :param eta: Predicted denoiser values for :math:`x_t`.
      :type eta: torch.Tensor

      :returns: *dict[str, torch.Tensor]* -- A dictionary of loss values, ``loss``, ``loss_velocity``, and ``loss_denoiser``.



   .. py:attribute:: __slots__
      :value: ()



.. py:class:: TrigonometricInterpolant(velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`e3ti.interpolant.abstract.LinearInterpolant`


   Trigonometric interpolant I(t, x_0, x_1) = cos(pi / 2 * t) * x_0 + sin(pi / 2 * t) * x_1 between points x_0 and x_1
   from two distributions p_0 and p_1 at times t.

   Construct trigonometric interpolant.


   .. py:method:: alpha(t)

      Alpha function alpha(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: alpha_dot(t)

      Time derivative of the alpha function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta(t)

      Beta function beta(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta_dot(t)

      Time derivative of the beta function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: get_corrector()

      Get the corrector implied by the interpolant.

      :return:
          Identity corrector that does nothing.
      :rtype: Corrector



   .. py:method:: gamma(t)

      Gamma function in the stochastic interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: gamma_dot(t)

      Time derivative of the gamma function in the stochastic interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



   .. py:method:: interpolate(t, x_0, x_1)

      Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

      In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
      :math:`x_1` and the alpha and beta functions.

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Points from p_0.
      :type x_0: torch.Tensor
      :param x_1: Points from p_1.
      :type x_1: torch.Tensor

      :returns: *tuple[torch.Tensor, torch.Tensor]* -- Interpolated value and the latent noise.



   .. py:method:: interpolate_derivative(t, x_0, x_1, z)

      Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
      :math:`t` with respect to time.

      In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
      the unwrapped :math:`x_1` and the alpha and beta functions.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor
      :param x_0: Points from :math:`p_0`.
      :type x_0: torch.Tensor
      :param x_1: Points from: math:`p_1`.
      :type x_1: torch.Tensor

      :returns: *torch.Tensor* -- Derivative of the interpolant.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: loss(t, x_0, x_1, z, b, eta=None)

      Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
      Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

      .. math::
      L_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\,\lVert b\rVert^2 - 2\, b \cdot \dot I\,\right] \\
      L_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\,\lVert \eta\rVert^2 - 2\, \eta \cdot z\,\right] \\
      L(\theta) = \text{velocity\_weight}\,L_{\text{velocity}}(\theta) + \text{denoiser\_weight}\,L_{\text{denoiser}}(\theta)

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Samples from the base distribution rho_0.
      :type x_0: torch.Tensor
      :param x_1: Samples from the data distribution rho_0.
      :type x_1: torch.Tensor
      :param z: Latent noise values :math:`z \sim \mathcal{N}(0, 1)`.
      :type z: torch.Tensor
      :param b: Predicted velocity values for :math:`x_t`.
      :type b: torch.Tensor
      :param eta: Predicted denoiser values for :math:`x_t`.
      :type eta: torch.Tensor

      :returns: *dict[str, torch.Tensor]* -- A dictionary of loss values, ``loss``, ``loss_velocity``, and ``loss_denoiser``.



   .. py:attribute:: __slots__
      :value: ()



.. py:class:: EncoderDecoderInterpolant(a = 1.0, switch_time = 0.5, power = 1.0, velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`e3ti.interpolant.abstract.LinearInterpolant`


   Encoder-decoder interpolant
   I(t, x_0, x_1) = cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p +  (t - switch_time * t)^p)) * 1_[0, switch_time) * x_0
                  + cos^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p)) * 1_(1-switch_time, 1] * x_1
   between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

   gamma(t) = sqrt(a) * sin^2(pi * (t - switch_time * t)^p / ((switch_time - switch_time * t)^p + (t - switch_time * t)^p))

   For a=1 p=1 and switch_time=0.5, this interpolant becomes
   I(t, x_0, x_1) = cos^2(pi * t) * 1_[0, switch_time) * x_0 + cos^2(pi * t) * 1_(1-switch_time, 1] * x_1,
   and gamma(t) = sin^2(pi * t) which was considered in the stochastic interpolants paper.

   Note that the time derivatives are only bounded for p>=0.5.

   :param a:
       Constant a > 0.
       Defaults to 1.0.
   :type a: float
   :param switch_time:
       Time in (0, 1) at which to switch from x_0 to x_1.
       Defaults to 0.5.
   :type switch_time: float
   :param power:
       Power p in the interpolant.
       Defaults to 1.0.
   :type power: float
   :param velocity_weight:
       Constant velocity_weight > 0 which scaless loss of the velocity
   :type velocity_weight: float
   :param denoiser_weight:
       Constant denoiser_weight > 0 which scaless loss of the denoiser
   :type velocity_weight: float


   :raises ValueError:
       If switch_time is not in (0,1) or power is less than 0.5.

   Construct encoder-decoder interpolant.


   .. py:method:: alpha(t)

      Alpha function alpha(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: alpha_dot(t)

      Time derivative of the alpha function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta(t)

      Beta function beta(t) in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta_dot(t)

      Time derivative of the beta function in the linear interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: get_corrector()

      Get the corrector implied by the interpolant.

      :return:
          Identity corrector that does nothing.
      :rtype: Corrector



   .. py:method:: gamma(t)

      Evaluate the gamma function gamma(t) in the latent variable gamma(t) * z at the times t.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Gamma function gamma(t).
      :rtype: torch.Tensor



   .. py:method:: gamma_derivative(t)

      Compute the derivative of the gamma function gamma(t) in the latent variable gamma(t) * z with respect to time.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivative of the gamma function.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



   .. py:method:: interpolate(t, x_0, x_1)

      Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

      In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
      :math:`x_1` and the alpha and beta functions.

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Points from p_0.
      :type x_0: torch.Tensor
      :param x_1: Points from p_1.
      :type x_1: torch.Tensor

      :returns: *tuple[torch.Tensor, torch.Tensor]* -- Interpolated value and the latent noise.



   .. py:method:: interpolate_derivative(t, x_0, x_1, z)

      Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
      :math:`t` with respect to time.

      In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
      the unwrapped :math:`x_1` and the alpha and beta functions.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor
      :param x_0: Points from :math:`p_0`.
      :type x_0: torch.Tensor
      :param x_1: Points from: math:`p_1`.
      :type x_1: torch.Tensor

      :returns: *torch.Tensor* -- Derivative of the interpolant.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: gamma_dot(t)
      :abstractmethod:


      Time derivative :math:`\gamma'(t)` in the stochastic interpolant.
      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of :math:`\gamma'(t)` at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: loss(t, x_0, x_1, z, b, eta=None)

      Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
      Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

      .. math::
      L_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\,\lVert b\rVert^2 - 2\, b \cdot \dot I\,\right] \\
      L_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\,\lVert \eta\rVert^2 - 2\, \eta \cdot z\,\right] \\
      L(\theta) = \text{velocity\_weight}\,L_{\text{velocity}}(\theta) + \text{denoiser\_weight}\,L_{\text{denoiser}}(\theta)

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Samples from the base distribution rho_0.
      :type x_0: torch.Tensor
      :param x_1: Samples from the data distribution rho_0.
      :type x_1: torch.Tensor
      :param z: Latent noise values :math:`z \sim \mathcal{N}(0, 1)`.
      :type z: torch.Tensor
      :param b: Predicted velocity values for :math:`x_t`.
      :type b: torch.Tensor
      :param eta: Predicted denoiser values for :math:`x_t`.
      :type eta: torch.Tensor

      :returns: *dict[str, torch.Tensor]* -- A dictionary of loss values, ``loss``, ``loss_velocity``, and ``loss_denoiser``.



   .. py:attribute:: __slots__
      :value: ()



.. py:class:: MirrorInterpolant(velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`e3ti.interpolant.abstract.LinearInterpolant`


   Mirror interpolant I(t, x_0, x_1) = x_1 between points x_0 and x_1 from the same distribution p_1 at times t.

   :param velocity_weight:
       Constant velocity_weight > 0 which scaless loss of the velocity
   :type velocity_weight: float
   :param denoiser_weight:
       Constant denoiser_weight > 0 which scaless loss of the denoiser
   :type velocity_weight: float

   Construct mirror interpolant.


   .. py:method:: alpha(t)

      Alpha function alpha(t) in the mirror interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: alpha_dot(t)

      Time derivative of the alpha function in the mirror interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the alpha function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta(t)

      Beta function beta(t) in the mirror interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: beta_dot(t)

      Time derivative of the beta function in the mirror interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the beta function at the given times.
      :rtype: torch.Tensor



   .. py:method:: gamma(t)

      Gamma function in the stochastic interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Values of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: gamma_dot(t)

      Time derivative of the gamma function in the stochastic interpolant.

      :param t:
          Times in [0,1].
      :type t: torch.Tensor

      :return:
          Derivatives of the gamma function at the given times.
      :rtype: torch.Tensor



   .. py:method:: get_corrector()

      Get the corrector implied by the interpolant.

      :return:
          Identity corrector that does nothing.
      :rtype: Corrector



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



   .. py:method:: interpolate(t, x_0, x_1)

      Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

      In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
      :math:`x_1` and the alpha and beta functions.

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Points from p_0.
      :type x_0: torch.Tensor
      :param x_1: Points from p_1.
      :type x_1: torch.Tensor

      :returns: *tuple[torch.Tensor, torch.Tensor]* -- Interpolated value and the latent noise.



   .. py:method:: interpolate_derivative(t, x_0, x_1, z)

      Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
      :math:`t` with respect to time.

      In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
      this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
      this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
      the unwrapped :math:`x_1` and the alpha and beta functions.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor
      :param x_0: Points from :math:`p_0`.
      :type x_0: torch.Tensor
      :param x_1: Points from: math:`p_1`.
      :type x_1: torch.Tensor

      :returns: *torch.Tensor* -- Derivative of the interpolant.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: loss(t, x_0, x_1, z, b, eta=None)

      Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
      Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

      .. math::
      L_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\,\lVert b\rVert^2 - 2\, b \cdot \dot I\,\right] \\
      L_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\,\lVert \eta\rVert^2 - 2\, \eta \cdot z\,\right] \\
      L(\theta) = \text{velocity\_weight}\,L_{\text{velocity}}(\theta) + \text{denoiser\_weight}\,L_{\text{denoiser}}(\theta)

      :param t: Times in [0,1].
      :type t: torch.Tensor
      :param x_0: Samples from the base distribution rho_0.
      :type x_0: torch.Tensor
      :param x_1: Samples from the data distribution rho_0.
      :type x_1: torch.Tensor
      :param z: Latent noise values :math:`z \sim \mathcal{N}(0, 1)`.
      :type z: torch.Tensor
      :param b: Predicted velocity values for :math:`x_t`.
      :type b: torch.Tensor
      :param eta: Predicted denoiser values for :math:`x_t`.
      :type eta: torch.Tensor

      :returns: *dict[str, torch.Tensor]* -- A dictionary of loss values, ``loss``, ``loss_velocity``, and ``loss_denoiser``.



   .. py:attribute:: __slots__
      :value: ()



