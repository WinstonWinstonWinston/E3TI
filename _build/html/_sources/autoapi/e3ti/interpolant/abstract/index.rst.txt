e3ti.interpolant.abstract
=========================

.. py:module:: e3ti.interpolant.abstract




Module Contents
---------------

.. py:class:: Corrector

   Bases: :py:obj:`abc.ABC`


   Abstract interface for coordinate/feature correction.

   For instance Use this to implement operations like wrapping back coordinates
   to a specific cell in periodic boundary conditions


   .. py:method:: correct(x)
      :abstractmethod:


      Return a corrected version of :math:`x`.

      :param x: Input tensor to correct.
      :type x: torch.Tensor

      :returns: The corrected tensor.

      :raises NotImplementedError: Subclasses must implement this method.



   .. py:method:: unwrap(x_0, x_1)
      :abstractmethod:


      Correct the input :math:`x_1` based on the reference input :math:`x_0` (for instance, return the image of :math:`x_1` closest to :math:`x_0` in
      periodic boundary conditions).

      :param x_0: Reference input.
      :type x_0: torch.Tensor
      :param x_1: Input to correct.
      :type x_1: torch.Tensor

      :returns: *torch.Tensor* -- Unwrapped x_1 value.

      :raises NotImplementedError: Subclasses must implement this method.



   .. py:method:: summarize_cfg()
      :abstractmethod:


      Prints details about the configuration defining the corrector.

      :returns: None

      :raises NotImplementedError: Subclasses must implement this method.



   .. py:attribute:: __slots__
      :value: ()



.. py:class:: Interpolant(velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`abc.ABC`


   Abstract class for defining an interpolant

   .. math::
   x_t = I(t, x_0, x_1) + \gamma(t) z

   between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

   :param velocity_weight: Constant velocity_weight > 0 which scaless loss of the velocity
   :type velocity_weight: float
   :param denoiser_weight: Constant denoiser_weight > 0 which scaless loss of the denoiser
   :type denoiser_weight: float


   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: interpolate(t, x_0, x_1)
      :abstractmethod:


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

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: interpolate_derivative(t, x_0, x_1, z)
      :abstractmethod:


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



   .. py:method:: get_corrector()
      :abstractmethod:


      Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
      conditions).

      :returns: *Corrector* -- Corrector.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: gamma(t)
      :abstractmethod:


      Gamma function :math:`\gamma(t)` in the stochastic interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of :math:`\gamma(t)` at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: gamma_dot(t)
      :abstractmethod:


      Time derivative :math:`\gamma'(t)` in the stochastic interpolant.
      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of :math:`\gamma'(t)` at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



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



.. py:class:: LinearInterpolant(velocity_weight = 1.0, denoiser_weight = 1.0)

   Bases: :py:obj:`Interpolant`


   Abstract class for defining an interpolant
   :math:`I(t, x_0, x_1) = \alpha(t) x_0 + \beta(t) x_1`
   in a stochastic setting between points :math:`x_0` and :math:`x_1` from distributions :math:`p_0` and :math:`p_1` at time :math:`t`.


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



   .. py:method:: alpha(t)
      :abstractmethod:


      Alpha function :math:`\alpha(t)` in the linear interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of the alpha function at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: alpha_dot(t)
      :abstractmethod:


      Time derivative of the alpha function :math:`\dot{\alpha}(t)` in the linear interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Derivatives of the alpha function at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: beta(t)
      :abstractmethod:


      Beta function :math:`\beta(t)` in the linear interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of the beta function at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: beta_dot(t)
      :abstractmethod:


      Time derivative of the beta function :math:`\dot{\beta}(t)` in the linear interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Derivatives of the beta function at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:attribute:: velocity_weight
      :value: 1.0



   .. py:attribute:: denoiser_weight
      :value: 1.0



   .. py:method:: get_corrector()
      :abstractmethod:


      Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
      conditions).

      :returns: *Corrector* -- Corrector.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: gamma(t)
      :abstractmethod:


      Gamma function :math:`\gamma(t)` in the stochastic interpolant.

      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of :math:`\gamma(t)` at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: gamma_dot(t)
      :abstractmethod:


      Time derivative :math:`\gamma'(t)` in the stochastic interpolant.
      :param t: Times in :math:`[0,1]`.
      :type t: torch.Tensor

      :returns: *torch.Tensor* -- Values of :math:`\gamma'(t)` at the given times.

      :raises NotImplementedError: Must be implemented by subclasses.



   .. py:method:: summarize_cfg()

      Prints details about the configuration defining the interpolant



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



