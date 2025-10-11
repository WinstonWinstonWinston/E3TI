from abc import ABC, abstractmethod
import torch
from typing import Dict


class Corrector(ABC):
    r"""
    Abstract class for defining a corrector function that corrects the input :math:`x` (for instance, wrapping back coordinates
    to a specific cell in periodic boundary conditions).
    """

    @abstractmethod
    def correct(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        Correct the input :math:`x`.

        :param x:
            Input to correct.
        :type x: torch.Tensor

        :return:
            Corrected input.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        r"""
        Correct the input :math:`x_1` based on the reference input :math:`x_0` (for instance, return the image of :math:`x_1` closest to  :math:`x_0` in
        periodic boundary conditions).

        :param x_0:
            Reference input.
        :type x_0: torch.Tensor
        :param x_1:
            Input to correct.
        :type x_1: torch.Tensor

        :return:
            Unwrapped x_1 value.
        :rtype: torch.Tensor
        """
        raise NotImplementedError
    
    def summarize_cfg(self):
        r"""
        Prints details about the configuration defining the corrector
        """
        raise NotImplementedError
    
class Interpolant(ABC):
    r"""
    Abstract class for defining an interpolant

    .. math::

        x_t = I(t, x_0, x_1) + \gamma(t)z
    
    in a stochastic interpolant between points  :math:`x_0` and  :math:`x_1` from two distributions  :math:`p_0` and  :math:`p_1` at times t.

    :param velocity_weight:
        Constant velocity_weight > 0 which scaless loss of the velocity
    :type velocity_weight: float
    :param denoiser_weight:
        Constant denoiser_weight > 0 which scaless loss of the denoiser
    :type velocity_weight: float
    """
    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        self.velocity_weight = velocity_weight
        self.denoiser_weight = denoiser_weight

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        r"""
        Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
        :math:`x_1` and function :math:`I(t, x_0, x_1)`

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor

        :return:
            Interpolated value :math:`x_t` and the latent noise :math:`z`.
        :rtype:  tuple[torch.Tensor, torch.Tensor]
        """
        raise NotImplementedError

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""
        Compute the derivative of the interpolant :math:`\dot{x}_t` with respect to time between points :math:`x_0` and :math:`x_1` 
        from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor

        :return:
            Derivative of the interpolant.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def get_corrector(self) -> Corrector:
        r"""
        Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
        conditions).

        :return:
            Corrector.
        :rtype: Corrector
        """
        raise NotImplementedError
    
    @abstractmethod
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Gamma function :math:`\gamma(t)` in the stochastic interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the gamma function :math:`\gamma(t)` at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_dot(self, t: torch.Tensor):
        r"""
        Time derivative of the gamma function :math:`\dot{\gamma}(t)` in the stochastic interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the gamma function :math:`\dot{\gamma}(t)` at the given times.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def summarize_cfg(self):
        """
        Prints details about the configuration defining the interpolant
        """
        self.get_corrector().summarize_cfg()
        print(f"[{self.__class__.__name__}] velocity_weight={self.velocity_weight:.6g}, denoiser_weight={self.denoiser_weight:.6g}")
    
    def loss(self, t, x_0, x_1, z, b, eta=None) -> Dict[str, torch.Tensor]:
        r"""
        Loss value for a batch of data. If the eta term is None this corresponds only to the velocity loss.
        Otherwise it gives a weighted average between them based off of init params velocity_weight, and denoiser_weight.

        * :math:`\mathcal{L}_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\|b\|^{2} - 2\, b \cdot \dot I\right]`
        * :math:`\mathcal{L}_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\|\eta\|^{2} - 2\, \eta \cdot z\right]`
        * :math:`\mathcal{L}(\theta) = \mathrm{velocity\_weight}\,\mathcal{L}_{\text{velocity}}(\theta) + \mathrm{denoiser\_weight}\,\mathcal{L}_{\text{denoiser}}(\theta)`

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_1`.
        :type x_0: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor
        :param b:
            Predicted velocity values :math:`b(t)` for :math:`x_t`.
        :type eta: torch.Tensor
        :param eta:
            Predicted denoiser values :math:`\eta(t)` for :math:`x_t`.
        :type eta: torch.Tensor

        :return:
            A dictionary of loss values with keys loss, loss_velocity, and loss_denoiser
        :rtype: dict[str, torch.Tensor]
        """
        interpolant_dot = self.interpolate_derivative(t,x_0,x_1,z)
        loss_velocity  = torch.mean(torch.einsum('BD,BD->B', b, b    ) - 2*torch.einsum('BD, BD', b, interpolant_dot))
        loss_denoiser  = torch.mean(torch.einsum('BD,BD->B', eta, eta) - 2*torch.einsum('BD, BD', eta, z) if eta is not None else torch.zeros_like(loss_velocity))
        loss = self.velocity_weight*loss_velocity + self.denoiser_weight*loss_denoiser
        return {"loss": loss,
                "loss_velocity": loss_velocity,
                "loss_denoiser": loss_denoiser}

class LinearInterpolant(Interpolant):
    r"""
    Abstract class for defining a spatially linear interpolant

    .. math::

        I(t, x_0, x_1) = \alpha(t)\cdot x_0 + \beta(t) \cdot x_1
     
    in a stochastic interpolant between points :math:`x_0` and :math:`x_1` from two distributions 
    :math:`p_0` and :math:`p_1` at times :math:`t`.
    """

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Interpolate between points x_0 and x_1 from two distributions p_0 and p_1 at times t.

        .. math::
            
            x_t = \alpha(t)\cdot x_0 + \beta(t) \cdot x_1 + \gamma(y)\cdot z

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor

        :return:
            Interpolated value :math:`x_t` and the latent noise :math:`z`.
        :rtype:  tuple[torch.Tensor, torch.Tensor]
        """
        z = torch.randn_like(x_0)
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime + z*self.gamma(t)
        return self.get_corrector().correct(x_t), z

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        r"""
         Compute the derivative of the interpolant :math:`\dot{x}_t` with respect to time between points :math:`x_0` and :math:`x_1` 
        from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

        .. math::
        
            \dot{x_t} = \dot{\alpha}(t)\cdot x_0 + \dot{\beta}(t)\cdot x_1 + \dot{\gamma}(y)\cdot z

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor
        :param x_0:
            Points sampled from :math:`p_0`.
        :type x_0: torch.Tensor
        :param x_1:
            Points sampled from :math:`p_1`.
        :type x_1: torch.Tensor
        :param z:
            Latent normally distributed noise :math:`z \sim N(0,1)`.
        :type z: torch.Tensor

        :return:
            Derivative of the interpolant \dot{x_t}.
        :rtype: torch.Tensor
        """
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime + self.gamma_dot(t) * z

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Alpha function :math:`\alpha(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the alpha function :math:`\alpha(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Derivative of the alpha function :math:`\dot{\alpha}(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the alpha function :math:`\dot{\alpha}(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Beta function :math:`\beta(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Values of the beta function :math:`\beta(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: torch.Tensor):
        r"""
        Derivative of the beta function :math:`\dot{\beta}(t)` in the linear interpolant.

        :param t:
            Times in :math:`t \in [0,1]`.
        :type t: torch.Tensor

        :return:
            Derivatives of the beta function :math:`\dot{\beta}(t)` at the given times :math:`t`.
        :rtype: torch.Tensor
        """
        raise NotImplementedError