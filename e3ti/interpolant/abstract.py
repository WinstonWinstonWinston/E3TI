from abc import ABC, abstractmethod
import torch
from typing import Dict


class Corrector(ABC):
    """Abstract interface for coordinate/feature correction.

    For instance Use this to implement operations like wrapping back coordinates
    to a specific cell in periodic boundary conditions
    """

    @abstractmethod
    def correct(self, x: torch.Tensor) -> torch.Tensor:
        """Return a corrected version of :math:`x`.

        Args:
            x (torch.Tensor): Input tensor to correct.

        Returns:
            The corrected tensor.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def unwrap(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """
        Correct the input :math:`x_1` based on the reference input :math:`x_0` (for instance, return the image of :math:`x_1` closest to :math:`x_0` in
        periodic boundary conditions).

        Args:
            x_0 (torch.Tensor): Reference input.
            x_1 (torch.Tensor): Input to correct.

        Returns:
            torch.Tensor: Unwrapped x_1 value.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError
    
    def summarize_cfg(self):
        """
        Prints details about the configuration defining the corrector.

        Returns:
            None
        
        Raises:
            NotImplementedError: Subclasses must implement this method.
        """

        raise NotImplementedError
    
class Interpolant(ABC):
    r"""
    Abstract class for defining an interpolant

    .. math::
    x_t = I(t, x_0, x_1) + \gamma(t) z

    between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times :math:`t`.

    Args:
        velocity_weight (float): Constant velocity_weight > 0 which scaless loss of the velocity
        denoiser_weight (float): Constant denoiser_weight > 0 which scaless loss of the denoiser
    """

    def __init__(self, velocity_weight: float = 1.0, denoiser_weight: float = 1.0) -> None:
        self.velocity_weight = velocity_weight
        self.denoiser_weight = denoiser_weight

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

        In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
        :math:`x_1` and the alpha and beta functions.

        Args:
            t (torch.Tensor): Times in [0,1].
            x_0 (torch.Tensor): Points from p_0.
            x_1 (torch.Tensor): Points from p_1.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Interpolated value and the latent noise.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
        :math:`t` with respect to time.

        In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
        the unwrapped :math:`x_1` and the alpha and beta functions.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.
            x_0 (torch.Tensor): Points from :math:`p_0`.
            x_1 (torch.Tensor): Points from: math:`p_1`.

        Returns:
            torch.Tensor: Derivative of the interpolant.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def get_corrector(self) -> Corrector:
        """
        Get the corrector implied by the interpolant (for instance, a corrector that considers periodic boundary
        conditions).

        Returns:
            Corrector: Corrector.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    @abstractmethod
    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Gamma function :math:`\gamma(t)` in the stochastic interpolant.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Values of :math:`\gamma(t)` at the given times.
        
        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def gamma_dot(self, t: torch.Tensor):
        """
        Time derivative :math:`\gamma'(t)` in the stochastic interpolant.
        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Values of :math:`\gamma'(t)` at the given times.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
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

        .. math::
        L_{\text{velocity}}(\theta) = \mathbb{E}\!\left[\,\lVert b\rVert^2 - 2\, b \cdot \dot I\,\right] \\
        L_{\text{denoiser}}(\theta) = \mathbb{E}\!\left[\,\lVert \eta\rVert^2 - 2\, \eta \cdot z\,\right] \\
        L(\theta) = \text{velocity\_weight}\,L_{\text{velocity}}(\theta) + \text{denoiser\_weight}\,L_{\text{denoiser}}(\theta)

        Args:
            t (torch.Tensor): Times in [0,1].
            x_0 (torch.Tensor): Samples from the base distribution rho_0.
            x_1 (torch.Tensor): Samples from the data distribution rho_0.
            z (torch.Tensor): Latent noise values :math:`z \sim \mathcal{N}(0, 1)`.
            b (torch.Tensor): Predicted velocity values for :math:`x_t`.
            eta (torch.Tensor): Predicted denoiser values for :math:`x_t`.

        Returns:
            dict[str, torch.Tensor]: A dictionary of loss values, ``loss``, ``loss_velocity``, and ``loss_denoiser``.
        """
        interpolant_dot = self.interpolate_derivative(t,x_0,x_1,z)
        loss_velocity  = torch.mean(torch.einsum('BND,BND->B', b, b    ) - 2*torch.einsum('BND, BND', b, interpolant_dot))
        loss_denoiser  = torch.mean(torch.einsum('BND,BND->B', eta, eta) - 2*torch.einsum('BND, BND', eta, z) if eta is not None else torch.zeros_like(loss_velocity))
        loss = self.velocity_weight*loss_velocity + self.denoiser_weight*loss_denoiser
        return {"loss": loss,
                "loss_velocity": loss_velocity,
                "loss_denoiser": loss_denoiser}

class LinearInterpolant(Interpolant):
    r"""
    Abstract class for defining an interpolant
    :math:`I(t, x_0, x_1) = \alpha(t) x_0 + \beta(t) x_1`
    in a stochastic setting between points :math:`x_0` and :math:`x_1` from distributions :math:`p_0` and :math:`p_1` at time :math:`t`.
    """

    def interpolate(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Interpolate between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times t.

        In order to possibly allow for periodic boundary conditions, x_1 is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant is then computed based on the unwrapped
        :math:`x_1` and the alpha and beta functions.

        Args:
            t (torch.Tensor): Times in [0,1].
            x_0 (torch.Tensor): Points from p_0.
            x_1 (torch.Tensor): Points from p_1.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Interpolated value and the latent noise.
        """
        z = torch.randn_like(x_0)
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        x_t = self.alpha(t) * x_0prime + self.beta(t) * x_1prime + z*self.gamma(t)
        return self.get_corrector().correct(x_t), z

    def interpolate_derivative(self, t: torch.Tensor, x_0: torch.Tensor, x_1: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Compute the derivative of the interpolant between points :math:`x_0` and :math:`x_1` from two distributions :math:`p_0` and :math:`p_1` at times
        :math:`t` with respect to time.

        In order to possibly allow for periodic boundary conditions, :math:`x_1` is first unwrapped based on the corrector of
        this interpolant. For the identity corrector, this unwrapping does nothing. For periodic boundary conditions,
        this unwrapping returns the closest image of :math:`x_1` to :math:`x_0`. The interpolant derivative is then computed based on
        the unwrapped :math:`x_1` and the alpha and beta functions.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.
            x_0 (torch.Tensor): Points from :math:`p_0`.
            x_1 (torch.Tensor): Points from: math:`p_1`.

        Returns:
            torch.Tensor: Derivative of the interpolant.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        x_0prime = self.get_corrector().correct(x_0)
        x_1prime = self.get_corrector().unwrap(x_0prime, x_1)
        return self.alpha_dot(t) * x_0prime + self.beta_dot(t) * x_1prime + self.gamma_dot(t) * z

    @abstractmethod
    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Alpha function :math:`\alpha(t)` in the linear interpolant.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Values of the alpha function at the given times.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def alpha_dot(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Time derivative of the alpha function :math:`\dot{\alpha}(t)` in the linear interpolant.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Derivatives of the alpha function at the given times.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError

    @abstractmethod
    def beta(self, t: torch.Tensor) -> torch.Tensor:
        r"""
        Beta function :math:`\beta(t)` in the linear interpolant.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Values of the beta function at the given times.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """

        raise NotImplementedError

    @abstractmethod
    def beta_dot(self, t: torch.Tensor):
        r"""
        Time derivative of the beta function :math:`\dot{\beta}(t)` in the linear interpolant.

        Args:
            t (torch.Tensor): Times in :math:`[0,1]`.

        Returns:
            torch.Tensor: Derivatives of the beta function at the given times.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError