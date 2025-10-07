e3ti.interpolant.corrector
==========================

.. py:module:: e3ti.interpolant.corrector




Module Contents
---------------

.. py:class:: IdentityCorrector

   Bases: :py:obj:`e3ti.interpolant.abstract.Corrector`


   Corrector that does nothing.

   Construct identity corrector.


   .. py:method:: correct(x)

      Correct the input x.

      :param x:
          Input to correct.
      :type x: torch.Tensor

      :return:
          Corrected input.
      :rtype: torch.Tensor



   .. py:method:: unwrap(x_0, x_1)

      Correct the input x_1 based on the reference input x_0.

      This method just returns x_1.

      :param x_0:
          Reference input.
      :type x_0: torch.Tensor
      :param x_1:
          Input to correct.
      :type x_1: torch.Tensor

      :return:
          Unwrapped x_1 value.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()
      :abstractmethod:


      Prints details about the configuration defining the corrector.

      :returns: None

      :raises NotImplementedError: Subclasses must implement this method.



   .. py:attribute:: __slots__
      :value: ()



.. py:class:: PeriodicBoundaryConditionsCorrector(min_value, max_value)

   Bases: :py:obj:`e3ti.interpolant.abstract.Corrector`


   Corrector function that wraps back coordinates to the interval [min, max] with periodic boundary conditions.

   :param min_value:
       Minimum value of the interval.
   :type min_value: float
   :param max_value:
       Maximum value of the interval.
   :type max_value: float

   :raises ValueError:
       If the minimum value is greater than the maximum value.

   Construct corrector function.


   .. py:method:: correct(x)

      Correct the input x.

      :param x:
          Input to correct.
      :type x: torch.Tensor

      :return:
          Corrected input.
      :rtype: torch.Tensor



   .. py:method:: unwrap(x_0, x_1)

      Correct the input x_1 based on the reference input x_0.

      This method returns the image of x_1 closest to x_0 in periodic boundary conditions.

      :param x_0:
          Reference input.
      :type x_0: torch.Tensor
      :param x_1:
          Input to correct.
      :type x_1: torch.Tensor

      :return:
          Unwrapped x_1 value.
      :rtype: torch.Tensor



   .. py:method:: summarize_cfg()
      :abstractmethod:


      Prints details about the configuration defining the corrector.

      :returns: None

      :raises NotImplementedError: Subclasses must implement this method.



   .. py:attribute:: __slots__
      :value: ()



