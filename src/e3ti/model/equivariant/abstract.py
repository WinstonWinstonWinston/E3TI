from abc import abstractmethod
# import torch

from e3ti.model.abstract import E3TIModel

class EquivariantE3TIModel(E3TIModel):

    @abstractmethod
    def test_equivariance(batch, tol, rtol) -> bool:
        """
        Applies random rotations to the data and forward passes through the model.

        :param batch:
            A torch batch of geometric data objects which come from a data loader. 
        :type batch: torch_geometric.data.Data

        :return:
            A boolean condition on whether or not the model is equivariant
        :rtype: bool
        """
        raise NotImplementedError
