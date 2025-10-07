dataset-checkpoint
==================

.. py:module:: dataset-checkpoint




Module Contents
---------------

.. py:class:: MolCrystalDataset(data_cfg, data_fname)

   Bases: :py:obj:`torch.utils.data.Dataset`


   An abstract class representing a :class:`Dataset`.

   All datasets that represent a map from keys to data samples should subclass
   it. All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
   data sample for a given key. Subclasses could also optionally overwrite
   :meth:`__len__`, which is expected to return the size of the dataset by many
   :class:`~torch.utils.data.Sampler` implementations and the default options
   of :class:`~torch.utils.data.DataLoader`. Subclasses could also
   optionally implement :meth:`__getitems__`, for speedup batched samples
   loading. This method accepts list of indices of samples of batch and returns
   list of samples.

   .. note::
     :class:`~torch.utils.data.DataLoader` by default constructs an index
     sampler that yields integral indices.  To make it work with a map-style
     dataset with non-integral indices/keys, a custom sampler must be provided.


   .. py:method:: lattice_to_lengths_and_angles(lattice_matrix)
      :staticmethod:


      Convert a lattice matrix to lengths and angles using ASE.

      Written by: Cheng Zeng

      :param lattice_matrix: numpy array of shape (3, 3) representing the lattice matrix.

      :returns: *lengths_and_angles* -- Tuple of (a, b, c, alpha, beta, gamma).



   .. py:method:: save_processed_to_xyz(processed_data, output_xyz_file)

      Save processed structures to an XYZ file.

      Written by: Cheng Zeng

      :param processed_data: List of processed features dictionaries for each structure.
      :param output_xyz_file: Path to save the output XYZ file.



   .. py:method:: __len__()


   .. py:method:: __getitem__(idx)

      Returns: dictionary with following keys:
          - rotmats_1: [M, 3, 3]
          - trans_1: [M, 3]
          - local_coords: [N, 3]
          - gt_coords: [N, 3]
          - bb_num_vec: [M,]
          - atom_types: [N,]
          - lattice_1: [1, 6]
          - cell_1: [1, 3, 3]



   .. py:method:: __add__(other)


