ADP_dataset-checkpoint
======================

.. py:module:: ADP_dataset-checkpoint




Module Contents
---------------

.. py:class:: ADPDataset(data_cfg)

   Bases: :py:obj:`e3ti.data.dataset.E3TIDataset`


   TODO: Modify this to store the e3nn irreps of the object as attributes


   .. py:attribute:: pdb_file
      :value: 'alanine-dipeptide-nowater.pdb'



   .. py:attribute:: parm_file
      :value: 'AA.prmtop'



   .. py:method:: summarize_cfg()
      :abstractmethod:


      TODO: DO this ??Prints details about the configuration defining the corrector



   .. py:method:: __getitem__(idx)

      Returns: torch_geometric Data object with following keys:
          - x: [N, 3]
          - x_0: [N, 3] (only if noneq)
          - t: [N] (only if noneq)
          - charge: [N]
          - mass: [N]
          - sigma: [N]
          - epsilon: [N]



   .. py:attribute:: split


   .. py:attribute:: total_frames_train


   .. py:attribute:: total_frames_test


   .. py:attribute:: total_frames_valid


   .. py:attribute:: lag


   .. py:attribute:: normalize


   .. py:attribute:: node_features


   .. py:attribute:: augment_rotations


   .. py:method:: lag_whitening_stats(x, t_max, unbiased = True, eps = 1e-08)

      x      : [T, *S] time-major tensor
      t_max  : largest lag to compute (clipped to T-1)
      unbiased: use sample std if True (Torch's default behavior)
      eps    : small additive to std for numerical safety (e.g., 1e-8)

      Returns:
      MU  : [t_max, *S]  where MU[t-1]  = mean over i of (x_{i+t} - x_i) at lag t
      STD : [t_max, *S]  where STD[t-1] = std  over i of (x_{i+t} - x_i) at lag t



   .. py:method:: __len__()


   .. py:method:: __add__(other)


