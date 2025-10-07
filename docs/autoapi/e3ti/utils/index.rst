e3ti.utils
==========

.. py:module:: e3ti.utils




Module Contents
---------------

.. py:function:: batch2loss(batch, stratified=False)

.. py:function:: batch2interp(batch)

.. py:function:: flatten_dict(raw_dict)

   Flattens a nested dict.


.. py:function:: set_seed(seed)

.. py:function:: channels_arr_to_string(query_channels)

.. py:function:: parse_activation(spec)

   Parse an activation spec string and return an nn.Module.

   Examples
   --------
   >>> parse_activation('relu')
   >>> parse_activation('leaky_relu:0.2')
   >>> parse_activation('SiLU')


.. py:function:: repeat_interleave(repeats)

.. py:function:: periodic_radius_graph(pos, batch, rcut, cell_lengths)

   pos:           (N,3)  atom positions
   batch:         (N,)   graph indices
   rcut:          float  cutoff distance
   cell_lengths:  (B,3)  orthorhombic box lengths for each graph
   :returns: *edge_index* -- (2,E)  [src, dst] pairs (may include multiple p‐image edges)
             edge_vec:    (E,3)  the corresponding displacement vectors


.. py:function:: combine_features(pairs, *, dim = -1, tidy = False)

   Concatenate feature blocks and return (tensor, irreps).

   If `tidy=True` the result is canonical (`regroup()`); the function
   then permutes the channels so data and metadata agree.


