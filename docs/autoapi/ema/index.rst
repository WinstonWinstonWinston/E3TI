ema
===

.. py:module:: ema




Module Contents
---------------

.. py:class:: ExponentialMovingAverage(parameters, decay, use_num_updates = True)

   Bases: :py:obj:`torch.nn.Module`


   Maintains (exponential) moving average of a set of parameters.

   :param parameters: Iterable of `torch.nn.Parameter` (typically from
                      `model.parameters()`).
                      Note that EMA is computed on *all* provided parameters,
                      regardless of whether or not they have `requires_grad = True`;
                      this allows a single EMA object to be consistantly used even
                      if which parameters are trainable changes step to step.

                      If you want to some parameters in the EMA, do not pass them
                      to the object in the first place. For example:

                          ExponentialMovingAverage(
                              parameters=[p for p in model.parameters() if p.requires_grad],
                              decay=0.9
                          )

                      will ignore parameters that do not require grad.
   :param decay: The exponential decay.
   :param use_num_updates: Whether to use number of updates when computing
                           averages.

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:attribute:: decay


   .. py:attribute:: num_updates
      :value: 0



   .. py:attribute:: shadow_params


   .. py:attribute:: collected_params
      :value: None



   .. py:method:: update(parameters = None)

      Update currently maintained parameters.

      Call this every time the parameters are updated, such as the result of
      the `optimizer.step()` call.

      :param parameters: Iterable of `torch.nn.Parameter`; usually the same set of
                         parameters used to initialize this object. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.



   .. py:method:: copy_to(parameters = None)

      Copy current averaged parameters into given collection of parameters.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         updated with the stored moving averages. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.



   .. py:method:: store(parameters = None)

      Save the current parameters for restoring later.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         temporarily stored. If `None`, the parameters of with which this
                         `ExponentialMovingAverage` was initialized will be used.



   .. py:method:: restore(parameters = None)

      Restore the parameters stored with the `store` method.
      Useful to validate the model with EMA parameters without affecting the
      original optimization process. Store the parameters before the
      `copy_to` method. After validation (or model saving), use this to
      restore the former parameters.

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         updated with the stored parameters. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.



   .. py:method:: average_parameters(parameters = None)

      Context manager for validation/inference with averaged parameters.

      Equivalent to:

          ema.store()
          ema.copy_to()
          try:
              ...
          finally:
              ema.restore()

      :param parameters: Iterable of `torch.nn.Parameter`; the parameters to be
                         updated with the stored parameters. If `None`, the
                         parameters with which this `ExponentialMovingAverage` was
                         initialized will be used.



   .. py:method:: to(device=None, dtype=None)

      Move internal buffers of the ExponentialMovingAverage to `device`.

      :param device: like `device` argument to `torch.Tensor.to`



   .. py:method:: load_state_dict(state_dict)

      Loads the ExponentialMovingAverage state.

      :param state_dict: EMA state. Should be an object returned
                         from a call to :meth:`state_dict`.
      :type state_dict: dict



   .. py:attribute:: dump_patches
      :type:  bool
      :value: False



   .. py:attribute:: training
      :type:  bool


   .. py:attribute:: call_super_init
      :type:  bool
      :value: False



   .. py:attribute:: forward
      :type:  Callable[Ellipsis, Any]


   .. py:method:: register_buffer(name, tensor, persistent = True)

      Add a buffer to the module.

      This is typically used to register a buffer that should not to be
      considered a model parameter. For example, BatchNorm's ``running_mean``
      is not a parameter, but is part of the module's state. Buffers, by
      default, are persistent and will be saved alongside parameters. This
      behavior can be changed by setting :attr:`persistent` to ``False``. The
      only difference between a persistent buffer and a non-persistent buffer
      is that the latter will not be a part of this module's
      :attr:`state_dict`.

      Buffers can be accessed as attributes using given names.

      :param name: name of the buffer. The buffer can be accessed
                   from this module using the given name
      :type name: str
      :param tensor: buffer to be registered. If ``None``, then operations
                     that run on buffers, such as :attr:`cuda`, are ignored. If ``None``,
                     the buffer is **not** included in the module's :attr:`state_dict`.
      :type tensor: Tensor or None
      :param persistent: whether the buffer is part of this module's
                         :attr:`state_dict`.
      :type persistent: bool

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> self.register_buffer('running_mean', torch.zeros(num_features))




   .. py:method:: register_parameter(name, param)

      Add a parameter to the module.

      The parameter can be accessed as an attribute using given name.

      :param name: name of the parameter. The parameter can be accessed
                   from this module using the given name
      :type name: str
      :param param: parameter to be added to the module. If
                    ``None``, then operations that run on parameters, such as :attr:`cuda`,
                    are ignored. If ``None``, the parameter is **not** included in the
                    module's :attr:`state_dict`.
      :type param: Parameter or None



   .. py:method:: add_module(name, module)

      Add a child module to the current module.

      The module can be accessed as an attribute using the given name.

      :param name: name of the child module. The child module can be
                   accessed from this module using the given name
      :type name: str
      :param module: child module to be added to the module.
      :type module: Module



   .. py:method:: register_module(name, module)

      Alias for :func:`add_module`.



   .. py:method:: get_submodule(target)

      Return the submodule given by ``target`` if it exists, otherwise throw an error.

      For example, let's say you have an ``nn.Module`` ``A`` that
      looks like this:

      .. code-block:: text

          A(
              (net_b): Module(
                  (net_c): Module(
                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                  )
                  (linear): Linear(in_features=100, out_features=200, bias=True)
              )
          )

      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
      submodule ``net_b``, which itself has two submodules ``net_c``
      and ``linear``. ``net_c`` then has a submodule ``conv``.)

      To check whether or not we have the ``linear`` submodule, we
      would call ``get_submodule("net_b.linear")``. To check whether
      we have the ``conv`` submodule, we would call
      ``get_submodule("net_b.net_c.conv")``.

      The runtime of ``get_submodule`` is bounded by the degree
      of module nesting in ``target``. A query against
      ``named_modules`` achieves the same result, but it is O(N) in
      the number of transitive modules. So, for a simple check to see
      if some submodule exists, ``get_submodule`` should always be
      used.

      :param target: The fully-qualified string name of the submodule
                     to look for. (See above example for how to specify a
                     fully-qualified string.)

      :returns: *torch.nn.Module* -- The submodule referenced by ``target``

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not an
          ``nn.Module``



   .. py:method:: set_submodule(target, module)

      Set the submodule given by ``target`` if it exists, otherwise throw an error.

      For example, let's say you have an ``nn.Module`` ``A`` that
      looks like this:

      .. code-block:: text

          A(
              (net_b): Module(
                  (net_c): Module(
                      (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
                  )
                  (linear): Linear(in_features=100, out_features=200, bias=True)
              )
          )

      (The diagram shows an ``nn.Module`` ``A``. ``A`` has a nested
      submodule ``net_b``, which itself has two submodules ``net_c``
      and ``linear``. ``net_c`` then has a submodule ``conv``.)

      To overide the ``Conv2d`` with a new submodule ``Linear``, you
      would call
      ``set_submodule("net_b.net_c.conv", nn.Linear(33, 16))``.

      :param target: The fully-qualified string name of the submodule
                     to look for. (See above example for how to specify a
                     fully-qualified string.)
      :param module: The module to set the submodule to.

      :raises ValueError: If the target string is empty
      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not an
          ``nn.Module``



   .. py:method:: get_parameter(target)

      Return the parameter given by ``target`` if it exists, otherwise throw an error.

      See the docstring for ``get_submodule`` for a more detailed
      explanation of this method's functionality as well as how to
      correctly specify ``target``.

      :param target: The fully-qualified string name of the Parameter
                     to look for. (See ``get_submodule`` for how to specify a
                     fully-qualified string.)

      :returns: *torch.nn.Parameter* -- The Parameter referenced by ``target``

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not an
          ``nn.Parameter``



   .. py:method:: get_buffer(target)

      Return the buffer given by ``target`` if it exists, otherwise throw an error.

      See the docstring for ``get_submodule`` for a more detailed
      explanation of this method's functionality as well as how to
      correctly specify ``target``.

      :param target: The fully-qualified string name of the buffer
                     to look for. (See ``get_submodule`` for how to specify a
                     fully-qualified string.)

      :returns: *torch.Tensor* -- The buffer referenced by ``target``

      :raises AttributeError: If the target string references an invalid
          path or resolves to something that is not a
          buffer



   .. py:method:: get_extra_state()

      Return any extra state to include in the module's state_dict.

      Implement this and a corresponding :func:`set_extra_state` for your module
      if you need to store extra state. This function is called when building the
      module's `state_dict()`.

      Note that extra state should be picklable to ensure working serialization
      of the state_dict. We only provide provide backwards compatibility guarantees
      for serializing Tensors; other objects may break backwards compatibility if
      their serialized pickled form changes.

      :returns: *object* -- Any extra state to store in the module's state_dict



   .. py:method:: set_extra_state(state)

      Set extra state contained in the loaded `state_dict`.

      This function is called from :func:`load_state_dict` to handle any extra state
      found within the `state_dict`. Implement this function and a corresponding
      :func:`get_extra_state` for your module if you need to store extra state within its
      `state_dict`.

      :param state: Extra state from the `state_dict`
      :type state: dict



   .. py:method:: apply(fn)

      Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

      Typical use includes initializing the parameters of a model
      (see also :ref:`nn-init-doc`).

      :param fn: function to be applied to each submodule
      :type fn: :class:`Module` -> None

      :returns: *Module* -- self

      Example::

          >>> @torch.no_grad()
          >>> def init_weights(m):
          >>>     print(m)
          >>>     if type(m) == nn.Linear:
          >>>         m.weight.fill_(1.0)
          >>>         print(m.weight)
          >>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
          >>> net.apply(init_weights)
          Linear(in_features=2, out_features=2, bias=True)
          Parameter containing:
          tensor([[1., 1.],
                  [1., 1.]], requires_grad=True)
          Linear(in_features=2, out_features=2, bias=True)
          Parameter containing:
          tensor([[1., 1.],
                  [1., 1.]], requires_grad=True)
          Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          )




   .. py:method:: cuda(device = None)

      Move all model parameters and buffers to the GPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on GPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: *Module* -- self



   .. py:method:: ipu(device = None)

      Move all model parameters and buffers to the IPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on IPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: *Module* -- self



   .. py:method:: xpu(device = None)

      Move all model parameters and buffers to the XPU.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on XPU while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: *Module* -- self



   .. py:method:: mtia(device = None)

      Move all model parameters and buffers to the MTIA.

      This also makes associated parameters and buffers different objects. So
      it should be called before constructing optimizer if the module will
      live on MTIA while being optimized.

      .. note::
          This method modifies the module in-place.

      :param device: if specified, all parameters will be
                     copied to that device
      :type device: int, optional

      :returns: *Module* -- self



   .. py:method:: cpu()

      Move all model parameters and buffers to the CPU.

      .. note::
          This method modifies the module in-place.

      :returns: *Module* -- self



   .. py:method:: type(dst_type)

      Casts all parameters and buffers to :attr:`dst_type`.

      .. note::
          This method modifies the module in-place.

      :param dst_type: the desired type
      :type dst_type: type or string

      :returns: *Module* -- self



   .. py:method:: float()

      Casts all floating point parameters and buffers to ``float`` datatype.

      .. note::
          This method modifies the module in-place.

      :returns: *Module* -- self



   .. py:method:: double()

      Casts all floating point parameters and buffers to ``double`` datatype.

      .. note::
          This method modifies the module in-place.

      :returns: *Module* -- self



   .. py:method:: half()

      Casts all floating point parameters and buffers to ``half`` datatype.

      .. note::
          This method modifies the module in-place.

      :returns: *Module* -- self



   .. py:method:: bfloat16()

      Casts all floating point parameters and buffers to ``bfloat16`` datatype.

      .. note::
          This method modifies the module in-place.

      :returns: *Module* -- self



   .. py:method:: to_empty(*, device, recurse = True)

      Move the parameters and buffers to the specified device without copying storage.

      :param device: The desired device of the parameters
                     and buffers in this module.
      :type device: :class:`torch.device`
      :param recurse: Whether parameters and buffers of submodules should
                      be recursively moved to the specified device.
      :type recurse: bool

      :returns: *Module* -- self



   .. py:method:: register_full_backward_pre_hook(hook, prepend = False)

      Register a backward pre-hook on the module.

      The hook will be called every time the gradients for the module are computed.
      The hook should have the following signature::

          hook(module, grad_output) -> tuple[Tensor] or None

      The :attr:`grad_output` is a tuple. The hook should
      not modify its arguments, but it can optionally return a new gradient with
      respect to the output that will be used in place of :attr:`grad_output` in
      subsequent computations. Entries in :attr:`grad_output` will be ``None`` for
      all non-Tensor arguments.

      For technical reasons, when this hook is applied to a Module, its forward function will
      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
      of each Tensor returned by the Module's forward function.

      .. warning ::
          Modifying inputs inplace is not allowed when using backward hooks and
          will raise an error.

      :param hook: The user-defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``backward_pre`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``backward_pre`` hooks
                      on this :class:`torch.nn.modules.Module`. Note that global
                      ``backward_pre`` hooks registered with
                      :func:`register_module_full_backward_pre_hook` will fire before
                      all hooks registered by this method.
      :type prepend: bool

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:method:: register_backward_hook(hook)

      Register a backward hook on the module.

      This function is deprecated in favor of :meth:`~torch.nn.Module.register_full_backward_hook` and
      the behavior of this function will change in future versions.

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:method:: register_full_backward_hook(hook, prepend = False)

      Register a backward hook on the module.

      The hook will be called every time the gradients with respect to a module
      are computed, i.e. the hook will execute if and only if the gradients with
      respect to module outputs are computed. The hook should have the following
      signature::

          hook(module, grad_input, grad_output) -> tuple(Tensor) or None

      The :attr:`grad_input` and :attr:`grad_output` are tuples that contain the gradients
      with respect to the inputs and outputs respectively. The hook should
      not modify its arguments, but it can optionally return a new gradient with
      respect to the input that will be used in place of :attr:`grad_input` in
      subsequent computations. :attr:`grad_input` will only correspond to the inputs given
      as positional arguments and all kwarg arguments are ignored. Entries
      in :attr:`grad_input` and :attr:`grad_output` will be ``None`` for all non-Tensor
      arguments.

      For technical reasons, when this hook is applied to a Module, its forward function will
      receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
      of each Tensor returned by the Module's forward function.

      .. warning ::
          Modifying inputs or outputs inplace is not allowed when using backward hooks and
          will raise an error.

      :param hook: The user-defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``backward`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``backward`` hooks on
                      this :class:`torch.nn.modules.Module`. Note that global
                      ``backward`` hooks registered with
                      :func:`register_module_full_backward_hook` will fire before
                      all hooks registered by this method.
      :type prepend: bool

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:method:: register_forward_pre_hook(hook, *, prepend = False, with_kwargs = False)

      Register a forward pre-hook on the module.

      The hook will be called every time before :func:`forward` is invoked.


      If ``with_kwargs`` is false or not specified, the input contains only
      the positional arguments given to the module. Keyword arguments won't be
      passed to the hooks and only to the ``forward``. The hook can modify the
      input. User can either return a tuple or a single modified value in the
      hook. We will wrap the value into a tuple if a single value is returned
      (unless that value is already a tuple). The hook should have the
      following signature::

          hook(module, args) -> None or modified input

      If ``with_kwargs`` is true, the forward pre-hook will be passed the
      kwargs given to the forward function. And if the hook modifies the
      input, both the args and kwargs should be returned. The hook should have
      the following signature::

          hook(module, args, kwargs) -> None or a tuple of modified input and kwargs

      :param hook: The user defined hook to be registered.
      :type hook: Callable
      :param prepend: If true, the provided ``hook`` will be fired before
                      all existing ``forward_pre`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``forward_pre`` hooks
                      on this :class:`torch.nn.modules.Module`. Note that global
                      ``forward_pre`` hooks registered with
                      :func:`register_module_forward_pre_hook` will fire before all
                      hooks registered by this method.
                      Default: ``False``
      :type prepend: bool
      :param with_kwargs: If true, the ``hook`` will be passed the kwargs
                          given to the forward function.
                          Default: ``False``
      :type with_kwargs: bool

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:method:: register_forward_hook(hook, *, prepend = False, with_kwargs = False, always_call = False)

      Register a forward hook on the module.

      The hook will be called every time after :func:`forward` has computed an output.

      If ``with_kwargs`` is ``False`` or not specified, the input contains only
      the positional arguments given to the module. Keyword arguments won't be
      passed to the hooks and only to the ``forward``. The hook can modify the
      output. It can modify the input inplace but it will not have effect on
      forward since this is called after :func:`forward` is called. The hook
      should have the following signature::

          hook(module, args, output) -> None or modified output

      If ``with_kwargs`` is ``True``, the forward hook will be passed the
      ``kwargs`` given to the forward function and be expected to return the
      output possibly modified. The hook should have the following signature::

          hook(module, args, kwargs, output) -> None or modified output

      :param hook: The user defined hook to be registered.
      :type hook: Callable
      :param prepend: If ``True``, the provided ``hook`` will be fired
                      before all existing ``forward`` hooks on this
                      :class:`torch.nn.modules.Module`. Otherwise, the provided
                      ``hook`` will be fired after all existing ``forward`` hooks on
                      this :class:`torch.nn.modules.Module`. Note that global
                      ``forward`` hooks registered with
                      :func:`register_module_forward_hook` will fire before all hooks
                      registered by this method.
                      Default: ``False``
      :type prepend: bool
      :param with_kwargs: If ``True``, the ``hook`` will be passed the
                          kwargs given to the forward function.
                          Default: ``False``
      :type with_kwargs: bool
      :param always_call: If ``True`` the ``hook`` will be run regardless of
                          whether an exception is raised while calling the Module.
                          Default: ``False``
      :type always_call: bool

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:attribute:: __call__
      :type:  Callable[Ellipsis, Any]


   .. py:method:: __getstate__()


   .. py:method:: __setstate__(state)


   .. py:method:: __getattr__(name)


   .. py:method:: __setattr__(name, value)


   .. py:method:: __delattr__(name)


   .. py:method:: register_state_dict_post_hook(hook)

      Register a post-hook for the :meth:`~torch.nn.Module.state_dict` method.

      It should have the following signature::
          hook(module, state_dict, prefix, local_metadata) -> None

      The registered hooks can modify the ``state_dict`` inplace.



   .. py:method:: register_state_dict_pre_hook(hook)

      Register a pre-hook for the :meth:`~torch.nn.Module.state_dict` method.

      It should have the following signature::
          hook(module, prefix, keep_vars) -> None

      The registered hooks can be used to perform pre-processing before the ``state_dict``
      call is made.



   .. py:attribute:: T_destination


   .. py:method:: state_dict(*, destination: T_destination, prefix: str = ..., keep_vars: bool = ...) -> T_destination
                  state_dict(*, prefix: str = ..., keep_vars: bool = ...) -> Dict[str, Any]

      Return a dictionary containing references to the whole state of the module.

      Both parameters and persistent buffers (e.g. running averages) are
      included. Keys are corresponding parameter and buffer names.
      Parameters and buffers set to ``None`` are not included.

      .. note::
          The returned object is a shallow copy. It contains references
          to the module's parameters and buffers.

      .. warning::
          Currently ``state_dict()`` also accepts positional arguments for
          ``destination``, ``prefix`` and ``keep_vars`` in order. However,
          this is being deprecated and keyword arguments will be enforced in
          future releases.

      .. warning::
          Please avoid the use of argument ``destination`` as it is not
          designed for end-users.

      :param destination: If provided, the state of module will
                          be updated into the dict and the same object is returned.
                          Otherwise, an ``OrderedDict`` will be created and returned.
                          Default: ``None``.
      :type destination: dict, optional
      :param prefix: a prefix added to parameter and buffer
                     names to compose the keys in state_dict. Default: ``''``.
      :type prefix: str, optional
      :param keep_vars: by default the :class:`~torch.Tensor` s
                        returned in the state dict are detached from autograd. If it's
                        set to ``True``, detaching will not be performed.
                        Default: ``False``.
      :type keep_vars: bool, optional

      :returns: *dict* --     a dictionary containing a whole state of the module

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> module.state_dict().keys()
          ['bias', 'weight']




   .. py:method:: register_load_state_dict_pre_hook(hook)

      Register a pre-hook to be run before module's :meth:`~nn.Module.load_state_dict` is called.

      It should have the following signature::
          hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None  # noqa: B950

      :param hook: Callable hook that will be invoked before
                   loading the state dict.
      :type hook: Callable



   .. py:method:: register_load_state_dict_post_hook(hook)

      Register a post-hook to be run after module's :meth:`~nn.Module.load_state_dict` is called.

      It should have the following signature::
          hook(module, incompatible_keys) -> None

      The ``module`` argument is the current module that this hook is registered
      on, and the ``incompatible_keys`` argument is a ``NamedTuple`` consisting
      of attributes ``missing_keys`` and ``unexpected_keys``. ``missing_keys``
      is a ``list`` of ``str`` containing the missing keys and
      ``unexpected_keys`` is a ``list`` of ``str`` containing the unexpected keys.

      The given incompatible_keys can be modified inplace if needed.

      Note that the checks performed when calling :func:`load_state_dict` with
      ``strict=True`` are affected by modifications the hook makes to
      ``missing_keys`` or ``unexpected_keys``, as expected. Additions to either
      set of keys will result in an error being thrown when ``strict=True``, and
      clearing out both missing and unexpected keys will avoid an error.

      :returns: :class:`torch.utils.hooks.RemovableHandle` --     a handle that can be used to remove the added hook by calling
                    ``handle.remove()``



   .. py:method:: parameters(recurse = True)

      Return an iterator over module parameters.

      This is typically passed to an optimizer.

      :param recurse: if True, then yields parameters of this module
                      and all submodules. Otherwise, yields only parameters that
                      are direct members of this module.
      :type recurse: bool

      :Yields: *Parameter* -- module parameter

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for param in model.parameters():
          >>>     print(type(param), param.size())
          <class 'torch.Tensor'> (20L,)
          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)




   .. py:method:: named_parameters(prefix = '', recurse = True, remove_duplicate = True)

      Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

      :param prefix: prefix to prepend to all parameter names.
      :type prefix: str
      :param recurse: if True, then yields parameters of this module
                      and all submodules. Otherwise, yields only parameters that
                      are direct members of this module.
      :type recurse: bool
      :param remove_duplicate: whether to remove the duplicated
                               parameters in the result. Defaults to True.
      :type remove_duplicate: bool, optional

      :Yields: *(str, Parameter)* -- Tuple containing the name and parameter

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, param in self.named_parameters():
          >>>     if name in ['bias']:
          >>>         print(param.size())




   .. py:method:: buffers(recurse = True)

      Return an iterator over module buffers.

      :param recurse: if True, then yields buffers of this module
                      and all submodules. Otherwise, yields only buffers that
                      are direct members of this module.
      :type recurse: bool

      :Yields: *torch.Tensor* -- module buffer

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for buf in model.buffers():
          >>>     print(type(buf), buf.size())
          <class 'torch.Tensor'> (20L,)
          <class 'torch.Tensor'> (20L, 1L, 5L, 5L)




   .. py:method:: named_buffers(prefix = '', recurse = True, remove_duplicate = True)

      Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

      :param prefix: prefix to prepend to all buffer names.
      :type prefix: str
      :param recurse: if True, then yields buffers of this module
                      and all submodules. Otherwise, yields only buffers that
                      are direct members of this module. Defaults to True.
      :type recurse: bool, optional
      :param remove_duplicate: whether to remove the duplicated buffers in the result. Defaults to True.
      :type remove_duplicate: bool, optional

      :Yields: *(str, torch.Tensor)* -- Tuple containing the name and buffer

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, buf in self.named_buffers():
          >>>     if name in ['running_var']:
          >>>         print(buf.size())




   .. py:method:: children()

      Return an iterator over immediate children modules.

      :Yields: *Module* -- a child module



   .. py:method:: named_children()

      Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

      :Yields: *(str, Module)* -- Tuple containing a name and child module

      Example::

          >>> # xdoctest: +SKIP("undefined vars")
          >>> for name, module in model.named_children():
          >>>     if name in ['conv4', 'conv5']:
          >>>         print(module)




   .. py:method:: modules()

      Return an iterator over all modules in the network.

      :Yields: *Module* -- a module in the network

      .. note::

         Duplicate modules are returned only once. In the following
         example, ``l`` will be returned only once.

      Example::

          >>> l = nn.Linear(2, 2)
          >>> net = nn.Sequential(l, l)
          >>> for idx, m in enumerate(net.modules()):
          ...     print(idx, '->', m)

          0 -> Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          )
          1 -> Linear(in_features=2, out_features=2, bias=True)




   .. py:method:: named_modules(memo = None, prefix = '', remove_duplicate = True)

      Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

      :param memo: a memo to store the set of modules already added to the result
      :param prefix: a prefix that will be added to the name of the module
      :param remove_duplicate: whether to remove the duplicated module instances in the result
                               or not

      :Yields: *(str, Module)* -- Tuple of name and module

      .. note::

         Duplicate modules are returned only once. In the following
         example, ``l`` will be returned only once.

      Example::

          >>> l = nn.Linear(2, 2)
          >>> net = nn.Sequential(l, l)
          >>> for idx, m in enumerate(net.named_modules()):
          ...     print(idx, '->', m)

          0 -> ('', Sequential(
            (0): Linear(in_features=2, out_features=2, bias=True)
            (1): Linear(in_features=2, out_features=2, bias=True)
          ))
          1 -> ('0', Linear(in_features=2, out_features=2, bias=True))




   .. py:method:: train(mode = True)

      Set the module in training mode.

      This has any effect only on certain modules. See documentations of
      particular modules for details of their behaviors in training/evaluation
      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
      etc.

      :param mode: whether to set training mode (``True``) or evaluation
                   mode (``False``). Default: ``True``.
      :type mode: bool

      :returns: *Module* -- self



   .. py:method:: eval()

      Set the module in evaluation mode.

      This has any effect only on certain modules. See documentations of
      particular modules for details of their behaviors in training/evaluation
      mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
      etc.

      This is equivalent with :meth:`self.train(False) <torch.nn.Module.train>`.

      See :ref:`locally-disable-grad-doc` for a comparison between
      `.eval()` and several similar mechanisms that may be confused with it.

      :returns: *Module* -- self



   .. py:method:: requires_grad_(requires_grad = True)

      Change if autograd should record operations on parameters in this module.

      This method sets the parameters' :attr:`requires_grad` attributes
      in-place.

      This method is helpful for freezing part of the module for finetuning
      or training parts of a model individually (e.g., GAN training).

      See :ref:`locally-disable-grad-doc` for a comparison between
      `.requires_grad_()` and several similar mechanisms that may be confused with it.

      :param requires_grad: whether autograd should record operations on
                            parameters in this module. Default: ``True``.
      :type requires_grad: bool

      :returns: *Module* -- self



   .. py:method:: zero_grad(set_to_none = True)

      Reset gradients of all model parameters.

      See similar function under :class:`torch.optim.Optimizer` for more context.

      :param set_to_none: instead of setting to zero, set the grads to None.
                          See :meth:`torch.optim.Optimizer.zero_grad` for details.
      :type set_to_none: bool



   .. py:method:: share_memory()

      See :meth:`torch.Tensor.share_memory_`.



   .. py:method:: extra_repr()

      Set the extra representation of the module.

      To print customized extra information, you should re-implement
      this method in your own modules. Both single-line and multi-line
      strings are acceptable.



   .. py:method:: __repr__()


   .. py:method:: __dir__()


   .. py:method:: compile(*args, **kwargs)

      Compile this Module's forward using :func:`torch.compile`.

      This Module's `__call__` method is compiled and all arguments are passed as-is
      to :func:`torch.compile`.

      See :func:`torch.compile` for details on the arguments for this function.



