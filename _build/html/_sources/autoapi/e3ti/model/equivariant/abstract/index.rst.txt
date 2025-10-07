e3ti.model.equivariant.abstract
===============================

.. py:module:: e3ti.model.equivariant.abstract




Module Contents
---------------

.. py:class:: EquivariantE3TIModule(cfg)

   Bases: :py:obj:`e3ti.module.E3TIModule`


   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self) -> None:
               super().__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. note::
       As per the example above, an ``__init__()`` call to the parent class
       must be made before assignment on the child.

   :ivar training: Boolean represents whether this module is in training or
                   evaluation mode.
   :vartype training: bool

   Initialize internal Module state, shared by both nn.Module and ScriptModule.


   .. py:method:: test_equivariance(batch, tolerance_dict)

      Applies random rotations to the data and forward passes through the model.

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data
      :param tolerance_dict:
          Key value pairs corresponding to the tolerance of a commutator to pass the test.
      :type tolerance_dict: dict

      :return:
          A boolean condition on whether or not the model is equivariant
      :rtype: bool



   .. py:method:: commutator_error(out_before, out_after)

      Returns dictionary of key value pairs which maps a key to its commutator error between a batch out_before which was
      acted on by apply_group_action before .forward() and out_after which was acted on after.


      :param out_before:
          A torch batch of geometric data objects which was rotated before forward.
      :type batch: torch_geometric.data.Data
      :param out_after:
          A torch batch of geometric data objects which was rotated after forward.
      :type batch: torch_geometric.data.Data

      :return:
          A dictionary of mean absolute value of error values.
      :rtype: dict



   .. py:method:: random_group_action(batch)

      Samples a random group action for the system in question.

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          An object specifying the group action.
      :rtype: dict



   .. py:method:: apply_group_action(R, batch)

      "
      Applies a group action R to the batch. It does so by taking advantage of the get_irreps method.

      :param R:
          An object specifying the group action.
      :type R: Any
      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          A torch batch of geometric data objects which have been acted on by the group action R.
      :type batch: torch_geometric.data.Data



   .. py:attribute:: cfg


   .. py:attribute:: prior
      :value: None



   .. py:attribute:: embedder
      :value: None



   .. py:attribute:: model
      :value: None



   .. py:attribute:: interpolant
      :value: None



   .. py:method:: forward(batch)

      TODO: Finish return param typing here
      Implements a forward pass through the embedders and model.

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          A new batch object with modified keys containing velocity, score, denoised point etc.
      :rtype: torch_geometric.data.Data??



   .. py:method:: configure_optimizers()

      Parses configuration for the optimizer for lightning

      https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.configure_optimizers




   .. py:method:: training_step(batch)

      Implements a training step.

          1) corrupt batch appropriately using interpolant
          2) call forward
          3) compute loss

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data

      :return:
          A dictionary of loss values, loss, loss_velocity, and loss_denoiser
      :rtype: dict[str, torch.Tensor]



   .. py:method:: validation_step(batch)

      Implements a validation step.

          1) corrupt batch appropriately using interpolant
              1a) do so stratified on [0,1]
          2) call forward
          3) compute loss

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Datax

      :return:
          A dictionary of loss values, loss, loss_velocity, and loss_denoiser
      :rtype: dict[str, torch.Tensor]



   .. py:method:: predict_step(batch)
      :abstractmethod:


      Use the batch of data to perform experiments on the model based off of config

      1) parse experiments from config and instantiate experiment objects
      2) prepare model for experiment (disable dropout, training depedent layers, etc. )
      3) run experiment
      4) go back to 2

      :param batch:
          A torch batch of geometric data objects which come from a data loader.
      :type batch: torch_geometric.data.Data



   .. py:method:: summarize_cfg()

      Produces a print statement summarizing relevant contents within the configuration object.

      TODO: Add a experiment summarize call



   .. py:attribute:: __jit_unused_properties__
      :type:  list[str]
      :value: ['example_input_array', 'on_gpu', 'current_epoch', 'global_step', 'global_rank', 'local_rank',...



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_KEY
      :value: 'hyper_parameters'



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_NAME
      :value: 'hparams_name'



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_TYPE
      :value: 'hparams_type'



   .. py:method:: optimizers(use_pl_optimizer: Literal[True] = True) -> Union[pytorch_lightning.core.optimizer.LightningOptimizer, list[pytorch_lightning.core.optimizer.LightningOptimizer]]
                  optimizers(use_pl_optimizer: Literal[False]) -> Union[torch.optim.optimizer.Optimizer, list[torch.optim.optimizer.Optimizer]]
                  optimizers(use_pl_optimizer: bool) -> MODULE_OPTIMIZERS

      Returns the optimizer(s) that are being used during training. Useful for manual optimization.

      :param use_pl_optimizer: If ``True``, will wrap the optimizer(s) in a
                               :class:`~pytorch_lightning.core.optimizer.LightningOptimizer` for automatic handling of precision,
                               profiling, and counting of step calls for proper logging and checkpointing. It specifically wraps the
                               ``step`` method and custom optimizers that don't have this method are not supported.

      :returns: A single optimizer, or a list of optimizers in case multiple ones are present.



   .. py:method:: lr_schedulers()

      Returns the learning rate scheduler(s) that are being used during training. Useful for manual optimization.

      :returns: A single scheduler, or a list of schedulers in case multiple ones are present, or ``None`` if no
                schedulers were returned in :meth:`~pytorch_lightning.core.LightningModule.configure_optimizers`.



   .. py:property:: trainer
      :type: pytorch_lightning.Trainer



   .. py:property:: fabric
      :type: Optional[lightning_fabric.Fabric]



   .. py:property:: example_input_array
      :type: Optional[Union[torch.Tensor, tuple, dict]]


      The example input array is a specification of what the module can consume in the :meth:`forward` method. The
      return type is interpreted as follows:

      -   Single tensor: It is assumed the model takes a single argument, i.e.,
          ``model.forward(model.example_input_array)``
      -   Tuple: The input array should be interpreted as a sequence of positional arguments, i.e.,
          ``model.forward(*model.example_input_array)``
      -   Dict: The input array represents named keyword arguments, i.e.,
          ``model.forward(**model.example_input_array)``


   .. py:property:: current_epoch
      :type: int


      The current epoch in the ``Trainer``, or 0 if not attached.


   .. py:property:: global_step
      :type: int


      Total training batches seen across all epochs.

      If no Trainer is attached, this property is 0.


   .. py:property:: global_rank
      :type: int


      The index of the current process across all nodes and devices.


   .. py:property:: local_rank
      :type: int


      The index of the current process within a single node.


   .. py:property:: on_gpu
      :type: bool


      Returns ``True`` if this model is currently located on a GPU.

      Useful to set flags around the LightningModule for different CPU vs GPU behavior.


   .. py:property:: automatic_optimization
      :type: bool


      If set to ``False`` you are responsible for calling ``.backward()``, ``.step()``, ``.zero_grad()``.


   .. py:property:: strict_loading
      :type: bool


      Determines how Lightning loads this model using `.load_state_dict(..., strict=model.strict_loading)`.


   .. py:property:: logger
      :type: Optional[Union[pytorch_lightning.loggers.Logger, lightning_fabric.loggers.Logger]]


      Reference to the logger object in the Trainer.


   .. py:property:: loggers
      :type: Union[list[pytorch_lightning.loggers.Logger], list[lightning_fabric.loggers.Logger]]


      Reference to the list of loggers in the Trainer.


   .. py:property:: device_mesh
      :type: Optional[torch.distributed.device_mesh.DeviceMesh]


      Strategies like ``ModelParallelStrategy`` will create a device mesh that can be accessed in the
      :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` hook to parallelize the LightningModule.


   .. py:method:: print(*args, **kwargs)

      Prints only from process 0. Use this in any distributed mode to log only once.

      :param \*args: The thing to print. The same as for Python's built-in print function.
      :param \*\*kwargs: The same as for Python's built-in print function.

      Example::

          def forward(self, x):
              self.print(x, 'in forward')




   .. py:method:: log(name, value, prog_bar = False, logger = None, on_step = None, on_epoch = None, reduce_fx = 'mean', enable_graph = False, sync_dist = False, sync_dist_group = None, add_dataloader_idx = True, batch_size = None, metric_attribute = None, rank_zero_only = False)

      Log a key, value pair.

      Example::

          self.log('train_loss', loss)

      The default behavior per hook is documented here: :ref:`extensions/logging:Automatic Logging`.

      :param name: key to log. Must be identical across all processes if using DDP or any other distributed strategy.
      :param value: value to log. Can be a ``float``, ``Tensor``, or a ``Metric``.
      :param prog_bar: if ``True`` logs to the progress bar.
      :param logger: if ``True`` logs to the logger.
      :param on_step: if ``True`` logs at this step. The default value is determined by the hook.
                      See :ref:`extensions/logging:Automatic Logging` for details.
      :param on_epoch: if ``True`` logs epoch accumulated metrics. The default value is determined by the hook.
                       See :ref:`extensions/logging:Automatic Logging` for details.
      :param reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
      :param enable_graph: if ``True``, will not auto detach the graph.
      :param sync_dist: if ``True``, reduces the metric across devices. Use with care as this may lead to a significant
                        communication overhead.
      :param sync_dist_group: the DDP group to sync across.
      :param add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                                 the name (when using multiple dataloaders). If False, user needs to give unique names for
                                 each dataloader to not mix the values.
      :param batch_size: Current batch_size. This will be directly inferred from the loaded batch,
                         but for some data structures you might need to explicitly provide it.
      :param metric_attribute: To restore the metric state, Lightning requires the reference of the
                               :class:`torchmetrics.Metric` in your model. This is found automatically if it is a model attribute.
      :param rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
                             rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
                             (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
                             :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.



   .. py:method:: log_dict(dictionary, prog_bar = False, logger = None, on_step = None, on_epoch = None, reduce_fx = 'mean', enable_graph = False, sync_dist = False, sync_dist_group = None, add_dataloader_idx = True, batch_size = None, rank_zero_only = False)

      Log a dictionary of values at once.

      Example::

          values = {'loss': loss, 'acc': acc, ..., 'metric_n': metric_n}
          self.log_dict(values)

      :param dictionary: key value pairs.
                         Keys must be identical across all processes if using DDP or any other distributed strategy.
                         The values can be a ``float``, ``Tensor``, ``Metric``, or ``MetricCollection``.
      :param prog_bar: if ``True`` logs to the progress base.
      :param logger: if ``True`` logs to the logger.
      :param on_step: if ``True`` logs at this step.
                      ``None`` auto-logs for training_step but not validation/test_step.
                      The default value is determined by the hook.
                      See :ref:`extensions/logging:Automatic Logging` for details.
      :param on_epoch: if ``True`` logs epoch accumulated metrics.
                       ``None`` auto-logs for val/test step but not ``training_step``.
                       The default value is determined by the hook.
                       See :ref:`extensions/logging:Automatic Logging` for details.
      :param reduce_fx: reduction function over step values for end of epoch. :meth:`torch.mean` by default.
      :param enable_graph: if ``True``, will not auto-detach the graph
      :param sync_dist: if ``True``, reduces the metric across GPUs/TPUs. Use with care as this may lead to a significant
                        communication overhead.
      :param sync_dist_group: the ddp group to sync across.
      :param add_dataloader_idx: if ``True``, appends the index of the current dataloader to
                                 the name (when using multiple). If ``False``, user needs to give unique names for
                                 each dataloader to not mix values.
      :param batch_size: Current batch size. This will be directly inferred from the loaded batch,
                         but some data structures might need to explicitly provide it.
      :param rank_zero_only: Tells Lightning if you are calling ``self.log`` from every process (default) or only from
                             rank 0. If ``True``, you won't be able to use this metric as a monitor in callbacks
                             (e.g., early stopping). Warning: Improper use can lead to deadlocks! See
                             :ref:`Advanced Logging <visualize/logging_advanced:rank_zero_only>` for more details.



   .. py:method:: all_gather(data, group = None, sync_grads = False)

      Gather tensors or collections of tensors from multiple processes.

      This method needs to be called on all processes and the tensors need to have the same shape across all
      processes, otherwise your program will stall forever.

      :param data: int, float, tensor of shape (batch, ...), or a (possibly nested) collection thereof.
      :param group: the process group to gather results from. Defaults to all processes (world)
      :param sync_grads: flag that allows users to synchronize gradients for the all_gather operation

      :returns: A tensor of shape (world_size, batch, ...), or if the input was a collection
                the output will also be a collection with tensors of this shape. For the special case where
                world_size is 1, no additional dimension is added to the tensor(s).



   .. py:method:: test_step(*args, **kwargs)

      Operates on a single batch of data from the test set. In this step you'd normally generate examples or
      calculate anything of interest such as accuracy.

      :param batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
      :param batch_idx: The index of this batch.
      :param dataloader_idx: The index of the dataloader that produced this batch.
                             (only if multiple dataloaders used)

      :returns:

                - :class:`~torch.Tensor` - The loss tensor
                - ``dict`` - A dictionary. Can include any keys, but must include the key ``'loss'``.
                - ``None`` - Skip to the next batch.

      .. code-block:: python

          # if you have one test dataloader:
          def test_step(self, batch, batch_idx): ...


          # if you have multiple test dataloaders:
          def test_step(self, batch, batch_idx, dataloader_idx=0): ...

      Examples::

          # CASE 1: A single test dataset
          def test_step(self, batch, batch_idx):
              x, y = batch

              # implement your own
              out = self(x)
              loss = self.loss(out, y)

              # log 6 example images
              # or generated text... or whatever
              sample_imgs = x[:6]
              grid = torchvision.utils.make_grid(sample_imgs)
              self.logger.experiment.add_image('example_images', grid, 0)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              test_acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # log the outputs!
              self.log_dict({'test_loss': loss, 'test_acc': test_acc})

      If you pass in multiple test dataloaders, :meth:`test_step` will have an additional argument. We recommend
      setting the default value of 0 so that you can quickly switch between single and multiple dataloaders.

      .. code-block:: python

          # CASE 2: multiple test dataloaders
          def test_step(self, batch, batch_idx, dataloader_idx=0):
              # dataloader_idx tells you which dataset this is.
              x, y = batch

              # implement your own
              out = self(x)

              if dataloader_idx == 0:
                  loss = self.loss0(out, y)
              else:
                  loss = self.loss1(out, y)

              # calculate acc
              labels_hat = torch.argmax(out, dim=1)
              acc = torch.sum(y == labels_hat).item() / (len(y) * 1.0)

              # log the outputs separately for each dataloader
              self.log_dict({f"test_loss_{dataloader_idx}": loss, f"test_acc_{dataloader_idx}": acc})

      .. note:: If you don't need to test you don't need to implement this method.

      .. note::

         When the :meth:`test_step` is called, the model has been put in eval mode and
         PyTorch gradients have been disabled. At the end of the test epoch, the model goes back
         to training mode and gradients are enabled.



   .. py:method:: configure_callbacks()

      Configure model-specific callbacks. When the model gets attached, e.g., when ``.fit()`` or ``.test()`` gets
      called, the list or a callback returned here will be merged with the list of callbacks passed to the Trainer's
      ``callbacks`` argument. If a callback returned here has the same type as one or several callbacks already
      present in the Trainer's callbacks list, it will take priority and replace them. In addition, Lightning will
      make sure :class:`~pytorch_lightning.callbacks.model_checkpoint.ModelCheckpoint` callbacks run last.

      :returns: A callback or a list of callbacks which will extend the list of callbacks in the Trainer.

      Example::

          def configure_callbacks(self):
              early_stop = EarlyStopping(monitor="val_acc", mode="max")
              checkpoint = ModelCheckpoint(monitor="val_loss")
              return [early_stop, checkpoint]




   .. py:method:: manual_backward(loss, *args, **kwargs)

      Call this directly from your :meth:`training_step` when doing optimizations manually. By using this,
      Lightning can ensure that all the proper scaling gets applied when using mixed precision.

      See :ref:`manual optimization<common/optimization:Manual optimization>` for more examples.

      Example::

          def training_step(...):
              opt = self.optimizers()
              loss = ...
              opt.zero_grad()
              # automatically applies scaling, etc...
              self.manual_backward(loss)
              opt.step()

      :param loss: The tensor on which to compute gradients. Must have a graph attached.
      :param \*args: Additional positional arguments to be forwarded to :meth:`~torch.Tensor.backward`
      :param \*\*kwargs: Additional keyword arguments to be forwarded to :meth:`~torch.Tensor.backward`



   .. py:method:: backward(loss, *args, **kwargs)

      Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your own
      implementation if you need to.

      :param loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
                   holds the normalized value (scaled by 1 / accumulation steps).

      Example::

          def backward(self, loss):
              loss.backward()




   .. py:method:: toggle_optimizer(optimizer)

      Makes sure only the gradients of the current optimizer's parameters are calculated in the training step to
      prevent dangling gradients in multiple-optimizer setup.

      It works with :meth:`untoggle_optimizer` to make sure ``param_requires_grad_state`` is properly reset.

      :param optimizer: The optimizer to toggle.



   .. py:method:: untoggle_optimizer(optimizer)

      Resets the state of required gradients that were toggled with :meth:`toggle_optimizer`.

      :param optimizer: The optimizer to untoggle.



   .. py:method:: toggled_optimizer(optimizer)

      Makes sure only the gradients of the current optimizer's parameters are calculated in the training step to
      prevent dangling gradients in multiple-optimizer setup. Combines :meth:`toggle_optimizer` and
      :meth:`untoggle_optimizer` into context manager.

      :param optimizer: The optimizer to toggle.

      Example::

          def training_step(...):
              opt = self.optimizers()
              with self.toggled_optimizer(opt):
                  loss = ...
                  opt.zero_grad()
                  self.manual_backward(loss)
                  opt.step()




   .. py:method:: clip_gradients(optimizer, gradient_clip_val = None, gradient_clip_algorithm = None)

      Handles gradient clipping internally.

      .. note::

         - Do not override this method. If you want to customize gradient clipping, consider using
           :meth:`configure_gradient_clipping` method.
         - For manual optimization (``self.automatic_optimization = False``), if you want to use
           gradient clipping, consider calling
           ``self.clip_gradients(opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")``
           manually in the training step.

      :param optimizer: Current optimizer being used.
      :param gradient_clip_val: The value at which to clip gradients.
      :param gradient_clip_algorithm: The gradient clipping algorithm to use. Pass ``gradient_clip_algorithm="value"``
                                      to clip by value, and ``gradient_clip_algorithm="norm"`` to clip by norm.



   .. py:method:: configure_gradient_clipping(optimizer, gradient_clip_val = None, gradient_clip_algorithm = None)

      Perform gradient clipping for the optimizer parameters. Called before :meth:`optimizer_step`.

      :param optimizer: Current optimizer being used.
      :param gradient_clip_val: The value at which to clip gradients. By default, value passed in Trainer
                                will be available here.
      :param gradient_clip_algorithm: The gradient clipping algorithm to use. By default, value
                                      passed in Trainer will be available here.

      Example::

          def configure_gradient_clipping(self, optimizer, gradient_clip_val, gradient_clip_algorithm):
              # Implement your own custom logic to clip gradients
              # You can call `self.clip_gradients` with your settings:
              self.clip_gradients(
                  optimizer,
                  gradient_clip_val=gradient_clip_val,
                  gradient_clip_algorithm=gradient_clip_algorithm
              )




   .. py:method:: lr_scheduler_step(scheduler, metric)

      Override this method to adjust the default way the :class:`~pytorch_lightning.trainer.trainer.Trainer` calls
      each scheduler. By default, Lightning calls ``step()`` and as shown in the example for each scheduler based on
      its ``interval``.

      :param scheduler: Learning rate scheduler.
      :param metric: Value of the monitor used for schedulers like ``ReduceLROnPlateau``.

      Examples::

          # DEFAULT
          def lr_scheduler_step(self, scheduler, metric):
              if metric is None:
                  scheduler.step()
              else:
                  scheduler.step(metric)

          # Alternative way to update schedulers if it requires an epoch value
          def lr_scheduler_step(self, scheduler, metric):
              scheduler.step(epoch=self.current_epoch)




   .. py:method:: optimizer_step(epoch, batch_idx, optimizer, optimizer_closure = None)

      Override this method to adjust the default way the :class:`~pytorch_lightning.trainer.trainer.Trainer` calls
      the optimizer.

      By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example.
      This method (and ``zero_grad()``) won't be called during the accumulation phase when
      ``Trainer(accumulate_grad_batches != 1)``. Overriding this hook has no benefit with manual optimization.

      :param epoch: Current epoch
      :param batch_idx: Index of current batch
      :param optimizer: A PyTorch optimizer
      :param optimizer_closure: The optimizer closure. This closure must be executed as it includes the
                                calls to ``training_step()``, ``optimizer.zero_grad()``, and ``backward()``.

      Examples::

          def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
              # Add your custom logic to run directly before `optimizer.step()`

              optimizer.step(closure=optimizer_closure)

              # Add your custom logic to run directly after `optimizer.step()`




   .. py:method:: optimizer_zero_grad(epoch, batch_idx, optimizer)

      Override this method to change the default behaviour of ``optimizer.zero_grad()``.

      :param epoch: Current epoch
      :param batch_idx: Index of current batch
      :param optimizer: A PyTorch optimizer

      Examples::

          # DEFAULT
          def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
              optimizer.zero_grad()

          # Set gradients to `None` instead of zero to improve performance (not required on `torch>=2.0.0`).
          def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
              optimizer.zero_grad(set_to_none=True)

      See :meth:`torch.optim.Optimizer.zero_grad` for the explanation of the above example.




   .. py:method:: freeze()

      Freeze all params for inference.

      Example::

          model = MyLightningModule(...)
          model.freeze()




   .. py:method:: unfreeze()

      Unfreeze all parameters for training.

      .. code-block:: python

          model = MyLightningModule(...)
          model.unfreeze()




   .. py:method:: to_onnx(file_path = None, input_sample = None, **kwargs)

      Saves the model in ONNX format.

      :param file_path: The path of the file the onnx model should be saved to. Default: None (no file saved).
      :param input_sample: An input for tracing. Default: None (Use self.example_input_array)
      :param \*\*kwargs: Will be passed to torch.onnx.export function.

      Example::

          class SimpleModel(LightningModule):
              def __init__(self):
                  super().__init__()
                  self.l1 = torch.nn.Linear(in_features=64, out_features=4)

              def forward(self, x):
                  return torch.relu(self.l1(x.view(x.size(0), -1)

          model = SimpleModel()
          input_sample = torch.randn(1, 64)
          model.to_onnx("export.onnx", input_sample, export_params=True)




   .. py:method:: to_torchscript(file_path = None, method = 'script', example_inputs = None, **kwargs)

      By default compiles the whole model to a :class:`~torch.jit.ScriptModule`. If you want to use tracing,
      please provided the argument ``method='trace'`` and make sure that either the `example_inputs` argument is
      provided, or the model has :attr:`example_input_array` set. If you would like to customize the modules that are
      scripted you should override this method. In case you want to return multiple modules, we recommend using a
      dictionary.

      :param file_path: Path where to save the torchscript. Default: None (no file saved).
      :param method: Whether to use TorchScript's script or trace method. Default: 'script'
      :param example_inputs: An input to be used to do tracing when method is set to 'trace'.
                             Default: None (uses :attr:`example_input_array`)
      :param \*\*kwargs: Additional arguments that will be passed to the :func:`torch.jit.script` or
                         :func:`torch.jit.trace` function.

      .. note::

         - Requires the implementation of the
           :meth:`~pytorch_lightning.core.LightningModule.forward` method.
         - The exported script will be set to evaluation mode.
         - It is recommended that you install the latest supported version of PyTorch
           to use this feature without limitations. See also the :mod:`torch.jit`
           documentation for supported features.

      Example::

          class SimpleModel(LightningModule):
              def __init__(self):
                  super().__init__()
                  self.l1 = torch.nn.Linear(in_features=64, out_features=4)

              def forward(self, x):
                  return torch.relu(self.l1(x.view(x.size(0), -1)))

          model = SimpleModel()
          model.to_torchscript(file_path="model.pt")

          torch.jit.save(model.to_torchscript(
              file_path="model_trace.pt", method='trace', example_inputs=torch.randn(1, 64))
          )

      :returns: This LightningModule as a torchscript, regardless of whether `file_path` is
                defined or not.



   .. py:method:: load_from_checkpoint(checkpoint_path, map_location = None, hparams_file = None, strict = None, **kwargs)

      Primary way of loading a model from a checkpoint. When Lightning saves a checkpoint it stores the arguments
      passed to ``__init__``  in the checkpoint under ``"hyper_parameters"``.

      Any arguments specified through \*\*kwargs will override args stored in ``"hyper_parameters"``.

      :param checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
      :param map_location: If your checkpoint saved a GPU model and you now load on CPUs
                           or a different number of GPUs, use this to map to the new setup.
                           The behaviour is the same as in :func:`torch.load`.
      :param hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                           as in this example::

                               drop_prob: 0.2
                               dataloader:
                                   batch_size: 32

                           You most likely won't need this since Lightning will always save the hyperparameters
                           to the checkpoint.
                           However, if your checkpoint weights don't have the hyperparameters saved,
                           use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                           These will be converted into a :class:`~dict` and passed into your
                           :class:`LightningModule` for use.

                           If your model's ``hparams`` argument is :class:`~argparse.Namespace`
                           and ``.yaml`` file has hierarchical structure, you need to refactor your model to treat
                           ``hparams`` as :class:`~dict`.
      :param strict: Whether to strictly enforce that the keys in :attr:`checkpoint_path` match the keys
                     returned by this module's state dict. Defaults to ``True`` unless ``LightningModule.strict_loading`` is
                     set, in which case it defaults to the value of ``LightningModule.strict_loading``.
      :param \**kwargs: Any extra keyword args needed to init the model. Can also be used to override saved
                        hyperparameter values.

      :returns: :class:`LightningModule` instance with loaded weights and hyperparameters (if available).

      .. note::

         ``load_from_checkpoint`` is a **class** method. You should use your :class:`LightningModule`
         **class** to call it instead of the :class:`LightningModule` instance, or a
         ``TypeError`` will be raised.

      .. note::

         To ensure all layers can be loaded from the checkpoint, this function will call
         :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` directly after instantiating the
         model if this hook is overridden in your LightningModule. However, note that ``load_from_checkpoint`` does
         not support loading sharded checkpoints, and you may run out of memory if the model is too large. In this
         case, consider loading through the Trainer via ``.fit(ckpt_path=...)``.

      Example::

          # load weights without mapping ...
          model = MyLightningModule.load_from_checkpoint('path/to/checkpoint.ckpt')

          # or load weights mapping all weights from GPU 1 to GPU 0 ...
          map_location = {'cuda:1':'cuda:0'}
          model = MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              map_location=map_location
          )

          # or load weights and hyperparameters from separate files.
          model = MyLightningModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              hparams_file='/path/to/hparams_file.yaml'
          )

          # override some of the params with new values
          model = MyLightningModule.load_from_checkpoint(
              PATH,
              num_layers=128,
              pretrained_ckpt_path=NEW_PATH,
          )

          # predict
          pretrained_model.eval()
          pretrained_model.freeze()
          y_hat = pretrained_model(x)




   .. py:method:: __getstate__()


   .. py:property:: dtype
      :type: Union[str, torch.dtype]



   .. py:property:: device
      :type: torch.device



   .. py:method:: to(*args, **kwargs)

      See :meth:`torch.nn.Module.to`.



   .. py:method:: cuda(device = None)

      Moves all model parameters and buffers to the GPU. This also makes associated parameters and buffers
      different objects. So it should be called before constructing optimizer if the module will live on GPU while
      being optimized.

      :param device: If specified, all parameters will be copied to that device. If `None`, the current CUDA device
                     index will be used.

      :returns: *Module* -- self



   .. py:method:: cpu()

      See :meth:`torch.nn.Module.cpu`.



   .. py:method:: type(dst_type)

      See :meth:`torch.nn.Module.type`.



   .. py:method:: float()

      See :meth:`torch.nn.Module.float`.



   .. py:method:: double()

      See :meth:`torch.nn.Module.double`.



   .. py:method:: half()

      See :meth:`torch.nn.Module.half`.



   .. py:attribute:: dump_patches
      :type:  bool
      :value: False



   .. py:attribute:: training
      :type:  bool


   .. py:attribute:: call_super_init
      :type:  bool
      :value: False



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



   .. py:method:: load_state_dict(state_dict, strict = True, assign = False)

      Copy parameters and buffers from :attr:`state_dict` into this module and its descendants.

      If :attr:`strict` is ``True``, then
      the keys of :attr:`state_dict` must exactly match the keys returned
      by this module's :meth:`~torch.nn.Module.state_dict` function.

      .. warning::
          If :attr:`assign` is ``True`` the optimizer must be created after
          the call to :attr:`load_state_dict` unless
          :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

      :param state_dict: a dict containing parameters and
                         persistent buffers.
      :type state_dict: dict
      :param strict: whether to strictly enforce that the keys
                     in :attr:`state_dict` match the keys returned by this module's
                     :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
      :type strict: bool, optional
      :param assign: When ``False``, the properties of the tensors
                     in the current module are preserved while when ``True``, the
                     properties of the Tensors in the state dict are preserved. The only
                     exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`s
                     for which the value from the module is preserved.
                     Default: ``False``
      :type assign: bool, optional

      :returns: ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields --

                    * **missing_keys** is a list of str containing any keys that are expected
                        by this module but missing from the provided ``state_dict``.
                    * **unexpected_keys** is a list of str containing the keys that are not
                        expected by this module but present in the provided ``state_dict``.

      .. note::

         If a parameter or buffer is registered as ``None`` and its corresponding key
         exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
         ``RuntimeError``.



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



   .. py:method:: save_hyperparameters(*args, ignore = None, frame = None, logger = True)

      Save arguments to ``hparams`` attribute.

      :param args: single object of `dict`, `NameSpace` or `OmegaConf`
                   or string names or arguments from class ``__init__``
      :param ignore: an argument name or a list of argument names from
                     class ``__init__`` to be ignored
      :param frame: a frame object. Default is None
      :param logger: Whether to send the hyperparameters to the logger. Default: True

      Example::
          >>> from pytorch_lightning.core.mixins import HyperparametersMixin
          >>> class ManuallyArgsModel(HyperparametersMixin):
          ...     def __init__(self, arg1, arg2, arg3):
          ...         super().__init__()
          ...         # manually assign arguments
          ...         self.save_hyperparameters('arg1', 'arg3')
          ...     def forward(self, *args, **kwargs):
          ...         ...
          >>> model = ManuallyArgsModel(1, 'abc', 3.14)
          >>> model.hparams
          "arg1": 1
          "arg3": 3.14

          >>> from pytorch_lightning.core.mixins import HyperparametersMixin
          >>> class AutomaticArgsModel(HyperparametersMixin):
          ...     def __init__(self, arg1, arg2, arg3):
          ...         super().__init__()
          ...         # equivalent automatic
          ...         self.save_hyperparameters()
          ...     def forward(self, *args, **kwargs):
          ...         ...
          >>> model = AutomaticArgsModel(1, 'abc', 3.14)
          >>> model.hparams
          "arg1": 1
          "arg2": abc
          "arg3": 3.14

          >>> from pytorch_lightning.core.mixins import HyperparametersMixin
          >>> class SingleArgModel(HyperparametersMixin):
          ...     def __init__(self, params):
          ...         super().__init__()
          ...         # manually assign single argument
          ...         self.save_hyperparameters(params)
          ...     def forward(self, *args, **kwargs):
          ...         ...
          >>> model = SingleArgModel(Namespace(p1=1, p2='abc', p3=3.14))
          >>> model.hparams
          "p1": 1
          "p2": abc
          "p3": 3.14

          >>> from pytorch_lightning.core.mixins import HyperparametersMixin
          >>> class ManuallyArgsModel(HyperparametersMixin):
          ...     def __init__(self, arg1, arg2, arg3):
          ...         super().__init__()
          ...         # pass argument(s) to ignore as a string or in a list
          ...         self.save_hyperparameters(ignore='arg2')
          ...     def forward(self, *args, **kwargs):
          ...         ...
          >>> model = ManuallyArgsModel(1, 'abc', 3.14)
          >>> model.hparams
          "arg1": 1
          "arg3": 3.14




   .. py:property:: hparams
      :type: Union[lightning_fabric.utilities.data.AttributeDict, collections.abc.MutableMapping]


      The collection of hyperparameters saved with :meth:`save_hyperparameters`. It is mutable by the user. For
      the frozen set of initial hyperparameters, use :attr:`hparams_initial`.

      :returns: Mutable hyperparameters dictionary


   .. py:property:: hparams_initial
      :type: lightning_fabric.utilities.data.AttributeDict


      The collection of hyperparameters saved with :meth:`save_hyperparameters`. These contents are read-only.
      Manual updates to the saved hyperparameters can instead be performed through :attr:`hparams`.

      :returns: *AttributeDict* -- immutable initial hyperparameters


   .. py:method:: on_fit_start()

      Called at the very beginning of fit.

      If on DDP it is called on every process




   .. py:method:: on_fit_end()

      Called at the very end of fit.

      If on DDP it is called on every process




   .. py:method:: on_train_start()

      Called at the beginning of training after sanity check.



   .. py:method:: on_train_end()

      Called at the end of training before logger experiment is closed.



   .. py:method:: on_validation_start()

      Called at the beginning of validation.



   .. py:method:: on_validation_end()

      Called at the end of validation.



   .. py:method:: on_test_start()

      Called at the beginning of testing.



   .. py:method:: on_test_end()

      Called at the end of testing.



   .. py:method:: on_predict_start()

      Called at the beginning of predicting.



   .. py:method:: on_predict_end()

      Called at the end of predicting.



   .. py:method:: on_train_batch_start(batch, batch_idx)

      Called in the training loop before anything happens for that batch.

      If you return -1 here, you will skip training for the rest of the current epoch.

      :param batch: The batched data as it is returned by the training DataLoader.
      :param batch_idx: the index of the batch



   .. py:method:: on_train_batch_end(outputs, batch, batch_idx)

      Called in the training loop after the batch.

      :param outputs: The outputs of training_step(x)
      :param batch: The batched data as it is returned by the training DataLoader.
      :param batch_idx: the index of the batch

      .. note::

         The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
         loss returned from ``training_step``.



   .. py:method:: on_validation_batch_start(batch, batch_idx, dataloader_idx = 0)

      Called in the validation loop before anything happens for that batch.

      :param batch: The batched data as it is returned by the validation DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx = 0)

      Called in the validation loop after the batch.

      :param outputs: The outputs of validation_step(x)
      :param batch: The batched data as it is returned by the validation DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_test_batch_start(batch, batch_idx, dataloader_idx = 0)

      Called in the test loop before anything happens for that batch.

      :param batch: The batched data as it is returned by the test DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_test_batch_end(outputs, batch, batch_idx, dataloader_idx = 0)

      Called in the test loop after the batch.

      :param outputs: The outputs of test_step(x)
      :param batch: The batched data as it is returned by the test DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_predict_batch_start(batch, batch_idx, dataloader_idx = 0)

      Called in the predict loop before anything happens for that batch.

      :param batch: The batched data as it is returned by the test DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx = 0)

      Called in the predict loop after the batch.

      :param outputs: The outputs of predict_step(x)
      :param batch: The batched data as it is returned by the prediction DataLoader.
      :param batch_idx: the index of the batch
      :param dataloader_idx: the index of the dataloader



   .. py:method:: on_validation_model_zero_grad()

      Called by the training loop to release gradients before entering the validation loop.



   .. py:method:: on_validation_model_eval()

      Called when the validation loop starts.

      The validation loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
      to change the behavior. See also :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_validation_model_train`.




   .. py:method:: on_validation_model_train()

      Called when the validation loop ends.

      The validation loop by default restores the `training` mode of the LightningModule to what it was before
      starting validation. Override this hook to change the behavior. See also
      :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_validation_model_eval`.




   .. py:method:: on_test_model_eval()

      Called when the test loop starts.

      The test loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
      to change the behavior. See also :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_test_model_train`.




   .. py:method:: on_test_model_train()

      Called when the test loop ends.

      The test loop by default restores the `training` mode of the LightningModule to what it was before
      starting testing. Override this hook to change the behavior. See also
      :meth:`~pytorch_lightning.core.hooks.ModelHooks.on_test_model_eval`.




   .. py:method:: on_predict_model_eval()

      Called when the predict loop starts.

      The predict loop by default calls ``.eval()`` on the LightningModule before it starts. Override this hook
      to change the behavior.




   .. py:method:: on_train_epoch_start()

      Called in the training loop at the very beginning of the epoch.



   .. py:method:: on_train_epoch_end()

      Called in the training loop at the very end of the epoch.

      To access all batch outputs at the end of the epoch, you can cache step outputs as an attribute of the
      :class:`~pytorch_lightning.LightningModule` and access them in this hook:

      .. code-block:: python

          class MyLightningModule(L.LightningModule):
              def __init__(self):
                  super().__init__()
                  self.training_step_outputs = []

              def training_step(self):
                  loss = ...
                  self.training_step_outputs.append(loss)
                  return loss

              def on_train_epoch_end(self):
                  # do something with all training_step outputs, for example:
                  epoch_mean = torch.stack(self.training_step_outputs).mean()
                  self.log("training_epoch_mean", epoch_mean)
                  # free up the memory
                  self.training_step_outputs.clear()




   .. py:method:: on_validation_epoch_start()

      Called in the validation loop at the very beginning of the epoch.



   .. py:method:: on_validation_epoch_end()

      Called in the validation loop at the very end of the epoch.



   .. py:method:: on_test_epoch_start()

      Called in the test loop at the very beginning of the epoch.



   .. py:method:: on_test_epoch_end()

      Called in the test loop at the very end of the epoch.



   .. py:method:: on_predict_epoch_start()

      Called at the beginning of predicting.



   .. py:method:: on_predict_epoch_end()

      Called at the end of predicting.



   .. py:method:: on_before_zero_grad(optimizer)

      Called after ``training_step()`` and before ``optimizer.zero_grad()``.

      Called in the training loop after taking an optimizer step and before zeroing grads.
      Good place to inspect weight information with weights updated.

      This is where it is called::

          for optimizer in optimizers:
              out = training_step(...)

              model.on_before_zero_grad(optimizer) # < ---- called here
              optimizer.zero_grad()

              backward()

      :param optimizer: The optimizer for which grads should be zeroed.



   .. py:method:: on_before_backward(loss)

      Called before ``loss.backward()``.

      :param loss: Loss divided by number of batches for gradient accumulation and scaled if using AMP.



   .. py:method:: on_after_backward()

      Called after ``loss.backward()`` and before optimizers are stepped.

      .. note::

         If using native AMP, the gradients will not be unscaled at this point.
         Use the ``on_before_optimizer_step`` if you need the unscaled gradients.



   .. py:method:: on_before_optimizer_step(optimizer)

      Called before ``optimizer.step()``.

      If using gradient accumulation, the hook is called once the gradients have been accumulated.
      See: :paramref:`~pytorch_lightning.trainer.trainer.Trainer.accumulate_grad_batches`.

      If using AMP, the loss will be unscaled before calling this hook.
      See these `docs <https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients>`__
      for more information on the scaling of gradients.

      If clipping gradients, the gradients will not have been clipped yet.

      :param optimizer: Current optimizer being used.

      Example::

          def on_before_optimizer_step(self, optimizer):
              # example to inspect gradient information in tensorboard
              if self.trainer.global_step % 25 == 0:  # don't make the tf file huge
                  for k, v in self.named_parameters():
                      self.logger.experiment.add_histogram(
                          tag=k, values=v.grad, global_step=self.trainer.global_step
                      )




   .. py:method:: configure_sharded_model()

      Deprecated.

      Use :meth:`~pytorch_lightning.core.hooks.ModelHooks.configure_model` instead.




   .. py:method:: configure_model()

      Hook to create modules in a strategy and precision aware context.

      This is particularly useful for when using sharded strategies (FSDP and DeepSpeed), where we'd like to shard
      the model instantly to save memory and initialization time.
      For non-sharded strategies, you can choose to override this hook or to initialize your model under the
      :meth:`~pytorch_lightning.trainer.trainer.Trainer.init_module` context manager.

      This hook is called during each of fit/val/test/predict stages in the same process, so ensure that
      implementation of this hook is **idempotent**, i.e., after the first time the hook is called, subsequent calls
      to it should be a no-op.




   .. py:attribute:: prepare_data_per_node
      :type:  bool
      :value: True



   .. py:attribute:: allow_zero_length_dataloader_with_multiple_devices
      :type:  bool
      :value: False



   .. py:method:: prepare_data()

      Use this to download and prepare data. Downloading and saving data with multiple processes (distributed
      settings) will result in corrupted data. Lightning ensures this method is called only within a single process,
      so you can safely add your downloading logic within.

      .. warning:: DO NOT set state to the model (use ``setup`` instead)
          since this is NOT called on every device

      Example::

          def prepare_data(self):
              # good
              download_data()
              tokenize()
              etc()

              # bad
              self.split = data_split
              self.some_state = some_other_state()

      In a distributed environment, ``prepare_data`` can be called in two ways
      (using :ref:`prepare_data_per_node<common/lightning_module:prepare_data_per_node>`)

      1. Once per node. This is the default and is only called on LOCAL_RANK=0.
      2. Once in total. Only called on GLOBAL_RANK=0.

      Example::

          # DEFAULT
          # called once per node on LOCAL_RANK=0 of that node
          class LitDataModule(LightningDataModule):
              def __init__(self):
                  super().__init__()
                  self.prepare_data_per_node = True


          # call on GLOBAL_RANK=0 (great for shared file systems)
          class LitDataModule(LightningDataModule):
              def __init__(self):
                  super().__init__()
                  self.prepare_data_per_node = False

      This is called before requesting the dataloaders:

      .. code-block:: python

          model.prepare_data()
          initialize_distributed()
          model.setup(stage)
          model.train_dataloader()
          model.val_dataloader()
          model.test_dataloader()
          model.predict_dataloader()




   .. py:method:: setup(stage)

      Called at the beginning of fit (train + validate), validate, test, or predict. This is a good hook when you
      need to build models dynamically or adjust something about them. This hook is called on every process when
      using DDP.

      :param stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``

      Example::

          class LitModel(...):
              def __init__(self):
                  self.l1 = None

              def prepare_data(self):
                  download_data()
                  tokenize()

                  # don't do this
                  self.something = else

              def setup(self, stage):
                  data = load_data(...)
                  self.l1 = nn.Linear(28, data.num_classes)




   .. py:method:: teardown(stage)

      Called at the end of fit (train + validate), validate, test, or predict.

      :param stage: either ``'fit'``, ``'validate'``, ``'test'``, or ``'predict'``



   .. py:method:: train_dataloader()

      An iterable or collection of iterables specifying training samples.

      For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

      The dataloader you return will not be reloaded unless you set
      :paramref:`~pytorch_lightning.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
      a positive integer.

      For data processing use the following pattern:

          - download in :meth:`prepare_data`
          - process and split in :meth:`setup`

      However, the above are only necessary for distributed processing.

      .. warning:: do not assign state in prepare_data

      - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
      - :meth:`prepare_data`
      - :meth:`setup`

      .. note::

         Lightning tries to add the correct sampler for distributed and arbitrary hardware.
         There is no need to set it yourself.



   .. py:method:: test_dataloader()

      An iterable or collection of iterables specifying test samples.

      For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

      For data processing use the following pattern:

          - download in :meth:`prepare_data`
          - process and split in :meth:`setup`

      However, the above are only necessary for distributed processing.

      .. warning:: do not assign state in prepare_data


      - :meth:`~pytorch_lightning.trainer.trainer.Trainer.test`
      - :meth:`prepare_data`
      - :meth:`setup`

      .. note::

         Lightning tries to add the correct sampler for distributed and arbitrary hardware.
         There is no need to set it yourself.

      .. note::

         If you don't need a test dataset and a :meth:`test_step`, you don't need to implement
         this method.



   .. py:method:: val_dataloader()

      An iterable or collection of iterables specifying validation samples.

      For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

      The dataloader you return will not be reloaded unless you set
      :paramref:`~pytorch_lightning.trainer.trainer.Trainer.reload_dataloaders_every_n_epochs` to
      a positive integer.

      It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

      - :meth:`~pytorch_lightning.trainer.trainer.Trainer.fit`
      - :meth:`~pytorch_lightning.trainer.trainer.Trainer.validate`
      - :meth:`prepare_data`
      - :meth:`setup`

      .. note::

         Lightning tries to add the correct sampler for distributed and arbitrary hardware
         There is no need to set it yourself.

      .. note::

         If you don't need a validation dataset and a :meth:`validation_step`, you don't need to
         implement this method.



   .. py:method:: predict_dataloader()

      An iterable or collection of iterables specifying prediction samples.

      For more information about multiple dataloaders, see this :ref:`section <multiple-dataloaders>`.

      It's recommended that all data downloads and preparation happen in :meth:`prepare_data`.

      - :meth:`~pytorch_lightning.trainer.trainer.Trainer.predict`
      - :meth:`prepare_data`
      - :meth:`setup`

      .. note::

         Lightning tries to add the correct sampler for distributed and arbitrary hardware
         There is no need to set it yourself.

      :returns: A :class:`torch.utils.data.DataLoader` or a sequence of them specifying prediction samples.



   .. py:method:: transfer_batch_to_device(batch, device, dataloader_idx)

      Override this hook if your :class:`~torch.utils.data.DataLoader` returns tensors wrapped in a custom data
      structure.

      The data types listed below (and any arbitrary nesting of them) are supported out of the box:

      - :class:`torch.Tensor` or anything that implements `.to(...)`
      - :class:`list`
      - :class:`dict`
      - :class:`tuple`

      For anything else, you need to define how the data is moved to the target device (CPU, GPU, TPU, ...).

      .. note::

         This hook should only transfer the data and not modify it, nor should it move the data to
         any other device than the one passed in as argument (unless you know what you are doing).
         To check the current state of execution of this hook you can use
         ``self.trainer.training/testing/validating/predicting`` so that you can
         add different logic as per your requirement.

      :param batch: A batch of data that needs to be transferred to a new device.
      :param device: The target device as defined in PyTorch.
      :param dataloader_idx: The index of the dataloader to which the batch belongs.

      :returns: A reference to the data on the new device.

      Example::

          def transfer_batch_to_device(self, batch, device, dataloader_idx):
              if isinstance(batch, CustomBatch):
                  # move all tensors in your custom data structure to the device
                  batch.samples = batch.samples.to(device)
                  batch.targets = batch.targets.to(device)
              elif dataloader_idx == 0:
                  # skip device transfer for the first dataloader or anything you wish
                  pass
              else:
                  batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
              return batch

      .. seealso::

         - :meth:`move_data_to_device`
         - :meth:`apply_to_collection`



   .. py:method:: on_before_batch_transfer(batch, dataloader_idx)

      Override to alter or apply batch augmentations to your batch before it is transferred to the device.

      .. note::

         To check the current state of execution of this hook you can use
         ``self.trainer.training/testing/validating/predicting`` so that you can
         add different logic as per your requirement.

      :param batch: A batch of data that needs to be altered or augmented.
      :param dataloader_idx: The index of the dataloader to which the batch belongs.

      :returns: A batch of data

      Example::

          def on_before_batch_transfer(self, batch, dataloader_idx):
              batch['x'] = transforms(batch['x'])
              return batch

      .. seealso::

         - :meth:`on_after_batch_transfer`
         - :meth:`transfer_batch_to_device`



   .. py:method:: on_after_batch_transfer(batch, dataloader_idx)

      Override to alter or apply batch augmentations to your batch after it is transferred to the device.

      .. note::

         To check the current state of execution of this hook you can use
         ``self.trainer.training/testing/validating/predicting`` so that you can
         add different logic as per your requirement.

      :param batch: A batch of data that needs to be altered or augmented.
      :param dataloader_idx: The index of the dataloader to which the batch belongs.

      :returns: A batch of data

      Example::

          def on_after_batch_transfer(self, batch, dataloader_idx):
              batch['x'] = gpu_transforms(batch['x'])
              return batch

      .. seealso::

         - :meth:`on_before_batch_transfer`
         - :meth:`transfer_batch_to_device`



   .. py:method:: on_load_checkpoint(checkpoint)

      Called by Lightning to restore your model. If you saved something with :meth:`on_save_checkpoint` this is
      your chance to restore this.

      :param checkpoint: Loaded checkpoint

      Example::

          def on_load_checkpoint(self, checkpoint):
              # 99% of the time you don't need to implement this method
              self.something_cool_i_want_to_save = checkpoint['something_cool_i_want_to_save']

      .. note::

         Lightning auto-restores global step, epoch, and train state including amp scaling.
         There is no need for you to restore anything regarding training.



   .. py:method:: on_save_checkpoint(checkpoint)

      Called by Lightning when saving a checkpoint to give you a chance to store anything else you might want to
      save.

      :param checkpoint: The full checkpoint dictionary before it gets dumped to a file.
                         Implementations of this hook can insert additional data into this dictionary.

      Example::

          def on_save_checkpoint(self, checkpoint):
              # 99% of use cases you don't need to implement this method
              checkpoint['something_cool_i_want_to_save'] = my_cool_pickable_object

      .. note::

         Lightning saves all aspects of training (epoch, global step, etc...)
         including amp scaling.
         There is no need for you to store anything about training.



