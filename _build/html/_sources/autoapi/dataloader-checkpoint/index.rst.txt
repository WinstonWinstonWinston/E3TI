dataloader-checkpoint
=====================

.. py:module:: dataloader-checkpoint




Module Contents
---------------

.. py:class:: MolCrystalDatamodule(*, loader_cfg, train_dataset, valid_dataset, test_dataset)

   Bases: :py:obj:`pytorch_lightning.LightningDataModule`


   A DataModule standardizes the training, val, test splits, data preparation and transforms. The main advantage is
   consistent data splits, data preparation and transforms across models.

   Example::

       import lightning.pytorch as L
       import torch.utils.data as data
       from pytorch_lightning.demos.boring_classes import RandomDataset

       class MyDataModule(L.LightningDataModule):
           def prepare_data(self):
               # download, IO, etc. Useful with shared filesystems
               # only called on 1 GPU/TPU in distributed
               ...

           def setup(self, stage):
               # make assignments here (val/train/test split)
               # called on every process in DDP
               dataset = RandomDataset(1, 100)
               self.train, self.val, self.test = data.random_split(
                   dataset, [80, 10, 10], generator=torch.Generator().manual_seed(42)
               )

           def train_dataloader(self):
               return data.DataLoader(self.train)

           def val_dataloader(self):
               return data.DataLoader(self.val)

           def test_dataloader(self):
               return data.DataLoader(self.test)

           def on_exception(self, exception):
               # clean up state after the trainer faced an exception
               ...

           def teardown(self):
               # clean up state after the trainer stops, delete files...
               # called on every process in DDP
               ...


   .. attribute:: prepare_data_per_node

      If True, each LOCAL_RANK=0 will call prepare data.
      Otherwise only NODE_RANK=0, LOCAL_RANK=0 will prepare data.

   .. attribute:: allow_zero_length_dataloader_with_multiple_devices

      If True, dataloader with zero length within local rank is allowed.
      Default value is False.


   .. py:method:: train_dataloader(shuffle=True)

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



   .. py:attribute:: name
      :type:  Optional[str]
      :value: None



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_KEY
      :value: 'datamodule_hyper_parameters'



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_NAME
      :value: 'datamodule_hparams_name'



   .. py:attribute:: CHECKPOINT_HYPER_PARAMS_TYPE
      :value: 'datamodule_hparams_type'



   .. py:attribute:: trainer
      :type:  Optional[pytorch_lightning.Trainer]
      :value: None



   .. py:method:: from_datasets(train_dataset = None, val_dataset = None, test_dataset = None, predict_dataset = None, batch_size = 1, num_workers = 0, **datamodule_kwargs)
      :classmethod:


      Create an instance from torch.utils.data.Dataset.

      :param train_dataset: Optional dataset or iterable of datasets to be used for train_dataloader()
      :param val_dataset: Optional dataset or iterable of datasets to be used for val_dataloader()
      :param test_dataset: Optional dataset or iterable of datasets to be used for test_dataloader()
      :param predict_dataset: Optional dataset or iterable of datasets to be used for predict_dataloader()
      :param batch_size: Batch size to use for each dataloader. Default is 1. This parameter gets forwarded to the
                         ``__init__`` if the datamodule has such a name defined in its signature.
      :param num_workers: Number of subprocesses to use for data loading. 0 means that the
                          data will be loaded in the main process. Number of CPUs available. This parameter gets forwarded to the
                          ``__init__`` if the datamodule has such a name defined in its signature.
      :param \*\*datamodule_kwargs: Additional parameters that get passed down to the datamodule's ``__init__``.



   .. py:method:: state_dict()

      Called when saving a checkpoint, implement to generate and save datamodule state.

      :returns: A dictionary containing datamodule state.



   .. py:method:: load_state_dict(state_dict)

      Called when loading a checkpoint, implement to reload datamodule state given datamodule state_dict.

      :param state_dict: the datamodule state returned by ``state_dict``.



   .. py:method:: on_exception(exception)

      Called when the trainer execution is interrupted by an exception.



   .. py:method:: load_from_checkpoint(checkpoint_path, map_location = None, hparams_file = None, **kwargs)

      Primary way of loading a datamodule from a checkpoint. When Lightning saves a checkpoint it stores the
      arguments passed to ``__init__``  in the checkpoint under ``"datamodule_hyper_parameters"``.

      Any arguments specified through \*\*kwargs will override args stored in ``"datamodule_hyper_parameters"``.

      :param checkpoint_path: Path to checkpoint. This can also be a URL, or file-like object
      :param map_location: If your checkpoint saved a GPU model and you now load on CPUs
                           or a different number of GPUs, use this to map to the new setup.
                           The behaviour is the same as in :func:`torch.load`.
      :param hparams_file: Optional path to a ``.yaml`` or ``.csv`` file with hierarchical structure
                           as in this example::

                               dataloader:
                                   batch_size: 32

                           You most likely won't need this since Lightning will always save the hyperparameters
                           to the checkpoint.
                           However, if your checkpoint weights don't have the hyperparameters saved,
                           use this method to pass in a ``.yaml`` file with the hparams you'd like to use.
                           These will be converted into a :class:`~dict` and passed into your
                           :class:`LightningDataModule` for use.

                           If your datamodule's ``hparams`` argument is :class:`~argparse.Namespace`
                           and ``.yaml`` file has hierarchical structure, you need to refactor your datamodule to treat
                           ``hparams`` as :class:`~dict`.
      :param \**kwargs: Any extra keyword args needed to init the datamodule. Can also be used to override saved
                        hyperparameter values.

      :returns: :class:`LightningDataModule` instance with loaded weights and hyperparameters (if available).

      .. note::

         ``load_from_checkpoint`` is a **class** method. You must use your :class:`LightningDataModule`
         **class** to call it instead of the :class:`LightningDataModule` instance, or a
         ``TypeError`` will be raised.

      Example::

          # load weights without mapping ...
          datamodule = MyLightningDataModule.load_from_checkpoint('path/to/checkpoint.ckpt')

          # or load weights and hyperparameters from separate files.
          datamodule = MyLightningDataModule.load_from_checkpoint(
              'path/to/checkpoint.ckpt',
              hparams_file='/path/to/hparams_file.yaml'
          )

          # override some of the params with new values
          datamodule = MyLightningDataModule.load_from_checkpoint(
              PATH,
              batch_size=32,
              num_workers=10,
          )




   .. py:method:: __str__()

      Return a string representation of the datasets that are set up.

      :returns: A string representation of the datasets that are setup.



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



   .. py:attribute:: __jit_unused_properties__
      :type:  list[str]
      :value: ['hparams', 'hparams_initial']



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


