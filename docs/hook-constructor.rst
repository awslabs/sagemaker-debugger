Hook from Python constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the Hook

.. autoclass:: smdebug.core.hook.BaseHook
  :members:
  :undoc-members:
  :show-inheritance:
  :inherited-members:


.. code:: python

   hook = HookClass(
       out_dir,
       export_tensorboard = False,
       tensorboard_dir = None,
       dry_run = False,
       reduction_config = None,
       save_config = None,
       include_regex = None,
       include_collections = None,
       save_all = False,
       include_workers="one"
   )

**Parameters:**

  -  ``out_dir`` (str): Path where to save tensors and metadata. This is a
     required argument. Please ensure that the ‘out_dir’ does not exist.
  -  ``export_tensorboard`` (bool): Whether to export TensorBoard
     summaries (distributions and histograms for tensors saved, and scalar
     summaries for scalars saved). Defaults to ``False``. Note that when
     running on SageMaker this parameter will be ignored. You will need to
     use the TensorBoardOutputConfig section in API to enable TensorBoard
     summaries. Refer `SageMaker page <sagemaker.md>`__ for an example.
  -  ``tensorboard_dir`` (str): Path where to save TensorBoard artifacts.
     If this is not passed and ``export_tensorboard`` is True, then
     TensorBoard artifacts are saved in ``out_dir/tensorboard`` . Note
     that when running on SageMaker this parameter will be ignored. You
     will need to use the TensorBoardOutputConfig section in API to enable
     TensorBoard summaries. Refer `SageMaker page <sagemaker.md>`__ for an
     example.
  -  ``dry_run`` (bool): If true, don’t write any files
  -  ``reduction_config``: (`ReductionConfig <#reductionconfig>`__ object)
     Specifies the reductions to be applied as default for tensors saved.
     A collection can have its own ``ReductionConfig`` object which
     overrides this for the tensors which belong to that collection.
  -  ``save_config``: (`SaveConfig <#saveconfig>`__ object) Specifies when
     to save tensors. A collection can have its own ``SaveConfig`` object
     which overrides this for the tensors which belong to that collection.
  -  ``include_regex`` (list[str]): list of regex patterns which specify
     the tensors to save. Tensors whose names match these patterns will be
     saved
  -  ``include_collections`` (list[str]): List of which collections to
     save specified by name
  -  ``save_all`` (bool): Saves all tensors and collections. Increases the
     amount of disk space used, and can reduce the performance of the
     training job significantly, depending on the size of the model.
  -  ``include_workers`` (str): Used for distributed training. It can take
     the values ``one`` or ``all``. ``one`` means only the tensors from
     one chosen worker will be saved. This is the default behavior.
     ``all`` means tensors from all workers will be saved.
