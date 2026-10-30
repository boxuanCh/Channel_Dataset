Channel Dataset
===============

Channel Dataset is an extensible framework for reproducible physical-layer
channel simulation. Version 0.1.0 focuses on wireless channels and provides
NumPy and PyTorch backends, injected random streams, validated channel batches,
standards-backed models, deterministic signal transforms, and persistence.

Start with :doc:`getting_started` for an end-to-end example, then use the
:doc:`api` guide for the complete data model.
Use the :doc:`reference/index` when you need the signature and full contract
of a particular class or function.

.. code-block:: python

   from channel_dataset.backends import create_random_stream
   from channel_dataset.channels.wireless import ChannelSamplingRequest
   from channel_dataset.channels.wireless.fading import RayleighBlockFading
   from channel_dataset.channels.wireless.fading import RayleighBlockFadingConfig
   from channel_dataset.core import ExecutionContext

   context = ExecutionContext(backend="numpy", precision="single", device="cpu")
   request = ChannelSamplingRequest(
       batch_size=2,
       num_time_samples=8,
       sampling_frequency_hz=15_000.0,
       execution=context,
       random_stream=create_random_stream("numpy", seed=7),
   )
   channel = RayleighBlockFading(RayleighBlockFadingConfig())
   cir = channel.realize(request)

.. toctree::
   :maxdepth: 2
   :caption: User guide

   getting_started
   api
   tutorial_notebooks
   architecture
   capabilities
   geometric_cdl
   ieee80211_models
   tr38811_models
   tr37885_models
   tgax_itu_3gpp_comparison
   validation

.. toctree::
   :maxdepth: 2
   :caption: API reference

   reference/index

.. toctree::
   :maxdepth: 1
   :caption: Development

   wireless_implementation_plan
   acoustic_channel_implementation_plan
   geometric_cdl_evolution_implementation_plan
   wifi_channel_implementation_plan
   tr38811_source_inventory
   tr37885_source_inventory
   v2x_channel_implementation_plan

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
