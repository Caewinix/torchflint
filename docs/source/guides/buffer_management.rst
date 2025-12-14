Buffer Management
=================

TorchFlint simplifies state management by allowing you to register buffers via simple assignment, bypassing the verbose ``register_buffer`` syntax.

Example: Using Buffers
----------------------

.. code-block:: python

    import torch
    import torchflint

    class MyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Registers a persistent buffer named 'buffer'
            # No need for self.register_buffer("buffer", ...)
            self.buffer = torchflint.buffer(torch.zeros(10), persistent=True)

    model = MyModule()