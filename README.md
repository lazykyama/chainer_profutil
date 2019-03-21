# chainer_profutil

This is an UNOFFICIAL Chainer related tool. This tool helps you to find forward, backward and update part from profiling details when you use NVIDIA Visual Profiler. As a result, you can improve your workload more efficiently.

## Simple example.

Adding 2 lines is all you need. First, import a function. Second, call it.

### Before.

```python
optimizer = chainer.optimizers.Adam(alpha=0.001)
optimizer.setup(model)
```

[![A profiling result without nvtx mark.](./docs/imgs/profiling_example_without_mark_small.png "A profiling result without nvtx mark.")](./docs/imgs/profiling_example_without_mark.png)

### After

```python
from chainer_profutil import create_marked_profile_optimizer

optimizer = create_marked_profile_optimizer(
    chainer.optimizers.Adam, sync=True)(alpha=0.001)
optimizer.setup(model)
```

[![A profiling result with nvtx mark.](./docs/imgs/profiling_example_with_mark_small.png "A profiling result with nvtx mark.")](./docs/imgs/profiling_example_with_mark.png)
