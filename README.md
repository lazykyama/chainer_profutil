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
    chainer.optimizers.Adam(alpha=0.001), sync=True)
optimizer.setup(model)
```

[![A profiling result with nvtx mark.](./docs/imgs/profiling_example_with_mark_small.png "A profiling result with nvtx mark.")](./docs/imgs/profiling_example_with_mark.png)

## Example for ChainerMN.

When you use ChainerMN's `create_multi_node_optimizer()`, you need to give an instance returned from `create_multi_node_optimizer()` to `create_marked_profile_optimizer()` as follows.

```python
optimizer = create_marked_profile_optimizer(
    chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(lr=0.01, momentum=0.9),
        comm),
    sync=False)
optimizer.setup(model)
```
