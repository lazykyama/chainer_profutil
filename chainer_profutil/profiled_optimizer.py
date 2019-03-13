

from chainer.function_hooks import CUDAProfileHook

from cupy import cuda
from cupy import prof
from cupy.cuda import runtime


def _try_to_sync_if_needed(self):
    if self._sync:
        runtime.deviceSynchronize()

class UpdateProfileMarkPreHook(object):
    name = 'preupdate'
    call_for_each_param = False
    timing = 'pre'

    def __init__(self, sync=True, argb_color=0xff00ff00):
        self._sync = sync
        self._argb_color = argb_color

    def __call__(self, rule):
        _try_to_sync_if_needed(self)
        cuda.nvtx.RangePushC('update', self._argb_color)

class UpdateProfileMarkPostHook(object):
    name = 'postupdate'
    call_for_each_param = False
    timing = 'post'

    def __init__(self, sync=True):
        self._sync = sync

    def __call__(self, rule):
        _try_to_sync_if_needed(self)
        cuda.nvtx.RangePop()


class FwdBwdProfileMarkHook(CUDAProfileHook):

    name = 'FwdBwdProfileMarkHook'

    def __init__(self, sync=True):
        super(FwdBwdProfileMarkHook, self).__init__()
        self._sync = sync

    def forward_preprocess(self, function, in_data):
        _try_to_sync_if_needed(self)
        super(FwdBwdProfileMarkHook, self).forward_preprocess(function, in_data)

    def forward_postprocess(self, function, in_data):
        _try_to_sync_if_needed(self)
        super(FwdBwdProfileMarkHook, self).forward_postprocess(function, in_data)

    def backward_preprocess(self, function, in_data, out_grad):
        _try_to_sync_if_needed(self)
        super(FwdBwdProfileMarkHook, self).backward_preprocess(function, in_data, out_grad)

    def backward_postprocess(self, function, in_data, out_grad):
        _try_to_sync_if_needed(self)
        super(FwdBwdProfileMarkHook, self).backward_postprocess(function, in_data, out_grad)

def _add_backward_mark(func, sync):
    def backward_wrapper(*args, **kwargs):
        with prof.TimeRangeDecorator('model.backward', sync=sync):
            with FwdBwdProfileMarkHook(sync=sync):
                ret = func(*args, **kwargs)
        return ret
    return backward_wrapper

def _add_forward_mark(func, sync):
    def forward_wrapper(*args, **kwargs):
        with prof.TimeRangeDecorator('model.forward', sync=sync):
            with FwdBwdProfileMarkHook(sync=sync):
                ret = func(*args, **kwargs)
        ret.backward = _add_backward_mark(ret.backward, sync)
        return ret
    return forward_wrapper

def _update_with_profiling_mark(self, lossfun=None, *args, **kwds):
    if lossfun is not None:
        lossfun = _add_forward_mark(lossfun, self.sync_for_prof)
    return super(self.__class__, self).update(lossfun, *args, **kwds)

def _setup(self, link):
    ret = super(self.__class__, self).setup(link)
    self.add_hook(UpdateProfileMarkPreHook(sync=self.sync_for_prof))
    self.add_hook(UpdateProfileMarkPostHook(sync=self.sync_for_prof))
    return ret


def create_marked_profile_optimizer(basecls, sync=True):
    if basecls:
        # TODO: check if basecls is subclass of Optimizer.
        pass

    MarkedProfileOptimizer = type(
        'MarkedProfileOptimizer',
        (basecls, ),
        {'update': _update_with_profiling_mark,
         'setup': _setup})
    def make_instance(*args, **kwargs):
        optimizer = MarkedProfileOptimizer(*args, **kwargs)
        optimizer.sync_for_prof = sync
        return optimizer

    return make_instance
