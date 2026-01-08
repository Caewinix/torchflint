from typing import Callable, Optional, Dict, Any, Union, Mapping, Iterable
from types import FrameType, MethodType, FunctionType, BuiltinFunctionType
from collections import OrderedDict
import inspect
import sys
import copyreg
import torch
from torch import nn
from torch.nn.parameter import UninitializedTensorMixin
import pyreflex
from pyreflex import typeattr
from pyreflex.pybase import framelocals
from packaging import version


def _addindent(s_: str, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class _BufferingModule:
    # This class instance will replace the `nn.Module` instance `self` in the `register_parameter` when registering a buffer,
    # while the `self` in the `__setattr__` will keep same.
    class FakeDict:
        def __setitem__(self, _, __): pass
    def __init__(self):
        self._parameters = _BufferingModule.FakeDict()


if version.parse(torch.__version__) >= version.parse('2.0.0'):
    class _BufferingFakeDict:
        # This class instance will replace the global variable `_global_parameter_registration_hooks` in the `register_parameter`
        # when registering a buffer, and it returns a iterator when `values()` is called. When iterating, the original
        # `_global_parameter_registration_hooks` will be restored.
        class Iterator:
            def __init__(self, frame, global_parameter_registration_hooks):
                self.frame = frame
                self.global_parameter_registration_hooks = global_parameter_registration_hooks
                
            def __iter__(self):
                return self
        
            def __next__(self):
                # Retore the original `_global_parameter_registration_hooks`.
                self.frame.f_globals['_global_parameter_registration_hooks'] = self.global_parameter_registration_hooks
                raise StopIteration
        
        def __init__(self, frame, global_parameter_registration_hooks):
            self.frame = frame
            self.global_parameter_registration_hooks = global_parameter_registration_hooks
        
        def values(self):
            return _BufferingFakeDict.Iterator(self.frame, self.global_parameter_registration_hooks)
    
    def _hook_global_parameter_registration(last_frame: FrameType):
        global_parameter_registration_hooks = last_frame.f_globals['_global_parameter_registration_hooks']
        last_frame.f_globals['_global_parameter_registration_hooks'] = _BufferingFakeDict(last_frame, global_parameter_registration_hooks)
else:
    def _hook_global_parameter_registration(_): pass


def _bufferingmethod(funcobj):
    # Only to tell `typeattr` that only the "buffering method" is required to be called.
    funcobj.__isbufferingmethod__ = True
    return funcobj


def _is_buffering_method(item):
    # The condition itself for a `typeattr` instance, indicating that only the
    # "buffering method" is required to be called.
    return getattr(item, '__isbufferingmethod__', False)


def _hidden(obj):
    obj.__buffering_is_hidden__ = True
    return obj


def _subclassvisible(obj):
    obj.__buffering_is_subclass_visible__ = True
    return obj


class BufferingHiddenBase:
    def __init_subclass__(cls, **kwargs):
        hidden_names = []
        for name, obj in cls.__dict__.items():
            if getattr(obj, '__buffering_is_hidden__', False):
                hidden_names.append(name)
        if len(hidden_names) > 0:
            original_getattribute = cls.__getattribute__
            def __getattribute__(self, name: str):
                get_attr = MethodType(original_getattribute, self)
                if type(self) is cls and name in hidden_names:
                    self_dict: dict = get_attr('__dict__')
                    item = self_dict.get(name)
                    if item:
                        return item
                    else:
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                else:
                    attr = get_attr(name)
                    if getattr(attr, '__buffering_is_hidden__', False) and not getattr(attr, '__buffering_is_subclass_visible__', False):
                        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
                    else:
                        return attr
            cls.__getattribute__ = __getattribute__


class Buffering(BufferingHiddenBase):
    '''
    A class used to represent a content as a buffer, which can be used in the `__setattr__` of the `nn.Module` instance,
    or registered by the method `register_buffer` called by `nn.Module` instance.
    '''
    @_bufferingmethod
    @_hidden
    def _register_to(self, module: nn.Module, name: str):
        module.register_buffer(name, typeattr(self, _is_buffering_method)[__class__._get_content.__code__.co_name](self), typeattr(self, _is_buffering_method)[__class__._get_persistent.__code__.co_name](self))
    
    @_bufferingmethod
    @_hidden
    def _modify_generically(self, func: Callable[[Any], Any], *args, **kwargs): pass
    
    @_bufferingmethod
    @_hidden
    def _copy_generically(self, func: Callable[[Any], Any], *args, **kwargs): pass
    
    @_bufferingmethod
    @_hidden
    @staticmethod
    def _standard_type(): pass
    
    @_bufferingmethod
    @_hidden
    def _get_apparent_type(self) -> Any: pass
    
    @_bufferingmethod
    @_hidden
    def _set_apparent_type(self, value): pass
    
    @_bufferingmethod
    @_hidden
    def _get_attr(self, name: str) -> Any: pass

    @_bufferingmethod
    @_hidden
    def _set_attribute(self, name: str, value): pass
    
    @_bufferingmethod
    @_hidden
    def _delete_attribute(self, name: str): pass
    
    @_bufferingmethod
    @_hidden
    def _get_content(self) -> Any: pass
    
    @_bufferingmethod
    @_hidden
    def _get_persistent(self) -> bool: pass
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return torch.Tensor.__torch_function__.__func__(cls, func, types, args, kwargs)
    
    def __getattr__(self, name: str):
        if name == 'grad_fn':
            # The attibute `grad_fn` is called by the `register_parameter` method, so it can be replaced by this getter to
            # insert any operations, which ignores the storage in `register_parameter` as a parameter, instead, `register_buffer`
            # will be called here.
            i = pyreflex.overriding_depth()
            first_frame = pyreflex.get_frame_at(i)
            second_frame = pyreflex.get_frame_at(i + 1)
            if (first_frame and first_frame.f_code.co_name == 'register_parameter') and (second_frame and second_frame.f_code.co_name == '__setattr__'):
                # Only if the last frame function is `register_parameter` and the last second frame fuction is `__setattr__`, 
                # which means it is in the parameterized hook process, this buffer storage will be called, else `grad_fn`
                # will be gotten normally.
                last_frame = first_frame
                module_self: nn.Module = pyreflex.self_from_frame(last_frame)
                if module_self is not None and isinstance(module_self, nn.Module):
                    co_varnames = last_frame.f_code.co_varnames
                    if len(co_varnames) > 1:
                        last_self_name = co_varnames[0]
                        last_name = last_frame.f_locals[co_varnames[1]]
                        
                        framelocals(last_frame)[last_self_name] = _BufferingModule()
                        _hook_global_parameter_registration(last_frame)
                        
                        typeattr(self, _is_buffering_method)[__class__._register_to.__code__.co_name](self, module_self, last_name)
                        return False
        attr = typeattr(self, _is_buffering_method)[__class__._get_attr.__code__.co_name](self, name)
        return attr
    
    def __setattr__(self, name: str, value):
        if name == '__class__':
            # Only the attribute `__class__` is required to be set using the property setter.
            super().__setattr__(name, value)
        else:
            # All other attributes should be set in the stored content.
            typeattr(self, _is_buffering_method)[__class__._set_attribute.__code__.co_name](self, name, value)
    
    def __delattr__(self, name: str):
        typeattr(self, _is_buffering_method)[__class__._delete_attribute.__code__.co_name](self, name)
    
    def __repr__(self) -> str:
        return f"<unregistered buffer of {typeattr(self, _is_buffering_method)[__class__._get_content.__code__.co_name](self).__repr__()}>"
    
    @property
    def __class__(self):
        apparent_type = None
        standard_type = typeattr(self, _is_buffering_method)[__class__._standard_type.__code__.co_name]()
        i = pyreflex.overriding_depth()
        while pyreflex.get_frame_at(i).f_code.co_name == '__getattribute__':
            i += 1
        first_frame = pyreflex.get_frame_at(i)
        second_frame = pyreflex.get_frame_at(i + 1)
        third_frame = pyreflex.get_frame_at(i + 2)
        if second_frame and (second_frame.f_code.co_name == '__setattr__' or 
            (third_frame and second_frame.f_code.co_name == 'register_parameter' and third_frame.f_code.co_name == '__setattr__')):
            # Consider the current frame function is `__instancecheck__`, if the last frame function is `__setattr__`,
            # or the last frame function is `register_parameter` and the last second frame function is `__setattr__`,
            # that means it is in the parameterized hook process.
            frame = second_frame
            module_self: nn.Module = pyreflex.self_from_frame(frame)
            apparent_type = typeattr(self, _is_buffering_method)[__class__._get_apparent_type.__code__.co_name](self)
            if module_self is not None and isinstance(module_self, nn.Module) and issubclass(apparent_type, standard_type):
                # When the `__class__` of the stored content is `standard_type` or its subclass, returning the `nn.Parameter`
                # to hook the `__instancecheck__`.
                return nn.Parameter
        elif first_frame and first_frame.f_code.co_name == 'register_buffer':
            # When the buffer is passed to the `register_buffer` method, the stored content and its persistent status will
            # be extracted into the `register_buffer` method. (Strangely, the `__instancecheck__` is not called, so the
            # stack index is less.)
            frame = first_frame
            module_self: nn.Module = pyreflex.self_from_frame(frame)
            if module_self is not None and isinstance(module_self, nn.Module):
                co_varnames = frame.f_code.co_varnames
                if len(co_varnames) > 3:
                    from pyreflex import get_instruction_at
                    tensor_name = co_varnames[2]
                    persistent_name = co_varnames[3]
                    lasti = frame.f_back.f_lasti
                    call_instruction = get_instruction_at(frame.f_back.f_code, lasti)
                    frame_locals = framelocals(frame)
                    if call_instruction.argval is not None and call_instruction.argval <= 2:
                        # Only when explicitly passing the `persistent` argument to the method `register_buffer`, the passed
                        # one will be applied, otherwise the `persistent` attribute in the buffer instance will be exploited.
                        # Meanwhile, using the form of "module.register_buffer(*args, **kwargs)" will only ignore `persistent`
                        # argument of the method, only keep the buffer's setting.
                        frame_locals[persistent_name] = typeattr(self, _is_buffering_method)[__class__._get_persistent.__code__.co_name](self)
                    frame_locals[tensor_name] =  typeattr(self, _is_buffering_method)[__class__._get_content.__code__.co_name](self)
                    
                    apparent_type = typeattr(self, _is_buffering_method)[__class__._get_apparent_type.__code__.co_name](self)
                    if issubclass(apparent_type, standard_type):
                        return torch.Tensor
        elif second_frame.f_code.co_name == '_load_from_state_dict' and first_frame.f_code.co_name == 'is_lazy' and issubclass(type(self), BufferObject):
            return UninitializedTensorMixin
        
        return apparent_type if apparent_type else typeattr(self, _is_buffering_method)[__class__._get_apparent_type.__code__.co_name](self)
    
    @__class__.setter
    def __class__(self, value):
        # The `__class__` should be set through the stored content.
        typeattr(self, _is_buffering_method)[__class__._set_apparent_type.__code__.co_name](self, value)
    
    def __copy__(self):
        from copy import copy
        return typeattr(self, _is_buffering_method)[__class__._copy_generically.__code__.co_name](self, copy)
    
    def __deepcopy__(self, memodict={}):
        from copy import deepcopy
        return typeattr(self, _is_buffering_method)[__class__._copy_generically.__code__.co_name](self, deepcopy, memodict)


class Buffer(Buffering):
    """
    A wrapper class for PyTorch tensors that enables transparent registration as buffers
    in PyTorch modules, with persistent storage control and full tensor interface support.
    
    This class wraps a tensor and intercepts the registration process to ensure it is
    registered as a buffer using Python assignment. It provides complete forwarding of
    tensor operations, properties, and methods while maintaining control over persistence
    behavior during state saving.
    
    Args:
        tensor (Tensor, optional): the tensor to wrap. If `None`, creates an empty buffer
        persistent (bool): whether the buffer should be persistent in state dict.
            Default: True
    
    Examples:
        >>> tensor = torch.randn(3, 4)
        >>> buffer = Buffer(tensor, persistent=True)
        >>>
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.my_buffer = buffer
        >>>
        >>> my_module = MyModule()
        >>> my_module.my_buffer.shape
        torch.Size([3, 4])
    """
    def __init__(self, tensor: Optional[torch.Tensor], persistent: bool = True):
        object.__setattr__(self, '__content', tensor)
        object.__setattr__(self, '__persistent', persistent)
    
    @_bufferingmethod
    @_hidden
    def _modify_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        content = object.__getattribute__(self, '__content')
        result = func(content, *args, **kwargs)
        if isinstance(result, torch.Tensor) or issubclass(type(result), Buffering):
            object.__setattr__(self, '__content', result)
        else:
            return result
        return self
    
    @_bufferingmethod
    @_hidden
    def _copy_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        content = object.__getattribute__(self, '__content')
        result = func(content, *args, **kwargs)
        if isinstance(result, torch.Tensor) or issubclass(type(result), Buffering):
            instance = type(self).__new__(type(self))
            object.__setattr__(instance, '__content', result)
        else:
            return result
        object.__setattr__(instance, '__persistent', object.__getattribute__(self, '__persistent'))
        return instance
    
    @_bufferingmethod
    @_hidden
    @staticmethod
    def _standard_type():
        return torch.Tensor
    
    @_bufferingmethod
    @_hidden
    def _get_apparent_type(self) -> Any:
        return object.__getattribute__(self, '__content').__class__
    
    @_bufferingmethod
    @_hidden
    def _set_apparent_type(self, value):
        object.__getattribute__(self, '__content').__class__ = value
    
    @_bufferingmethod
    @_hidden
    def _get_attr(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            content = object.__getattribute__(self, '__content')
            attr = content.__getattribute__(name)
            # If the instance is a builtin function of `torch.Tensor`, then wrap it through `modify_generically` when the function name ends with "_",
            # otherwise through `copy_generically`.
            def when_function(func):
                if name.endswith('_'):
                    def required_function(instance, *args, **kwargs):
                        return typeattr(instance, _is_buffering_method)[__class__._modify_generically.__code__.co_name](instance, func, *args, **kwargs)
                else:
                    def required_function(instance, *args, **kwargs):
                        return typeattr(instance, _is_buffering_method)[__class__._copy_generically.__code__.co_name](instance, func, *args, **kwargs)
                required_function.__name__ = name
                required_function.__qualname__ = name
                return MethodType(required_function, self)
            if type(attr) is BuiltinFunctionType:
                func = attr
                attr = when_function(lambda _, *args, **kwargs: func(*args, **kwargs))
            elif type(attr) is MethodType:
                attr = when_function(attr.__func__)
            elif type(attr) is FunctionType:
                attr = when_function(attr)
            return attr

    @_bufferingmethod
    @_hidden
    def _set_attribute(self, name: str, value):
        object.__getattribute__(self, '__content').__setattr__(name, value)
    
    @_bufferingmethod
    @_hidden
    def _delete_attribute(self, name: str):
        object.__getattribute__(self, '__content').__delattr__(name)
    
    @_bufferingmethod
    @_hidden
    def _get_content(self):
        return object.__getattribute__(self, '__content')
    
    @_bufferingmethod
    @_hidden
    def _get_persistent(self):
        return object.__getattribute__(self, '__persistent')
    
    def __del__(self): object.__delattr__(self, '__content')
    def __cmp__(self, other): return object.__getattribute__(self, '__content').__cmp__(other)
    def __eq__(self, other): return object.__getattribute__(self, '__content') == other
    def __ne__(self, other): return object.__getattribute__(self, '__content') != other
    def __lt__(self, other): return object.__getattribute__(self, '__content') < other
    def __gt__(self, other): return object.__getattribute__(self, '__content') > other
    def __le__(self, other): return object.__getattribute__(self, '__content') <= other
    def __ge__(self, other): return object.__getattribute__(self, '__content') >= other
    def __pos__(self): return +object.__getattribute__(self, '__content')
    def __neg__(self): return -object.__getattribute__(self, '__content')
    def __abs__(self): return abs(object.__getattribute__(self, '__content'))
    def __invert__(self): return ~object.__getattribute__(self, '__content')
    def __round__(self, n): return round(object.__getattribute__(self, '__content'), n)
    def __floor__(self): return object.__getattribute__(self, '__content').__floor__()
    def __ceil__(self): return object.__getattribute__(self, '__content').__ceil__()
    def __trunc__(self): return object.__getattribute__(self, '__content').__trunc__()
    def __add__(self, other): return object.__getattribute__(self, '__content') + other
    def __sub__(self, other): return object.__getattribute__(self, '__content') - other
    def __mul__(self, other): return object.__getattribute__(self, '__content') * other
    def __floordiv__(self, other): return object.__getattribute__(self, '__content') // other
    def __div__(self, other): return object.__getattribute__(self, '__content') / other
    def __truediv__(self, other): return object.__getattribute__(self, '__content').__truediv__(other)
    def __mod__(self, other): return object.__getattribute__(self, '__content') % other
    def __divmod__(self, other): return divmod(object.__getattribute__(self, '__content'), other)
    def __pow__(self, other): return object.__getattribute__(self, '__content') ** other
    def __lshift__(self, other): return object.__getattribute__(self, '__content') << other
    def __rshift__(self, other): return object.__getattribute__(self, '__content') >> other
    def __and__(self, other): return object.__getattribute__(self, '__content') & other
    def __or__(self, other): return object.__getattribute__(self, '__content') | other
    def __xor__(self, other): return object.__getattribute__(self, '__content') ^ other
    def __radd__(self, other): return other + object.__getattribute__(self, '__content')
    def __rsub__(self, other): return other - object.__getattribute__(self, '__content')
    def __rmul__(self, other): return other * object.__getattribute__(self, '__content')
    def __rfloordiv__(self, other): return other // object.__getattribute__(self, '__content')
    def __rdiv__(self, other): return other / object.__getattribute__(self, '__content')
    def __rtruediv__(self, other): return object.__getattribute__(self, '__content').__rtruediv__(other)
    def __rmod__(self, other): return other % object.__getattribute__(self, '__content')
    def __rdivmod__(self, other): return divmod(other, object.__getattribute__(self, '__content'))
    def __rpow__(self, other): return other ** object.__getattribute__(self, '__content')
    def __rlshift__(self, other): return other << object.__getattribute__(self, '__content')
    def __rrshift__(self, other): return other >> object.__getattribute__(self, '__content')
    def __rand__(self, other): return other & object.__getattribute__(self, '__content')
    def __ror__(self, other): return other | object.__getattribute__(self, '__content')
    def __rxor__(self, other): return other ^ object.__getattribute__(self, '__content')
    def __iadd__(self, other): object.__getattribute__(self, '__content').__iadd__(other)
    def __isub__(self, other): object.__getattribute__(self, '__content').__isub__(other)
    def __imul__(self, other): object.__getattribute__(self, '__content').__imul__(other)
    def __ifloordiv__(self, other): object.__getattribute__(self, '__content').__ifloordiv__(other)
    def __idiv__(self, other): object.__getattribute__(self, '__content').__idiv__(other)
    def __itruediv__(self, other): object.__getattribute__(self, '__content').__itruediv__(other)
    def __imod__(self, other): object.__getattribute__(self, '__content').__imod__(other)
    def __ipow__(self, other): object.__getattribute__(self, '__content').__ipow__(other)
    def __ilshift__(self, other): object.__getattribute__(self, '__content').__ilshift__(other)
    def __irshift__(self, other): object.__getattribute__(self, '__content').__irshift__(other)
    def __iand__(self, other): object.__getattribute__(self, '__content').__iand__(other)
    def __ior__(self, other): object.__getattribute__(self, '__content').__ior__(other)
    def __ixor__(self, other): object.__getattribute__(self, '__content').__ixor__(other)
    def __int__(self): return object.__getattribute__(self, '__content').__int__()
    def __long__(self): return object.__getattribute__(self, '__content').__long__()
    def __float__(self): return object.__getattribute__(self, '__content').__float__()
    def __complex__(self): return object.__getattribute__(self, '__content').__complex__()
    def __oct__(self): return object.__getattribute__(self, '__content').__oct__()
    def __hex__(self): return object.__getattribute__(self, '__content').__hex__()
    def __index__(self): return object.__getattribute__(self, '__content').__index__()
    def __trunc__(self): return object.__getattribute__(self, '__content').__trunc__()
    def __coerce__(self, other): return object.__getattribute__(self, '__content').__coerce__(other)
    def __str__(self): return object.__getattribute__(self, '__content').__str__()
    def __unicode__(self): return object.__getattribute__(self, '__content').__unicode__()
    def __format__(self, formatstr): return object.__getattribute__(self, '__content').__format__(formatstr)
    def __hash__(self): return object.__getattribute__(self, '__content').__hash__()
    def __nonzero__(self): return object.__getattribute__(self, '__content').__nonzero__()
    def __dir__(self): return object.__getattribute__(self, '__content').__dir__()
    def __sizeof__(self): return object.__getattribute__(self, '__content').__sizeof__()
    def __len__(self): return len(object.__getattribute__(self, '__content'))
    def __getitem__(self, key): return object.__getattribute__(self, '__content')[key]
    def __setitem__(self, key, value): object.__getattribute__(self, '__content')[key] = value
    def __delitem__(self, key): object.__getattribute__(self, '__content').__delitem__(key)
    def __iter__(self): return iter(object.__getattribute__(self, '__content'))
    def __reversed__(self): return reversed(object.__getattribute__(self, '__content'))
    def __contains__(self, item): return item in object.__getattribute__(self, '__content')
    def __missing__(self, key): return object.__getattribute__(self, '__content').__missing__(key)
    def __call__(self, *args, **kwargs): return object.__getattribute__(self, '__content')(*args, **kwargs)
    def __enter__(self): return object.__getattribute__(self, '__content').__enter__()
    def __exit__(self, exception_type, exception_value, traceback): return object.__getattribute__(self, '__content').__exit__(exception_type, exception_value, traceback)
    def __get__(self, instance, owner): return object.__getattribute__(self, '__content').__get__(instance, owner)
    def __set__(self, instance, value): object.__getattribute__(self, '__content').__set__(instance, value)
    def __delete__(self, instance): return object.__getattribute__(self, '__content').__delete__(instance)


def _content_named_forward(content, *args, **kwargs):
    # Perfectly forward the called method name of the `BufferObject` instance.
    try:
        return content.__getattribute__(pyreflex.function_name(3))(*args, **kwargs)
    except AttributeError:
        # Sometimes, when the `_contents` dictionary in the `BufferObject` instance contains the
        # element that is not a `torch.Tensor` or something that cannot be forwarded the function
        # name that originally comes from the `torch.Tensor`, returning the element itself is the
        # best trade-off operation, this situation basically occurs when the type is `BufferDict`
        # or `BufferList`.
        return content


def _named_forward_copy(instance, *args, **kwargs):
    return typeattr(instance, _is_buffering_method)[Buffering._copy_generically.__code__.co_name](instance, _content_named_forward, *args, **kwargs)


def _named_forward_modify(instance, *args, **kwargs):
    # Originally using `_content_named_forward_inplace`.
    return typeattr(instance, _is_buffering_method)[Buffering._modify_generically.__code__.co_name](instance, _content_named_forward, *args, **kwargs)


def _get_buffers_items(buffers, prefix: str = "", recurse: bool = True):
    if hasattr(buffers, 'items'):
        buffers_iterable = buffers.items()
    else:
        buffers_iterable = ((f'{i}', buffer) for i, buffer in enumerate(buffers))
    def prefix_name(name):
        if prefix == "":
            return name
        else:
            return f'{prefix}.{name}'
    for name, buffer in buffers_iterable:
        yield_name = prefix_name(name)
        if isinstance(buffer, BufferObject):
            if recurse:
                contents = object.__getattribute__(buffer, '_contents')
                yield from _get_buffers_items(contents, yield_name, recurse)
        elif isinstance(buffer, torch.Tensor):
            yield yield_name, buffer


class _BufferObjectDictionary(dict):
    def __init__(self, buffers: dict, prefix = "", recurse = True):
        self.__buffers = buffers
        self.__prefix = prefix
        self.__recurse = recurse
    
    def __getattribute__(self, name):
        if name in object.__getattribute__(self, '__dict__') or name in _BufferObjectDictionary.__dict__:
            return super().__getattribute__(name)
        elif name == '__class__':
            return dict
        else:
            return object.__getattribute__(self.__buffers, name)
    
    def __setattr__(self, name, value):
        if name == '__class__':
            dict.__class__ = value
            return
        return super().__setattr__(name, value)
    
    def __contains__(self, key):
        return key in self.__buffers
    
    def __len__(self) -> int:
        return len(self.__buffers)
    
    def __getitem__(self, key: str) -> object:
        return self.__buffers[key]
    
    def __setitem__(self, key: str, value: object):
        self.__buffers[key] = value
    
    def __delitem__(self, key: str):
        del self.__buffers[key]
    
    def __iter__(self):
        return iter(self.__buffers)
    
    def __eq__(self, value: object) -> bool:
        return self.__buffers == value
    
    def __reversed__(self):
        return self.__buffers.__reversed__()
    
    def items(self):
        yield from _get_buffers_items(self.__buffers, self.__prefix, self.__recurse)
    
    if sys.version_info >= (3, 9):
        def __or__(self, value: object) -> object:
            return self.__buffers.__or__(value)
        
        def __ror__(self, value: object) -> object:
            return self.__buffers.__ror__(value)
        
        def __ior__(self, value: object) -> object:
            return self.__buffers.__ior__(value)
_BufferObjectDictionary.__name__ = dict.__name__
_BufferObjectDictionary.__qualname__ = dict.__qualname__
_BufferObjectDictionary.__module__ = dict.__module__
_BufferObjectDictionary.__doc__ = dict.__doc__


def _remove_buffer_object_module(module, reference_type):
    module_type = type(module)
    bases = module_type.__bases__
    if issubclass(module_type, reference_type):
        super(module_type, module).__setattr__('__class__', bases[0])
    else:
        all_bases = [(module_type, base, base.__bases__) for base in bases]
        while len(all_bases) != 0:
            child, current_type, bases = all_bases.pop()
            if issubclass(current_type, reference_type):
                child.__bases__ = bases
                break
            all_bases.extend(((current_type, base, base.__bases__) for base in bases))


def _find_type_attr(type: type, name: str):
    mro = type.__mro__
    for current_type in mro:
        if name in type.__dict__:
            return current_type


def _reconstruct_buffer_object_module(cls, base, state):
    module = copyreg._reconstructor(cls, base, state)
    _generate_buffer_object_module_hook(module)
    return module


def _generate_buffer_object_module_hook(module: nn.Module, num_buffer_objects: int = 0):
    module_type = type(module)
    def check_buffer_object(module, buffer):
        if isinstance(buffer, BufferObject):
            current_type = type(module)
            current_type.__numbufferobjects__ -= 1
            if current_type.__numbufferobjects__ == 0:
                _remove_buffer_object_module(module, BufferObjectModule)
    
    class BufferObjectModule(module_type):
        def __reduce__(self):
            reduce = super().__reduce__()
            return _reconstruct_buffer_object_module, *reduce[1:]
        
        def __setstate__(self, state):
            buffers = state['_buffers']
            num_buffer_objects = sum((1 if isinstance(buffer, BufferObject) else 0 for buffer in buffers.values()))
            type(self).__numbufferobjects__ += num_buffer_objects
            return super().__setstate__(state)
        
        def __getattribute__(self, name):
            if name == '__class__' and _find_type_attr(type(self), '__class__') is None:
                return BufferObjectModule.__base__
            elif name == '_buffers':
                frame = inspect.currentframe().f_back.f_back.f_back
                if frame is not None and frame.f_code.co_name == 'named_buffers':
                    prefix = frame.f_locals.get('prefix', "")
                    recurse = frame.f_locals.get('recurse', True)
                    return _BufferObjectDictionary(self._buffers, prefix, recurse)
            return super().__getattribute__(name)
        
        def __setattr__(self, name, value):
            if name == '__class__' and _find_type_attr(type(self), '__class__') is None:
                BufferObjectModule.__bases__ = (value,)
            else:
                buffer = super().__getattribute__('_buffers').get(name, None)
                super().__setattr__(name, value)
                if isinstance(value, BufferObject):
                    type(self).__numbufferobjects__ += 1
                if buffer is not None:
                    check_buffer_object(self, buffer)
        
        def __delattr__(self, name: str):
            buffer = super().__getattribute__('_buffers').get(name, None)
            super().__delattr__(name)
            if buffer is not None:
                check_buffer_object(self, buffer)
    
    BufferObjectModule.__numbufferobjects__ = num_buffer_objects
    BufferObjectModule.__name__ = module_type.__name__
    BufferObjectModule.__qualname__ = module_type.__qualname__
    BufferObjectModule.__module__ = module_type.__module__
    BufferObjectModule.__doc__ = module_type.__doc__
    module.__class__ = BufferObjectModule


class BufferObject(Buffering):
    """
    BufferObject, as a base class, it serves as a container for multiple buffer elements,
    allowing them to be registered and managed as a single unit. It supports both tensor
    and nested buffer storage, with automatic persistence control and module integration.
    
    Examples:
        >>> class MyBufferObject(BufferObject):
        ...     def __init__(self):
        ...         self.weight = torch.randn(3, 4)
        ...         self.bias = torch.randn(4)
        >>> 
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.my_buffer = MyBufferObject()
        >>>
        >>> my_module = MyModule()
        >>> my_module.my_buffer.weight.shape
        torch.Size([3, 4])
    """
    _contents: Dict[str, Union[torch.Tensor, Buffering, None]]
    
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        object.__setattr__(instance, '__class', cls)
        object.__setattr__(instance, '__persistent', True)
        object.__setattr__(instance, '_contents', cls._make_contents())
        return instance
    
    @property
    def is_meta(self):
        return False
    
    @_bufferingmethod
    @_hidden
    def _modify_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        contents = object.__getattribute__(self, '_contents')
        for name, content in contents.items():
            contents[name] = func(content, *args, **kwargs)
        return self
    
    @_bufferingmethod
    @_hidden
    def _copy_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        instance = type(self).__new__(type(self))
        self_dict: dict = object.__getattribute__(self, '__dict__')
        for name, value in self_dict.items():
            type(self).__setattr__(instance, name, value)
        for name, content in object.__getattribute__(self, '_contents').items():
            object.__getattribute__(instance, '_contents')[name] = func(content, *args, **kwargs)
        return instance
    
    @_bufferingmethod
    @_hidden
    @staticmethod
    def _make_contents():
        return OrderedDict()
    
    @property
    def persistent(self):
        return object.__getattribute__(self, '__persistent')
    
    @persistent.setter
    def persistent(self, value):
        object.__setattr__(self, '__persistent', value)
    
    def set_persistent(self, value):
        self.persistent = value
        return self
    
    @_bufferingmethod
    @_hidden
    def _register_to(self, module: nn.Module, name: str):
        return self.register_to(module, name)
    
    @_bufferingmethod
    def register_to(self, module: nn.Module, name: str):
        module.register_buffer(name, self, object.__getattribute__(self, '__persistent'))
        if not hasattr(type(module), '__numbufferobjects__'):
            _generate_buffer_object_module_hook(module, 1)
    
    @_bufferingmethod
    @_hidden
    @staticmethod
    def _standard_type():
        return __class__
    
    @_bufferingmethod
    @_hidden
    def _get_apparent_type(self) -> Any:
        return object.__getattribute__(self, '__class')
    
    @_bufferingmethod
    @_hidden
    def _set_apparent_type(self, value):
        object.__setattr__(self, '__class', value)
    
    @_bufferingmethod
    @_hidden
    def _get_attr(self, name: str) -> Any:
        attr = object.__getattribute__(self, '_contents').get(name)
        if attr is None:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        else:
            return attr

    @_bufferingmethod
    @_hidden
    def _set_attribute(self, name: str, value):
        object.__getattribute__(self, '_contents').pop(name, None)
        try:
            object.__delattr__(self, name)
        except AttributeError: ...
        if isinstance(value, torch.Tensor) or isinstance(value, Buffering):
            object.__getattribute__(self, '_contents')[name] = value
        else:
            object.__setattr__(self, name, value)
    
    @_bufferingmethod
    @_hidden
    def _delete_attribute(self, name: str):
        try:
            object.__getattribute__(self, '_contents').pop(name)
        except KeyError:
            object.__delattr__(self, name)
    
    @_bufferingmethod
    @_hidden
    def _get_content(self):
        return self
    
    @_bufferingmethod
    @_hidden
    def _get_persistent(self):
        return object.__getattribute__(self, '__persistent')
    
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = ''
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, tensor in object.__getattribute__(self, '_contents').items():
            mod_str = repr(tensor)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = type(self).__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str
    
    def copy_(self, src, non_blocking: bool = False):
        contents = object.__getattribute__(self, '_contents')
        src_contents = object.__getattribute__(src, '_contents')
        for name in contents.keys():
            contents[name].copy_(src_contents[name], non_blocking)
        return self
    
    @property
    def dtype(self):
        contents = object.__getattribute__(self, '_contents')
        try:
            return contents[0].dtype
        except: ...
        return None
    
    @property
    def device(self):
        contents = object.__getattribute__(self, '_contents')
        try:
            return contents[0].device
        except: ...
        return None
    
    def to(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def cuda(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def ipu(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def xpu(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def cpu(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def type(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def float(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def double(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def half(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def bfloat16(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def detach(self, *args, **kwargs): return _named_forward_copy(self, *args, **kwargs)
    def is_floating_point(self): return False
    def is_complex(self): return False
    def to_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def cuda_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def ipu_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def xpu_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def cpu_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def type_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def float_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def double_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def half_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def bfloat16_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)
    def detach_(self, *args, **kwargs): return _named_forward_modify(self, *args, **kwargs)


class BufferDict(BufferObject):
    """
    A dictionary-like buffer container that maps keys to tensors or buffers.
    
    BufferDict provides a mutable mapping interface for storing and accessing buffers
    by name. It can be registered as a single buffer in PyTorch modules while containing
    multiple named elements. The dictionary contents are automatically traversed during
    module state management.
    
    Args:
        buffers (Mapping[str, Any], optional): initial buffer mapping
    
    Examples:
        >>> buffer_dict = BufferDict({
        ...     'weight': torch.randn(3, 4),
        ...     'bias': torch.randn(4)
        ... })
        >>> 
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.buffer_dict = buffer_dict
        >>>
        >>> my_module = MyModule()
        >>> my_module.buffer_dict['weight'].shape
        torch.Size([3, 4])
    """
    _contents: Mapping[str, Any]
    
    def __init__(self, buffers: Optional[Mapping[str, Any]] = None):
        if buffers is not None:
            self.update(buffers)
    
    @_bufferingmethod
    @_hidden
    def _get_attr(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @_bufferingmethod
    @_hidden
    def _set_attribute(self, name: str, value):
        object.__setattr__(self, name, value)
    
    @_bufferingmethod
    @_hidden
    def _delete_attribute(self, name: str):
        object.__delattr__(self, name)
    
    def __getitem__(self, key: str) -> Any:
        return object.__getattribute__(self, '_contents')[key]
    
    def __setitem__(self, key: str, value):
        object.__getattribute__(self, '_contents')[key] = value
    
    def __delitem__(self, key: str):
        object.__getattribute__(self, '_contents').pop(key)
    
    def __reversed__(self):
        return reversed(object.__getattribute__(self, '_contents'))
    
    def __len__(self):
        return len(object.__getattribute__(self, '_contents'))
    
    def __contains__(self, key: str):
        return object.__getattribute__(self, '_contents').__contains__(key)
    
    def __iter__(self):
        return iter(object.__getattribute__(self, '_contents'))
    
    def update(self, buffers: Mapping[str, Any]):
        object.__getattribute__(self, '_contents').update(buffers)
    
    def get(self, key: str):
        return object.__getattribute__(self, '_contents').get(key)
    
    def pop(self, key: str):
        return object.__getattribute__(self, '_contents').pop(key)
    
    def popitem(self):
        return object.__getattribute__(self, '_contents').popitem()
    
    def clear(self):
        object.__getattribute__(self, '_contents').clear()
    
    def copy(self):
        return self.__copy__()
    
    def setdefault(self, key: str, default: Any = None) -> Any:
        return object.__getattribute__(self, '_contents').setdefault(key, default)
    
    def items(self):
        return object.__getattribute__(self, '_contents').items()
    
    def keys(self):
        return object.__getattribute__(self, '_contents').keys()
    
    def values(self):
        return object.__getattribute__(self, '_contents').values()


class BufferList(BufferObject):
    """
    A list-like buffer container that stores tensors or buffers in sequence.
    
    BufferList provides a mutable sequence interface for storing ordered collections
    of buffers. It supports list operations like appending, indexing, and iteration,
    and can be registered as a single buffer in PyTorch modules while containing
    multiple elements.
    
    Args:
        buffers (Iterable[Any], optional): initial buffer sequence
    
    Examples:
        >>> buffer_list = BufferList([
        ...     torch.randn(3, 4),
        ...     torch.randn(4),
        ...     torch.randn(5)
        ... ])
        >>> 
        >>> class MyModule(torch.nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.buffer_list = buffer_list
        >>> 
        >>> my_module = MyModule()
        >>> my_module.buffer_list[0].shape
        torch.Size([3, 4])
        >>> len(buffer_list)
        3
    """
    _contents: Iterable[Any]
    
    def __init__(self, buffers: Optional[Iterable[Any]] = None):
        if buffers is not None:
            self += buffers
    
    @_bufferingmethod
    @_hidden
    def _modify_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        contents = object.__getattribute__(self, '_contents')
        for i, content in enumerate(contents):
            contents[i] = func(content, *args, **kwargs)
        return self
    
    @_bufferingmethod
    @_hidden
    def _copy_generically(self, func: Callable[[Any], Any], *args, **kwargs):
        instance = type(self).__new__(type(self))
        self_dict: dict = object.__getattribute__(self, '__dict__')
        for name, value in self_dict.items():
            type(self).__setattr__(instance, name, value)
        for i, content in enumerate(object.__getattribute__(self, '_contents')):
            object.__getattribute__(instance, '_contents')[i] = func(content, *args, **kwargs)
        return instance
    
    @_bufferingmethod
    @_hidden
    @staticmethod
    def _make_contents():
        return []
    
    @_bufferingmethod
    @_hidden
    def _get_attr(self, name: str) -> Any:
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @_bufferingmethod
    @_hidden
    def _set_attribute(self, name: str, value):
        object.__setattr__(self, name, value)
    
    @_bufferingmethod
    @_hidden
    def _delete_attribute(self, name: str):
        object.__delattr__(self, name)
    
    def __getitem__(self, index: int) -> Any:
        return object.__getattribute__(self, '_contents')[index]
    
    def __setitem__(self, index: int, value):
        object.__getattribute__(self, '_contents')[index] = value
    
    def __delitem__(self, index: int):
        object.__getattribute__(self, '_contents').pop(index)
    
    def __reversed__(self):
        return reversed(object.__getattribute__(self, '_contents'))
    
    def __len__(self):
        return len(object.__getattribute__(self, '_contents'))
    
    def __contains__(self, value):
        return object.__getattribute__(self, '_contents').__contains__(value)
    
    def __iter__(self):
        return iter(object.__getattribute__(self, '_contents'))
    
    def count(self, value: Any) -> int:
        return object.__getattribute__(self, '_contents').count(value)
    
    def pop(self, index: int):
        return object.__getattribute__(self, '_contents').pop(index)
    
    def clear(self):
        object.__getattribute__(self, '_contents').clear()
    
    def copy(self):
        return self.__copy__()
    
    def __iadd__(self, other: Iterable[Any]):
        object.__getattribute__(self, '_contents').__iadd__(other)
        return self

    def __add__(self, other: Iterable[Any]):
        combined = type(self)(object.__getattribute__(self, '_contents') + other)
        object.__setattr__(combined, '__persistent', object.__getattribute__(self, '__persistent'))
        return combined

    def insert(self, index: int, buffer: Any):
        object.__getattribute__(self, '_contents').insert(index, buffer)

    def append(self, buffer: Any):
        object.__getattribute__(self, '_contents').append(buffer)
        return self

    def extend(self, buffers: Iterable[Any]):
        object.__getattribute__(self, '_contents').extend(buffers)
        return self

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = ''
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for i, tensor in enumerate(object.__getattribute__(self, '_contents')):
            key = f'{i}'
            mod_str = repr(tensor)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = type(self).__name__ + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str