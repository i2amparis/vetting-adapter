"""Generally useful classes."""
import dataclasses
import typing as tp



def dataclassdict(
        _class: tp.Type[tp.Any],
        /,
        *,
        init: tp.Optional[bool] = None,
        repr: tp.Optional[bool] = None,
        eq: tp.Optional[bool] = None,
        order: tp.Optional[bool] = None,
        unsafe_hash: tp.Optional[bool] = None,
        frozen: tp.Optional[bool] = None,
        match_args: tp.Optional[bool] = None,
        kw_only: tp.Optional[bool] = None,
        slots: tp.Optional[bool] = None,
        weakref_slots: tp.Optional[bool] = None,
) -> type:
    """Decorator similar to `dataclasses.dataclass`, but which is also a dict.
    
    This function can be used as a decorator for classes to turn them into
    dataclasses in the same way that `dataclasses.dataclass` does, but the
    resulting class also acts as a dict, with the keys being the field names
    and the values being the field values. It supports `.keys()`, `.values()`,
    `.items()`, `len()` and iteration in the same way as a regular dict, as well
    as key autcompletion in IPython.

    The decorator suppors the same parameters as `dataclasses.dataclass`. Any
    parameters that are not specified will use the defaults from the
    `dataclasses.dataclass` function.
    """
    _dataclass_params: tuple[str, ...] = (
        'init', 'repr', 'eq', 'order', 'unsafe_hash', 'frozen', 'match_args',
        'kw_only', 'slots', 'weakref_slots'
    )
    dataclass_kwargs: dict[str, bool] = {
        _paramname: locals()[_paramname] for _paramname in _dataclass_params
        if locals()[_paramname] is not None
    }
    class _ReturnClass(dataclasses.dataclass(**dataclass_kwargs)(_class)):
        @property
        def _dict(self) -> dict:
            return dataclasses.asdict(self)
        def __getitem__(self, key: str):
            self._dict[key]
        def __setitem__(self, key: str, value):
            setattr(self, key, value)
        def __iter__(self):
            return iter(self._dict)
        def keys(self):
            return self._dict.keys()
        def values(self):
            return self._dict.values()
        def items(self):
            return self._dict.items()
        def __len__(self):
            return len(self._dict)
        def _ipython_key_completions_(self):
            return list(self.keys())
    return _ReturnClass
