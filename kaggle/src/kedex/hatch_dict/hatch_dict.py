from flatten_dict import flatten
from kedro.utils import load_obj
from six import iteritems
import operator
from typing import Union, List, Iterable  # NOQA


class HatchDict:
    def __init__(
        self,
        egg,  # type: Union[dict, List]
        lookup={},  # type: dict
        support_nested_keys=True,  # type: bool
        self_lookup_key="obj",  # type: str
        support_import=True,  # type: bool
        additional_import_modules=[__name__],  # type: Union[List, str]
        obj_key="obj",  # type: str
    ):
        # type: (...) -> None

        assert egg.__class__.__name__ in {"dict", "list"}
        assert lookup.__class__.__name__ in {"dict"}
        assert support_nested_keys.__class__.__name__ in {"bool"}
        assert self_lookup_key.__class__.__name__ in {"str"}
        assert additional_import_modules.__class__.__name__ in {"list", "str"}
        assert obj_key.__class__.__name__ in {"str"}

        aug_egg = {}
        if isinstance(egg, dict):
            if support_nested_keys:
                aug_egg = dot_flatten(egg)
            aug_egg.update(egg)
        self.aug_egg = aug_egg

        self.egg = egg

        self.lookup = {}

        self.lookup.update(_builtin_funcs())
        self.lookup.update(lookup)

        self.self_lookup_key = self_lookup_key
        self.support_import = support_import
        self.additional_import_modules = (
            [additional_import_modules]
            if isinstance(additional_import_modules, str)
            else additional_import_modules or [__name__]
        )
        self.obj_key = obj_key

        self.warmed_egg = None
        self.snapshot = None

    def get(
        self,
        key=None,  # type: Union[str, int]
        default=None,  # type: Any
        lookup={},  # type: dict
    ):
        # type: (...) -> Any

        assert key.__class__.__name__ in {"str", "int"}
        assert lookup.__class__.__name__ in {"dict"}

        if key is None:
            d = self.egg
        else:
            if isinstance(self.egg, dict):
                d = self.aug_egg.get(key, default)
            if isinstance(self.egg, list):
                assert isinstance(key, int)
                d = self.egg[key] if (0 <= key < len(self.egg)) else default

        if self.self_lookup_key:
            s = dict()
            while d != s:
                d, s = _dict_apply(
                    d_input=d,
                    lookup=self.aug_egg,
                    support_import=False,
                    obj_key=self.self_lookup_key,
                )
            self.warmed_egg = d

        lookup_input = {}
        lookup_input.update(self.lookup)
        lookup_input.update(lookup)

        if isinstance(self.egg, dict):
            forcing_module = self.egg.get("FORCING_MODULE", "")

        for m in self.additional_import_modules:
            d, s = _dict_apply(
                d_input=d,
                lookup=lookup_input,
                support_import=self.support_import,
                default_module=m,
                forcing_module=forcing_module,
                obj_key=self.obj_key,
            )
        self.snapshot = s
        return d

    def get_params(self):
        return self.snapshot


def _dict_apply(
    d_input,  # type: Any
    lookup={},  # type: dict
    support_import=False,  # type: bool
    default_module="__main__",  # type: str
    forcing_module="",  # type: str
    obj_key="obj",  # type: str
):
    # type: (...) -> Any

    d = d_input
    s = d_input

    if isinstance(d_input, dict):

        obj_str = d_input.get(obj_key)

        d, s = {}, {}
        for k, v in iteritems(d_input):
            d[k], s[k] = _dict_apply(
                v,
                lookup=lookup,
                support_import=support_import,
                default_module=default_module,
                forcing_module=forcing_module,
                obj_key=obj_key,
            )

        if obj_str:
            if obj_str in lookup:
                a = lookup.get(obj_str)
                d = _hatch(d, a, obj_key=obj_key, dummy_key="_")
            elif support_import:
                if forcing_module:
                    obj_str = "{}.{}".format(forcing_module, obj_str.rsplit(".", 1)[-1])
                a = load_obj(obj_str, default_obj_path=default_module)
                d = _hatch(d, a, obj_key=obj_key, dummy_key="_")

    if isinstance(d_input, list):

        d, s = [], []
        for v in d_input:
            _d, _s = _dict_apply(
                v,
                lookup=lookup,
                support_import=support_import,
                default_module=default_module,
                obj_key=obj_key,
            )
            d.append(_d)
            s.append(_s)

    return d, s


def _hatch(
    d,  # type: dict
    a,  # type: Any
    obj_key="obj",  # type: str
    dummy_key="_",
):
    d.pop(obj_key)
    if d:
        assert callable(a)
        d.pop(dummy_key, None)
        for k in d:
            assert isinstance(
                k, str
            ), "Non-string key '{}' in '{}' is not valid for callable: '{}'.".format(
                k, d, a.__name__
            )
        d = a(**d)
    else:
        d = a
    return d


def dot_flatten(d):
    def dot_reducer(k1, k2):
        return k1 + "." + k2 if k1 else k2

    return flatten(d, reducer=dot_reducer)


def _builtin_funcs():
    return dict(
        pass_=lambda *args, **kwargs: None,
        pass_through=lambda *args, **kwargs: (
            args[0] if args else list(kwargs.values())[0] if kwargs else None
        ),
        abs=lambda arg: abs(arg),
        all=lambda arg: all(arg),
        any=lambda arg: any(arg),
        bin=lambda arg: bin(arg),
        bool=lambda arg: bool(arg),
        bytearray=lambda arg: bytearray(arg),
        bytes=lambda arg: bytes(arg),
        callable=lambda arg: callable(arg),
        chr=lambda arg: chr(arg),
        classmethod=lambda arg: classmethod(arg),
        compile=lambda arg: compile(*arg),
        complex=lambda arg: complex(arg),
        delattr=lambda arg: delattr(*arg),
        dict=lambda arg: dict(arg),
        dir=lambda arg: dir(arg),
        divmod=lambda arg: divmod(*arg),
        enumerate=lambda arg: enumerate(arg),
        eval=lambda arg: eval(arg),
        exec=lambda arg: exec(arg),
        filter=lambda arg: filter(*arg),
        float=lambda arg: float(arg),
        format=lambda arg: format(arg),
        frozenset=lambda arg: frozenset(arg),
        getattr=lambda arg: getattr(*arg),
        globals=lambda arg: globals(),
        hasattr=lambda arg: hasattr(*arg),
        hash=lambda arg: hash(arg),
        help=lambda arg: help(arg),
        hex=lambda arg: hex(arg),
        id=lambda arg: id(arg),
        input=lambda arg: input(arg),
        int=lambda arg: int(arg),
        isinstance=lambda arg: isinstance(*arg),
        issubclass=lambda arg: issubclass(*arg),
        iter=lambda arg: iter(arg),
        len=lambda arg: len(arg),
        list=lambda arg: list(arg),
        locals=lambda arg: locals(),
        map=lambda arg: map(*arg),
        max=lambda arg: max(arg),
        memoryview=lambda arg: memoryview(arg),
        min=lambda arg: min(arg),
        next=lambda arg: next(arg),
        object=lambda arg: object(),
        oct=lambda arg: oct(arg),
        open=lambda arg: open(arg),
        ord=lambda arg: ord(arg),
        pow=lambda arg: pow(*arg),
        print=lambda arg: print(arg),
        property=lambda arg: property(arg),
        range=lambda arg: range(arg),
        repr=lambda arg: repr(arg),
        reversed=lambda arg: reversed(arg),
        round=lambda arg: round(arg),
        set=lambda arg: set(arg),
        setattr=lambda arg: setattr(*arg),
        slice=lambda arg: slice(arg),
        sorted=lambda arg: sorted(arg),
        str=lambda arg: str(arg),
        sum=lambda arg: sum(arg),
        super=lambda arg: super(arg),
        tuple=lambda arg: tuple(arg),
        type=lambda arg: type(arg),
        vars=lambda arg: vars(arg),
        zip=lambda arg: zip(arg),
        # abs=lambda arg: operator.abs(*arg),
        add=lambda arg: operator.add(*arg),
        and_=lambda arg: operator.and_(*arg),
        attrgetter=lambda arg: operator.attrgetter(*arg),
        concat=lambda arg: operator.concat(*arg),
        contains=lambda arg: operator.contains(*arg),
        countOf=lambda arg: operator.countOf(*arg),
        delitem=lambda arg: operator.delitem(*arg),
        eq=lambda arg: operator.eq(*arg),
        floordiv=lambda arg: operator.floordiv(*arg),
        ge=lambda arg: operator.ge(*arg),
        getitem=lambda arg: operator.getitem(*arg),
        gt=lambda arg: operator.gt(*arg),
        iadd=lambda arg: operator.iadd(*arg),
        iand=lambda arg: operator.iand(*arg),
        iconcat=lambda arg: operator.iconcat(*arg),
        ifloordiv=lambda arg: operator.ifloordiv(*arg),
        ilshift=lambda arg: operator.ilshift(*arg),
        imatmul=lambda arg: operator.imatmul(*arg),
        imod=lambda arg: operator.imod(*arg),
        imul=lambda arg: operator.imul(*arg),
        index=lambda arg: operator.index(*arg),
        indexOf=lambda arg: operator.indexOf(*arg),
        inv=lambda arg: operator.inv(*arg),
        invert=lambda arg: operator.invert(*arg),
        ior=lambda arg: operator.ior(*arg),
        ipow=lambda arg: operator.ipow(*arg),
        irshift=lambda arg: operator.irshift(*arg),
        is_=lambda arg: operator.is_(*arg),
        is_not=lambda arg: operator.is_not(*arg),
        isub=lambda arg: operator.isub(*arg),
        itemgetter=lambda arg: operator.itemgetter(*arg),
        itruediv=lambda arg: operator.itruediv(*arg),
        ixor=lambda arg: operator.ixor(*arg),
        le=lambda arg: operator.le(*arg),
        length_hint=lambda arg: operator.length_hint(*arg),
        lshift=lambda arg: operator.lshift(*arg),
        lt=lambda arg: operator.lt(*arg),
        matmul=lambda arg: operator.matmul(*arg),
        methodcaller=lambda arg: operator.methodcaller(*arg),
        mod=lambda arg: operator.mod(*arg),
        mul=lambda arg: operator.mul(*arg),
        ne=lambda arg: operator.ne(*arg),
        neg=lambda arg: operator.neg(*arg),
        not_=lambda arg: operator.not_(*arg),
        or_=lambda arg: operator.or_(*arg),
        pos=lambda arg: operator.pos(*arg),
        # pow=lambda arg: operator.pow(*arg),
        rshift=lambda arg: operator.rshift(*arg),
        setitem=lambda arg: operator.setitem(*arg),
        sub=lambda arg: operator.sub(*arg),
        truediv=lambda arg: operator.truediv(*arg),
        truth=lambda arg: operator.truth(*arg),
        xor=lambda arg: operator.xor(*arg),
    )
