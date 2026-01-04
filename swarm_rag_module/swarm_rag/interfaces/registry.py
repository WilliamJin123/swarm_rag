from __future__ import annotations

import functools
import types
from collections.abc import Callable
from enum import Enum
from typing import Any, ClassVar, Dict, Generic, overload, ParamSpec, TypeVar, cast

from .enums import GeneticKey, HeuristicKey

P = ParamSpec("P")                     # Parameter types of the registered callables
R = TypeVar("R")                       # Return type of the registered callables
K = TypeVar("K", bound=Enum)           # Enum used as key (HeuristicKey / GeneticKey)

class _BaseRegistry(Generic[K, P, R]):
    """
    A tiny, fully typed registry that can be used both as a decorator
    and as a plain function call.

    Concrete subclasses only have to provide a class‑level mapping
    ('_registry') and, optionally, a convenience 'all' method.
    """

    # The concrete subclass must create a dict that lives on the class.
    _registry: ClassVar[Dict[K | str, Callable[P, R]]] = {}
    _enum_type: type[Enum] | None = None

    @overload
    @classmethod
    def register(cls, key: K) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    @overload
    @classmethod
    def register(cls, fn: Callable[P, R]) -> Callable[P, R]: ...

    @overload
    @classmethod
    def register(cls, key: K, fn: Callable[P, R]) -> Callable[P, R]: ...

    @classmethod
    def register(cls, *args: Any, **kwargs: Any) -> Any:          # pragma: no‑cover
        """
        Unified registration API.

        Supported call styles:
        ------------------------------------------------------------
        1 @MyRegistry.register(MyEnum.FOO)          # decorator with explicit key
        2 @MyRegistry.register                       # decorator, key == fn.__name__
        3 MyRegistry.register(MyEnum.FOO, fn)       # direct call, explicit key
        4 MyRegistry.register(fn)                  # direct call, inferred key
        ------------------------------------------------------------
        The function (or the wrapper produced by 'functools.wraps') is
        stored under the supplied key (or the inferred one). The original
        callable is returned unchanged so that the decorator can be used
        transparently.
        """
        # Helper: store the function under 'key' and return it.
        def _store(key: K | str, fn: Callable[P, R]) -> Callable[P, R]:
            if key in cls._registry:
                raise KeyError(f"{cls.__name__}: key {key!r} already registered.")
            cls._registry[cast(K | str, key)] = fn
            return fn

        # Resolve which overload we are in.
        if len(args) == 1 and callable(args[0]) and not kwargs:
            # @MyRegistry.register   (fn only, no key)
            fn = cast(Callable[P, R], args[0])
            key = fn.__name__
            return _store(key, fn)

        if len(args) == 2 and not kwargs:
            # MyRegistry.register(key, fn)
            key, fn = args  # type: ignore[assignment]
            return _store(key, fn)

        if len(args) == 1 and not kwargs:
            # @MyRegistry.register(key)  → returns decorator
            key = args[0]

            def decorator(fn: Callable[P, R]) -> Callable[P, R]:
                return _store(key, fn)

            return decorator

        # Anything else is a programming error – give a clear message.
        raise TypeError(
            f"{cls.__name__}.register() received an unsupported signature: "
            f"args={args}, kwargs={kwargs!r}"
        )
    
    @classmethod
    def _normalise_key(cls, key: K | str) -> K | str:
        """
        Turn 'key' into the form stored inside '_registry'.

        * If 'key' is already an enum member → return it unchanged.
        * If 'key' is a string that equals the 'value' of any enum member
          of type 'K' → return that enum member (canonical form).
        * Otherwise return the string unchanged – this covers custom
          user-defined keys that are not part of the enum.
        """
        if isinstance(key, Enum):
            return key

        if cls._enum_type is not None:
            try:
                # Example: HeuristicKey('semantic_similarity')
                return cls._enum_type(key)   # type: ignore[call-arg]
            except Exception:
                pass                     # not a built‑in enum value → keep the string

        return key

    @classmethod
    def get(cls, key: K | str) -> Callable[P, R]:
        """Return the callable registered under 'key' (or raise KeyError)."""
        normalised = cls._normalise_key(key)
        try:
            return cls._registry[normalised]
        except KeyError as exc:
            # Build a helpful error message that lists *all* valid keys.
            allowed = sorted(
                (k.value if isinstance(k, Enum) else k) for k in cls._registry.keys()
            )
            raise KeyError(
                f"{cls.__name__}.get: key {key!r} not found. "
                f"Available keys: {allowed}"
            ) from exc

    @classmethod
    def all(cls) -> Dict[K | str, Callable[P, R]]:
        """A shallow copy of the internal dict callers can read but not modify."""
        return dict(cls._registry)
    

class _SelectionRegistry(_BaseRegistry["GeneticKey", P, R]):
    _registry: ClassVar[Dict["GeneticKey" | str, Callable[P, R]]] = {}
    _enum_type = GeneticKey   

class _CrossoverRegistry(_BaseRegistry["GeneticKey", P, R]):
    _registry: ClassVar[Dict["GeneticKey" | str, Callable[P, R]]] = {}
    _enum_type = GeneticKey   

class _MutationRegistry(_BaseRegistry["GeneticKey", P, R]):
    _registry: ClassVar[Dict["GeneticKey" | str, Callable[P, R]]] = {}
    _enum_type = GeneticKey   

class _MovementRegistry(_BaseRegistry["HeuristicKey", P, R]):
    _registry: ClassVar[Dict["HeuristicKey" | str, Callable[P, R]]] = {}
    _enum_type = HeuristicKey   

class _RankingRegistry(_BaseRegistry["HeuristicKey", P, R]):
    _registry: ClassVar[Dict["HeuristicKey" | str, Callable[P, R]]] = {}
    _enum_type = HeuristicKey   

class _DepositRegistry(_BaseRegistry["HeuristicKey", P, R]):
    _registry: ClassVar[Dict["HeuristicKey" | str, Callable[P, R]]] = {}
    _enum_type = HeuristicKey   