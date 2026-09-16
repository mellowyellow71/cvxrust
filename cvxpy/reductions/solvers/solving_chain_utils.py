from cvxpy.settings import (
    COO_CANON_BACKEND,
    CPP_CANON_BACKEND,
    RUST_CANON_BACKEND,
    SCIPY_CANON_BACKEND,
)
from cvxpy.utilities.warn import warn

# Backends that lower every LinOp canonicalization emits, including the N-D and
# broadcast expressions the C++ core rejects.
_FULL_COVERAGE_BACKENDS = frozenset(
    {SCIPY_CANON_BACKEND, COO_CANON_BACKEND, RUST_CANON_BACKEND}
)


def resolve_default_canon_backend() -> str:
    """The backend used when the caller passes ``canon_backend=None``.

    Honors the ``CVXPY_DEFAULT_CANON_BACKEND`` environment variable, then
    ``settings.DEFAULT_CANON_BACKEND``.
    """
    # Local import: canonInterface pulls in the whole backend registry.
    from cvxpy.cvxcore.python.canonInterface import get_default_canon_backend
    return get_default_canon_backend()


def get_canon_backend(problem, canon_backend: str) -> str:
    """
    Resolve the canonicalization backend for ``problem``.

    When no backend is requested and the default is a full-coverage backend
    (SCIPY, COO or RUST) it is returned as is. Otherwise, if the problem has
    expressions of dimension greater than 2 or lacks C++ support, this warns
    and falls back to SCIPY, or raises if 'CPP' was requested explicitly.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    canon_backend : str
        'CPP' (default) | 'SCIPY'
        Specifies which backend to use for canonicalization, which can affect
        compilation time. Defaults to None, i.e., selecting the default
        backend.
    Returns
    -------
    canon_backend : str
        The canonicalization backend to use.
    """

    if canon_backend is None:
        default = resolve_default_canon_backend()
        if default in _FULL_COVERAGE_BACKENDS:
            # Nothing to fall back from: the default handles every expression.
            return default

    if not problem._supports_cpp():
        if canon_backend is None:
            warn(
                f"The problem includes expressions that don't support {CPP_CANON_BACKEND} backend. "
                f"Defaulting to the {SCIPY_CANON_BACKEND} backend for canonicalization.")
            return SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(f"The {CPP_CANON_BACKEND} backend cannot be used with problems "
                             f"that have expressions which do not support it.")
        return canon_backend  # Use the specified backend (e.g., COO_CANON_BACKEND)

    if problem._max_ndim() > 2:
        if canon_backend is None:
            warn(
                f"The problem has an expression with dimension greater than 2. "
                f"Defaulting to the {SCIPY_CANON_BACKEND} backend for canonicalization.")
            return SCIPY_CANON_BACKEND
        if canon_backend == CPP_CANON_BACKEND:
            raise ValueError(
                f"Only the {COO_CANON_BACKEND}, {RUST_CANON_BACKEND}, and "
                f"{SCIPY_CANON_BACKEND} backends are supported for problems "
                f"with expressions of dimension greater than 2."
            )
    return canon_backend
