"""Root conftest: compatibility patches applied before test collection."""

import sys

# The test suite is written for 64-bit precision: ~26 test modules enable x64 at
# module level and several core tests require it (64-bit ``result_shape_dtypes``,
# float64 dtypes). Setting it here in the root conftest -- before any test module
# is collected, in every ``pytest-xdist`` worker -- makes the global flag
# deterministic and avoids order-dependent x64 state leaking across modules under
# ``pytest -n auto`` (otherwise a worker may run an x64-needing test before any
# x64-enabling module is imported, raising e.g. "result_shape_dtypes cannot
# specify 64-bit types when jax_enable_x64 is disabled").
import jax

jax.config.update("jax_enable_x64", True)

# typeguard 2.13.3 calls evaluate_forwardref(ref, globalns, localns, frozenset())
# passing recursive_guard positionally.  Python 3.12 made recursive_guard a
# keyword-only argument, so that call raises TypeError.  We patch the
# module-level evaluate_forwardref that resolve_forwardref() looks up each
# call so that the positional frozenset() is moved into the keyword slot.
if sys.version_info >= (3, 12):
    import typeguard as _tg

    _orig_evaluate_forwardref = _tg.evaluate_forwardref

    def _patched_evaluate_forwardref(ref, globalns, localns, *args, **kwargs):
        # args[0] is the old positional recursive_guard (a frozenset)
        if args and "recursive_guard" not in kwargs:
            kwargs["recursive_guard"] = args[0]
            args = args[1:]
        if "recursive_guard" not in kwargs:
            kwargs["recursive_guard"] = frozenset()
        return _orig_evaluate_forwardref(ref, globalns, localns, *args, **kwargs)

    _tg.evaluate_forwardref = _patched_evaluate_forwardref
