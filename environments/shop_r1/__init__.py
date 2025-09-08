"""Shop-R1 environment module.

Exports `load_environment` and `load_multiturn_environment` for discovery.
"""

from .shop_r1 import load_environment  # noqa: F401
try:  # MultiTurn optional depending on verifiers version
    from .shop_r1 import load_multiturn_environment  # noqa: F401
except Exception:  # pragma: no cover
    pass
