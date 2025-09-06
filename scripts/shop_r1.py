"""vf-eval entry module for the Shop‑R1 environment.

Exposes a top-level module name matching the env_id 'shop-r1'
so verifiers can import `shop_r1.load_environment`.
"""

from environments.shop_r1.shop_r1 import load_environment as _load_environment


def load_environment(**kwargs):
    return _load_environment(**kwargs)

