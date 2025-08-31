"""vf-eval entry module for the Shop‑R1 environment.

This thin wrapper exposes a top-level module name matching the env_id
expected by verifiers' loader (hyphens -> underscores).
"""

from environments.shop_r1.shop_r1 import load_environment as _load_environment


def load_environment(**kwargs):
    return _load_environment(**kwargs)

