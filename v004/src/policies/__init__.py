"""実験候補へ適用する知識ベースの方針。

``rules`` はCSVの形式と判定ロジックを、``regions`` は公開APIを提供します。
"""

from .regions import (
    KIND_FORBIDDEN,
    KIND_PREFERRED,
    PolicyResult,
    RegionCondition,
    RegionPolicy,
    RegionRule,
    apply_preferred_range_bonus,
    filter_experiment_candidates,
    load_search_regions,
    write_search_regions_template,
)

__all__ = [
    "KIND_FORBIDDEN",
    "KIND_PREFERRED",
    "PolicyResult",
    "RegionCondition",
    "RegionPolicy",
    "RegionRule",
    "apply_preferred_range_bonus",
    "filter_experiment_candidates",
    "load_search_regions",
    "write_search_regions_template",
]
