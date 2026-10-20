"""Limited, useful batches: improvement candidates and one uncertainty probe.

No extra model fitting or full-grid sorting. Scores are conditional acquisition
values, not calibrated guarantees of feasibility or information gain.
"""
from dataclasses import dataclass
import numpy as np
from ..search_config import SearchConfig


@dataclass(frozen=True)
class Selection:
    indices: list[int]
    roles: list[str]
    useful_count: int
    max_score: float
    reduction_reason: str


def select_efficient_candidates(x, score, feasibility, improvement, objective_scale,
                                distance, uncertainty, support, count, config: SearchConfig,
                                *, has_feasible_observation, diversity_weight):
    count=min(count,len(score))
    finite=np.isfinite(score)&np.isfinite(feasibility)&np.isfinite(improvement)
    scale=max(float(objective_scale),1e-12)
    max_score=float(np.max(score[finite])) if np.any(finite) else 0.0
    if not has_feasible_observation:
        # Feasibility search must not require a known feasible baseline.
        pool=finite.copy();roles_name='FEASIBILITY_SEARCH'
        limit=count
    else:
        worthwhile=finite & (score/scale >= config.min_normalized_improvement)
        worthwhile &= score >= max_score*config.min_relative_score
        pool=worthwhile & (feasibility>=config.min_recommendation_feasibility)
        roles_name='IMPROVEMENT';limit=count
    selected=[];roles=[];nearest=np.asarray(distance,dtype=float).copy()
    normalized=np.clip(score/max(max_score,1e-300),0,1)
    # Normalize using varying dimensions only; fixed incoming columns contribute
    # no distance and must not dilute the usefulness of diversity.
    dimensions=max(1,int(np.sum(np.ptp(x,axis=0)>1e-12)))
    maximum_distance=np.sqrt(dimensions)
    utility=np.ones(len(score))

    def choose(pool, role, number, information=False):
        available=pool.copy()
        for index in selected:available[index]=False
        for _ in range(number):
            if not np.any(available):break
            if selected:
                np.power(normalized,1-diversity_weight,out=utility)
                utility[:] *= np.power(np.clip(nearest/maximum_distance,0,1),diversity_weight)
            else:utility[:]=normalized
            if information:
                # Actual uncertainty, low support and a new location must justify
                # an information slot; merely being far away is insufficient.
                signal=np.clip(uncertainty/scale,0,1)*(1-np.clip(support,0,1))
                utility[:] *= signal
                available &= (signal>1e-6)&(nearest>=0.08)
            eligible=np.where(available,utility,-1.0)
            index=int(np.argmax(eligible))
            if eligible[index]<0:break
            selected.append(index);roles.append(role);available[index]=False
            delta=x-x[index]
            nearest[:]=np.minimum(nearest,np.sqrt(np.einsum('ij,ij->i',delta,delta)))

    # A small information allowance preserves boundary learning without forcing
    # four nearly impossible conditions into every batch.
    if has_feasible_observation:
        info=worthwhile & (feasibility>=config.min_information_feasibility)
        info &= ~pool
        # Always preserve an improvement slot, including --n 1 / last budget 1.
        information_limit=max(0,count-1) if np.any(pool) else count
        info_count=min(config.max_information_candidates,information_limit,int(np.sum(info)))
        choose(pool,roles_name,max(0,limit-info_count))
        choose(info,'INFORMATION',min(info_count,count-len(selected)),information=True)
        choose(pool,roles_name,count-len(selected))
        useful=int(np.sum(pool|info))
    else:
        choose(pool,roles_name,limit);useful=int(np.sum(pool))
    reason=''
    if len(selected)<count:
        reason='改善価値・合格確率・情報収集価値の基準を満たす未測定候補が少ないため推薦件数を縮小'
    return Selection(selected,roles,useful,max_score,reason)
