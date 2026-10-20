"""実際に上流を通過したワークから学習行を作り、v005へ渡します。

物理oracleはこの検証側だけで使用します。r002とv005へ正解関数を渡す処理は
ありません。内部状態までラベルにできる合成検証と、現場の観測を区別します。
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def physical_contract():
    """新ラインの接続定義と工程定義を一か所から取得します。"""
    from ...electrical_interconnect.pipeline import line_config
    from ...electrical_interconnect.stages import STAGE_MANIFESTS, evaluate_stage
    return line_config(), STAGE_MANIFESTS, evaluate_stage


def sample_controls(axes: dict, count: int, rng: np.random.Generator) -> dict:
    """直積を作らず、各軸の水準が均等になるよう列ごとに抽出します。"""
    if count < 1:
        raise ValueError("サンプル数は1以上です。")
    result = {}
    for name, axis in axes.items():
        levels = np.asarray(axis, dtype=float)
        indices = np.arange(count) % len(levels)
        rng.shuffle(indices)
        result[name] = levels[indices]
    return result


def stage_inputs(stage: str, manifest: dict, config: dict, controls: dict, outputs: dict) -> dict:
    """工程操作と、同じワーク系譜の上流出力を結合します。"""
    inputs = {name: controls[f"{stage}.{name}"] for name in manifest["controls"]}
    edges = {edge["to"]: edge["from"] for edge in config["connections"]}
    for name in manifest["incoming_context"]:
        qualified = f"{stage}.{name}"
        if qualified in edges:
            parent, source = edges[qualified].split(".")
            inputs[name] = outputs[parent][source]
        else:
            inputs[name] = config["external_context"][qualified]
    return inputs


def local_margin(outputs: dict, manifest: dict):
    """必須の工程制約を一つでも外れたワークは、次工程へ払い出しません。"""
    from r002.src.pipeline import specification_margin
    terms = [specification_margin(np.asarray(outputs[spec["name"]]), spec)
             for spec in manifest.get("local_constraints", [])]
    if not terms:
        return np.zeros_like(np.asarray(next(iter(outputs.values()))), dtype=float)
    return np.minimum.reduce(terms)


def generate_workpieces(count: int, seed: int = 0, levels: int = 4, *, stop_after: str | None = None,
                        selected_controls: dict | None = None, external_context: dict | None = None) -> dict:
    """切断→成形→二材組立→溶接の実施可能な履歴だけを残します。

    行番号が同じでも銅片とアルミ片は別IDであり、組立時にpair IDを付けます。
    同一部品を実験候補の複数行へ無制限に複製しないよう、払い出しも記録します。
    """
    from ...electrical_interconnect.pipeline import axes
    config, manifests, evaluate = physical_contract()
    if external_context is not None:
        if set(external_context) != set(config["external_context"]):
            raise ValueError("材料状態は全external_contextと一致させてください。")
        config["external_context"] = dict(external_context)
    control_axes = axes(levels)
    if selected_controls is None:
        controls = sample_controls(control_axes, count, np.random.default_rng(seed))
    else:
        if set(selected_controls) != set(control_axes):
            raise ValueError("製作条件は全stage_id.controlを指定してください。")
        controls = {name: np.broadcast_to(np.asarray(value, dtype=float), (count,))
                    for name, value in selected_controls.items()}
    outputs, passing, blocks = {}, {}, {}
    for stage, manifest in manifests.items():
        raw_inputs = stage_inputs(stage, manifest, config, controls, outputs)
        inputs = {name: np.broadcast_to(np.asarray(value, dtype=float), (count,))
                  for name, value in raw_inputs.items()}
        parents = {edge["from"].split(".")[0] for edge in config["connections"]
                   if edge["to"].split(".")[0] == stage}
        admitted = np.ones(count, dtype=bool)
        for parent in parents:
            admitted &= passing[parent]
        for name, values in inputs.items():
            lo, hi = manifest["input_bounds"][name]
            admitted &= np.isfinite(values) & (values >= lo) & (values <= hi)
        # 下流の入力範囲外をclipすると架空の良品になるため、行ごと除外します。
        measured = evaluate(stage, {name: values[admitted] for name, values in inputs.items()}) if admitted.any() else {}
        outputs[stage] = {name: np.full(count, np.nan) for name in manifest["predicted_outputs"]}
        for name, values in measured.items():
            outputs[stage][name][admitted] = values
        margin = local_margin(outputs[stage], manifest)
        passing[stage] = admitted & (margin >= 0)
        rows = np.flatnonzero(admitted)
        blocks[stage] = {
            "inputs": {name: value[admitted] for name, value in inputs.items()},
            "outputs": {name: value[admitted] for name, value in outputs[stage].items()},
            "row_indices": rows,
            "workpiece_ids": [f"s{seed}-{'pair' if stage in ('assembly', 'welding') else stage.split('_')[-1]}-{row:06d}" for row in rows],
            "lot_id": f"synthetic-lot-{seed}",
            "admitted_count": int(admitted.sum()), "passed_count": int(passing[stage].sum()),
            "observation_regime": "PRIVILEGED_SYNTHETIC_LABELS",
        }
        if stage == stop_after:
            break
    return {"blocks": blocks, "controls": controls, "stage_outputs": outputs,
            "passing": passing, "requested_count": count, "seed": seed,
            "external_context": dict(config["external_context"])}


def train_bundles(dataset: dict, destination: Path, max_training_samples: int = 256) -> tuple[dict, dict]:
    """観測データ配列だけで6モデルを学習し、読込後の予測器をr002へ渡します。"""
    from v005.src.stage_bundle import load_stage_bundle, train_stage_bundle
    config, manifests, _ = physical_contract()
    config["external_context"] = dict(dataset["external_context"])
    predictors = {}
    for stage, original in manifests.items():
        block = dataset["blocks"][stage]
        if block["admitted_count"] < 8:
            raise ValueError(f"{stage}の実施可能ワークが不足しています: {block['admitted_count']}")
        manifest = copy.deepcopy(original)
        manifest["bundle_schema_version"] = "2.0"
        manifest["provenance"] = {"source": "synthetic_achievable_workpieces",
                                  "observation_regime": block["observation_regime"],
                                  "lot_id": block["lot_id"],
                                  "retention_policy": block.get("retention_policy", "recent_plus_random")}
        path = train_stage_bundle(destination / stage, manifest, block["inputs"], block["outputs"],
                                  seed=dataset["seed"], max_training_samples=max_training_samples)
        predictors[stage] = load_stage_bundle(path)
        for stage_row in config["stages"]:
            if stage_row["id"] == stage:
                stage_row["bundle"] = str(path.resolve())
    return config, predictors


def available_inventory(dataset: dict, maximum: int = 8) -> list:
    """組立を通った未溶接ワークを、溶接の部分実験用に払い出します。

    この検証では学習用履歴と別ロットで候補を用意します。切断・成形は既に
    消費された扱いにし、同じワークを他工程へ二重に割り当てません。
    """
    if "welding" in dataset["blocks"]:
        raise ValueError("在庫はstop_after='assembly'で製作した未溶接ワークから作ってください。")
    eligible = np.flatnonzero(dataset["passing"]["assembly"])[:maximum]
    inventory = []
    for row in eligible:
        inventory.append({"workpiece_id": f"inventory-{dataset['seed']}-pair-{row:06d}",
                          "stage_id": "assembly", "lot_id": f"inventory-lot-{dataset['seed']}",
                          "available": True,
                          "outputs": {name: float(values[row]) for name, values in dataset["stage_outputs"]["assembly"].items()},
                          "observation_regime": "PRIVILEGED_SYNTHETIC_LABELS"})
    return inventory
