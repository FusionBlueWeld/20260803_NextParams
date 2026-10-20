"""電気接続の物理式と検査モデルをGUIの入力・出力定義へ変換します。

学習済み模型は使わず、操作条件と流入状態から各工程の物理式を直接評価します。
"""

from __future__ import annotations

from .gui_contract import FieldSpec, SimulatorSpec


STAGE_LABELS = {"cut_cu": "銅の切り出し", "cut_al": "アルミの切り出し",
                "form_cu": "銅の成形", "form_al": "アルミの成形",
                "assembly": "組立", "welding": "レーザー溶接"}
FIELD_LABELS = {
    "cut_speed_m_min": "切断速度", "tooth_feed_mm": "一刃送り", "clamp_force_n": "保持力",
    "finish_allowance_mm": "仕上げ取り代", "finish_feed_mm_rev": "仕上げ送り",
    "deburr_depth_mm": "バリ取り深さ", "length_offset_mm": "長さ補正",
    "first_overbend_deg": "1曲げ目の補正角", "second_overbend_deg": "2曲げ目の補正角",
    "punch_radius_mm": "パンチ半径", "die_opening_mm": "ダイ開口", "holding_force_n": "押さえ力",
    "ram_speed_mm_s": "ラム速度", "coining_pressure_mpa": "コイニング圧力",
    "x_correction_mm": "X位置補正", "y_correction_mm": "Y位置補正", "rotation_correction_deg": "回転補正",
    "bottom_clamp_force_n": "下材クランプ力", "top_clamp_force_n": "上材クランプ力",
    "clamp_span_mm": "クランプ間隔", "bottom_cleaning_dose": "下材清掃量", "top_cleaning_dose": "上材清掃量",
    "laser_power_kw": "レーザー出力", "travel_speed_m_min": "溶接送り", "focus_offset_mm": "焦点位置",
    "oscillation_amplitude_mm": "振動振幅", "oscillation_frequency_hz": "振動周波数",
    "seam_length_mm": "溶接長", "seam_offset_mm": "照射位置補正", "ramp_length_mm": "出力立上げ長",
    "stock_thickness_mm": "原材板厚", "stock_yield_mpa": "原材の降伏強さ", "elastic_modulus_gpa": "弾性率",
    "material_is_cu": "材種（1=銅、0=アルミ）", "stock_camber_mm": "原材の反り",
    "stock_oxide_nm": "原材の酸化膜厚", "stock_contamination_fraction": "原材の汚染率",
    "thickness_mm": "板厚", "length_mm": "長さ", "width_mm": "幅", "burr_mm": "バリ高さ",
    "roughness_um": "表面粗さ", "edge_damage_fraction": "切断端の損傷率",
    "edge_hardening_fraction": "切断端の硬化率", "camber_mm": "反り", "oxide_nm": "酸化膜厚",
    "contamination_fraction": "汚染率", "yield_strength_mpa": "加工後の降伏強さ",
    "first_angle_error_deg": "1曲げ目の角度誤差", "second_angle_error_deg": "2曲げ目の角度誤差",
    "angle_error_max_deg": "最大角度誤差", "pad_height_error_mm": "接合面の高さ誤差",
    "pad_slope_deg": "接合面の傾斜", "plastic_strain": "塑性ひずみ", "residual_stress_mpa": "残留応力",
    "form_damage_fraction": "成形損傷率", "damage_fraction": "損傷率", "thickness_loss_fraction": "減肉率",
    "gap_start_mm": "始点隙間", "gap_center_mm": "中央隙間", "gap_end_mm": "終点隙間", "gap_max_mm": "最大隙間",
    "contact_fraction": "接触率", "contact_pressure_mpa": "接触圧", "fixture_heat_loss_fraction": "治具放熱率",
    "surface_roughness_um": "接合面粗さ", "overlap_mm": "重なり長", "lateral_error_mm": "横ずれ",
    "assembly_plastic_set_mm": "組立永久変形", "assembly_stored_energy_mj": "組立弾性エネルギー",
    "line_energy_j_mm": "線エネルギー", "absorbed_energy_j_mm": "吸収エネルギー",
    "peak_intensity_kw_mm2": "ピーク強度", "top_penetration_mm": "上材融込み", "bottom_penetration_mm": "下材融込み",
    "joint_width_mm": "接合幅", "bonded_area_mm2": "接合面積", "mixing_fraction": "混合率",
    "imc_proxy_um": "IMC厚さの代理値", "porosity_fraction": "気孔率", "lack_of_fusion_fraction": "未接合率",
    "spatter_fraction": "スパッタ率", "base_path_resistance_uohm": "基材経路抵抗",
    "interface_resistance_uohm": "界面抵抗", "cold_unclamped_resistance_uohm": "冷却・治具解除後の抵抗",
    "shear_force_n": "せん断力", "temperature_rise_k": "温度上昇", "post_release_distortion_mm": "治具解除後の変形",
    "fusion_completion_fraction": "融合完了率", "cut_time_s": "切断時間", "form_time_s": "成形時間",
    "assembly_time_s": "組立時間", "weld_time_s": "溶接時間", "length_error_mm": "長さ誤差",
}


def field_label(name: str) -> str:
    """枝IDを除いた物理量名を翻訳します。未知の物理量は元の名前を保ちます。"""
    name = name.rsplit(".", 1)[-1]
    if name in FIELD_LABELS:
        return FIELD_LABELS[name]
    for prefix, label in (("bottom_", "下材"), ("top_", "上材")):
        if name.startswith(prefix):
            return label + "の" + FIELD_LABELS.get(name[len(prefix):], name[len(prefix):])
    return name


def _input(manifest: dict, name: str, group: str, *, qualified: str | None = None,
           default: float | None = None, readonly: bool = False) -> FieldSpec:
    lo, hi = manifest["input_bounds"][name]
    value = manifest.get("default_inputs", {}).get(name) if default is None else default
    categorical = manifest["input_units"][name] == "category"
    # 材種は連続値0.5を入力させず、選んだ工程・ラインの材種へ固定します。
    return FieldSpec(qualified or name, field_label(name), manifest["input_units"][name],
                     lo, hi, value, group, readonly or categorical)


def _output(manifest: dict, name: str, group: str, qualified: str | None = None) -> FieldSpec:
    policy = manifest.get("observation_policy", {})
    suffix = "［内部状態］" if name in policy.get("latent", []) else "［破壊検査］" if name in policy.get("destructive", []) else ""
    return FieldSpec(qualified or name, field_label(name) + suffix, manifest["output_units"][name], group=group)


def _line_diagram(variant: str):
    bottom, top = {"cu_al": ("銅", "アルミ"), "cu_cu": ("銅", "銅"), "al_al": ("アルミ", "アルミ")}[variant]
    nodes = (("cut_cu", bottom + "切り出し", .24, .16), ("cut_al", top + "切り出し", .76, .16),
             ("form_cu", bottom + "成形", .24, .38), ("form_al", top + "成形", .76, .38),
             ("assembly", "組立・隙間形成", .5, .61), ("welding", "レーザー溶接・検査", .5, .84))
    edges = (("cut_cu", "form_cu"), ("cut_al", "form_al"), ("form_cu", "assembly"),
             ("form_al", "assembly"), ("assembly", "welding"))
    return nodes, edges


def electrical_specs() -> list[SimulatorSpec]:
    """独立6工程＋3材質の連結ラインを、既存選択欄へ追加します。"""
    from .multistage.electrical_interconnect import pipeline
    from .multistage.electrical_interconnect.stages import STAGE_MANIFESTS, evaluate_stage
    from .multistage.electrical_interconnect.inspection import observe_stage
    specs = []
    for stage in pipeline.STAGE_ORDER:
        manifest = STAGE_MANIFESTS[stage]
        inputs = tuple([*(_input(manifest, name, "操作条件") for name in manifest["controls"]),
                        *(_input(manifest, name, "流入状態") for name in manifest["incoming_context"])])
        outputs = tuple([*(_output(manifest, name, "加工状態") for name in manifest["predicted_outputs"]),
                         FieldSpec("local_margin", "工程制約余裕（真値基準）", "1", group="工程合否"),
                         FieldSpec("local_feasible", "工程制約合格（真値基準）", "", group="工程合否")])

        def evaluate(values, sid=stage, definition=manifest):
            raw = evaluate_stage(sid, values)
            margin, passed = pipeline.constraint_margin(raw, definition["local_constraints"])
            return {**raw, "local_margin": margin, "local_feasible": passed}

        def observe(result, rng, sid=stage):
            inspected = observe_stage(sid, result, rng=rng, noise=True)
            return {**{name: None for name in result}, **inspected["measurements"],
                    "local_margin": result["local_margin"], "local_feasible": result["local_feasible"]}

        image = "milling" if stage.startswith("cut") else "press_forming" if stage.startswith("form") else "laser_welding" if stage == "welding" else None
        nodes = () if image else ((stage, STAGE_LABELS[stage], .5, .5),)
        specs.append(SimulatorSpec("interconnect_" + stage, "電気接続工程｜" + STAGE_LABELS[stage],
            inputs, outputs, evaluate, {}, description="未校正の合成物理モデル。ノイズありでは非破壊検査項目のみ表示し、内部状態・破壊検査値は未測定とします。合否は真値基準の参考です。",
            image_id=image, diagram_nodes=nodes, observe=observe, noise_label="検査ノイズあり（非破壊検査項目のみ）"))

    for variant, title in (("cu_al", "銅／アルミ"), ("cu_cu", "銅／銅"), ("al_al", "アルミ／アルミ")):
        material = pipeline.material_context(variant)
        inputs = []
        for stage in pipeline.STAGE_ORDER:
            manifest = STAGE_MANIFESTS[stage]
            group = ("下材" if stage.endswith("cu") else "上材") + ("切り出し" if stage.startswith("cut") else "成形") if stage.startswith(("cut", "form")) else STAGE_LABELS[stage]
            inputs.extend(_input(manifest, name, group + "｜操作", qualified=f"{stage}.{name}") for name in manifest["controls"])
        for key, value in material.items():
            stage, name = key.split(".")
            inputs.append(_input(STAGE_MANIFESTS[stage], name, "原材・ロット状態", qualified=key, default=value))
        outputs = [FieldSpec("line_feasible", "全工程・最終仕様合格（真値基準）", "", group="最終品質"),
                   FieldSpec("quality_margin", "最終仕様余裕（真値基準）", "1", group="最終品質")]
        # 同材variantの枝名は工程IDを維持し、表示には下材／上材を明示します。
        for stage in reversed(pipeline.STAGE_ORDER):
            manifest = STAGE_MANIFESTS[stage]
            group = "溶接後品質" if stage == "welding" else "組立状態" if stage == "assembly" else ("下材" if stage.endswith("cu") else "上材") + ("切断状態" if stage.startswith("cut") else "成形状態")
            outputs.extend(_output(manifest, name, group, f"{stage}.{name}") for name in manifest["predicted_outputs"])
            outputs.append(FieldSpec(f"{stage}.local_feasible", "工程制約合格（真値基準）", "", group=group))

        def evaluate(values, case=variant):
            controls = {name: value for name, value in values.items() if name in pipeline.DEFAULT_CONTROLS}
            material = {name: value for name, value in values.items() if name not in pipeline.DEFAULT_CONTROLS}
            result = pipeline.evaluate_line(controls, material_context=material, variant=case)
            flat = {f"{sid}.{name}": value for sid, row in result["stage_outputs"].items() for name, value in row.items()}
            flat.update({f"{sid}.local_feasible": value for sid, value in result["local_feasible"].items()})
            return {**flat, "line_feasible": result["final"]["line_feasible"], "quality_margin": result["final"]["quality_margin"]}

        def observe(result, rng):
            flat = {name: None for name in result}
            for sid in pipeline.STAGE_ORDER:
                raw = {name: result[f"{sid}.{name}"] for name in STAGE_MANIFESTS[sid]["predicted_outputs"]}
                measured = observe_stage(sid, raw, rng=rng, noise=True)["measurements"]
                flat.update({f"{sid}.{name}": value for name, value in measured.items()})
                flat[f"{sid}.local_feasible"] = result[f"{sid}.local_feasible"]
            flat.update(line_feasible=result["line_feasible"], quality_margin=result["quality_margin"])
            return flat

        nodes, edges = _line_diagram(variant)
        specs.append(SimulatorSpec("interconnect_" + variant, "複合工程全体｜電気接続 " + title,
            tuple(inputs), tuple(outputs), evaluate, {}, description="44操作と原材状態から6工程を接続。右欄に工程ごとの状態を表示します。ノイズ時の合否は真値基準の参考で、検査判定ではありません。",
            diagram_nodes=nodes, diagram_edges=edges, observe=observe,
            noise_label="検査ノイズあり（非破壊検査項目のみ）"))
    return specs
