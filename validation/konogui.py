"""validation物理シミュレータを手入力で評価するシンプルなTkinter GUI。

作業フォルダ直下から次のように起動します。

    python validation/konogui.py

画面を開かず全シミュレータの中点入力を確認する場合:

    python validation/konogui.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import queue
import threading
from pathlib import Path
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Mapping

import numpy as np


VALIDATION_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = VALIDATION_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_ROOT = VALIDATION_ROOT / "gui_img"
MEASUREMENT_NOISE_FRACTION = 0.02
PROCESS_DURATION_MS = 5000
ANIMATION_INTERVAL_MS = 50


from validation.gui_contract import FieldSpec, SimulatorSpec
from validation.gui_interconnect import electrical_specs


def evaluate_display(spec, values, noise, rng):
    """検査可能な値と内部真値を分け、合否フラグへ数値ノイズを加えません。"""
    result = dict(spec.evaluate(values))
    if noise and spec.allow_noise:
        if spec.observe is not None:
            return dict(spec.observe(result, rng))
        for name, scale in spec.noise_scales.items():
            if name in result and np.asarray(result[name]).dtype.kind in "fiu":
                result[name] = np.asarray(result[name]) + rng.normal(
                    0, MEASUREMENT_NOISE_FRACTION * scale, size=np.shape(result[name]))
    return result


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar(value: object) -> float | bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("GUIでは1条件ずつ評価してください。")
    item = array.reshape(-1)[0]
    if isinstance(item, (np.bool_, bool)):
        return bool(item)
    return float(item)


def _single_specs() -> list[SimulatorSpec]:
    from validation.single.src.simulators import available, load

    specs: list[SimulatorSpec] = []
    preferred_order = [
        "laser_welding",
        "milling",
        "press_forming",
        "thermal_curing",
        "convection_drying",
        "electroplating",
    ]
    known = set(available())
    for simulator_id in [*preferred_order, *sorted(known - set(preferred_order))]:
        if simulator_id not in known:
            continue
        simulator = load(simulator_id)
        inputs = tuple(
            FieldSpec(
                item.column,
                item.display_name or item.column,
                item.unit,
                item.lower,
                item.upper,
            )
            for item in simulator.parameters
        )
        outputs = tuple(
            FieldSpec(item.column, item.display_name or item.column, item.unit)
            for item in simulator.outputs
        )

        def evaluate(values: Mapping[str, float], current=simulator) -> Mapping[str, object]:
            result = current.evaluate(values)
            return {item.column: result[item.column] for item in current.outputs}

        grid_truth = simulator.evaluate_points(simulator.grid())
        noise_scales = {
            item.column: max(float(np.ptp(grid_truth[item.column])), 1.0e-12)
            for item in simulator.outputs
        }

        specs.append(
            SimulatorSpec(
                simulator.id,
                f"単工程｜{simulator.manifest['name']}",
                inputs,
                outputs,
                evaluate,
                noise_scales,
            )
        )
    return specs


def _field_from_manifest(row: Mapping[str, object]) -> FieldSpec:
    lower, upper = row["range"]
    return FieldSpec(
        str(row["name"]),
        str(row.get("description") or row["name"]),
        str(row.get("unit") or ""),
        float(lower),
        float(upper),
    )


def _multistage_specs() -> list[SimulatorSpec]:
    from validation.multistage.src.stages import STAGES
    from validation.multistage.functional_coating import pipeline

    root = VALIDATION_ROOT / "multistage" / "functional_coating"
    specs: list[SimulatorSpec] = []
    for stage_id in ("coating", "drying", "curing"):
        stage = STAGES[stage_id]
        manifest = _manifest(root / stage_id / "manifest.json")
        input_rows = [*manifest["controls"], *manifest["incoming_state"]]
        inputs = tuple(_field_from_manifest(row) for row in input_rows)
        outputs = tuple(
            FieldSpec(
                str(row["name"]),
                str(row["description"]),
                str(row["unit"]),
                float(row["range"][0]),
                float(row["range"][1]),
            )
            for row in manifest["outputs"]
        )
        noise_scales = {
            item.name: max(float(item.upper - item.lower), 1.0e-12)
            for item in outputs
        }

        def evaluate(values: Mapping[str, float], current=stage) -> Mapping[str, object]:
            return current.evaluate(values)

        specs.append(
            SimulatorSpec(
                stage_id,
                f"複合工程内｜{manifest['name']}",
                inputs,
                outputs,
                evaluate,
                noise_scales,
            )
        )

    coating_manifest = _manifest(root / "coating" / "manifest.json")
    drying_manifest = _manifest(root / "drying" / "manifest.json")
    curing_manifest = _manifest(root / "curing" / "manifest.json")
    control_rows = [
        *coating_manifest["controls"],
        *drying_manifest["controls"],
        *curing_manifest["controls"],
    ]
    inputs = [_field_from_manifest(row) for row in control_rows]
    material_fields = {
        "material_viscosity_pa_s": ("供給液の粘度", "Pa·s"),
        "material_solids_fraction": ("供給液の固形分率", "fraction"),
        "material_bubble_fraction": ("供給液の気泡体積分率", "fraction"),
    }
    for name, bounds in pipeline.MATERIAL_BOUNDS.items():
        label, unit = material_fields[name]
        inputs.append(
            FieldSpec(
                name,
                label,
                unit,
                float(bounds[0]),
                float(bounds[1]),
                float(pipeline.DEFAULT_MATERIAL_STATE[name]),
            )
        )

    final_outputs = [
        FieldSpec(
            str(row["name"]),
            str(row["description"]),
            str(row["unit"]),
            float(row["range"][0]),
            float(row["range"][1]),
        )
        for row in curing_manifest["outputs"]
    ]
    final_outputs.extend(
        [
            FieldSpec("total_thermal_energy_kj_m2", "全熱エネルギー", "kJ/m2"),
            FieldSpec("line_throughput_m2_h", "ライン処理量", "m2/h", 300.0, 1800.0),
        ]
    )
    noise_scales = {
        item.name: max(float(item.upper - item.lower), 1.0e-12)
        for item in final_outputs
        if item.lower is not None and item.upper is not None
    }
    noise_scales["total_thermal_energy_kj_m2"] = 20000.0

    def evaluate_line(values: Mapping[str, float]) -> Mapping[str, object]:
        return pipeline.evaluate_line(values)["final"]

    specs.append(
        SimulatorSpec(
            "functional_coating",
            "複合工程全体｜機能性コーティング",
            tuple(inputs),
            tuple(final_outputs),
            evaluate_line,
            noise_scales,
        )
    )
    return specs


def load_specs() -> tuple[SimulatorSpec, ...]:
    """GUIに表示する全シミュレータを安定した順序で読み込みます。"""

    electrical = electrical_specs()
    # 新しい連結ラインを先頭へ置き、スクロールしなくても見つけられるようにします。
    lines = [item for item in electrical if item.id in
             {"interconnect_cu_al", "interconnect_cu_cu", "interconnect_al_al"}]
    stages = [item for item in electrical if item not in lines]
    specs = tuple([*lines, *stages, *_single_specs(), *_multistage_specs()])
    missing = [item.id for item in specs if not item.diagram_nodes and
               not (IMAGE_ROOT / f"{item.image_id or item.id}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"対応する画像がありません: {', '.join(missing)}")
    return specs


class ScrollablePanel(ttk.Frame):
    """入力数・出力数が増えても画面内で縦スクロールできる枠。"""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#ffffff")
        bar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.content = ttk.Frame(self.canvas, style="Panel.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=bar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        bar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.content.bind("<Configure>", self._update_scrollregion)
        self.canvas.bind("<Configure>", self._fit_width)
        self.bind_all("<MouseWheel>", self._wheel, add="+")

    def _wheel(self, event):
        widget = event.widget
        while widget is not None:
            if widget is self:
                self.canvas.yview_scroll(-int(event.delta / 120), "units")
                return "break"
            widget = getattr(widget, "master", None)

    def _update_scrollregion(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _fit_width(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.window, width=event.width)


class ValidationGui(tk.Tk):
    def __init__(self, specs: tuple[SimulatorSpec, ...]) -> None:
        super().__init__()
        self.specs = specs
        self.by_label = {item.label: item for item in specs}
        self.current = specs[0]
        self.input_vars: dict[str, tk.StringVar] = {}
        self.output_vars: dict[str, tk.StringVar] = {}
        self._image_source: tk.PhotoImage | None = None
        self._image_photo: tk.PhotoImage | None = None
        self._image_item: int | None = None
        self._processing_started = 0.0
        self._pending_values: dict[str, float] | None = None
        self.noise_enabled = tk.BooleanVar(value=False)
        self.rng = np.random.default_rng()
        self._results = queue.Queue()
        self._ready = None
        self._last_result = None

        self.title("物理シミュレータ GUI")
        self.geometry("1500x900")
        self.minsize(1100, 700)
        self.configure(background="#eef2f6")
        self._configure_styles()
        self._build_layout()
        self._select(self.current.label)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("App.TFrame", background="#eef2f6")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Title.TLabel", background="#eef2f6", foreground="#17253b", font=("Yu Gothic UI", 18, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#6b7788", font=("Yu Gothic UI", 9))
        style.configure("Field.TLabel", background="#ffffff", foreground="#26364d", font=("Yu Gothic UI", 10))
        style.configure("Output.TLabel", background="#ffffff", foreground="#26364d", font=("Yu Gothic UI", 10))
        style.configure("Run.TButton", font=("Yu Gothic UI", 13, "bold"), padding=(28, 12))
        style.configure("Card.TLabelframe", background="#ffffff", padding=14)
        style.configure("Card.TLabelframe.Label", background="#ffffff", foreground="#17253b", font=("Yu Gothic UI", 12, "bold"))

    def _build_layout(self) -> None:
        header = ttk.Frame(self, style="App.TFrame", padding=(24, 18, 24, 12))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="物理シミュレータ", style="Title.TLabel").grid(row=0, column=0, padx=(0, 24))
        self.selector = ttk.Combobox(
            header,
            state="readonly",
            values=[item.label for item in self.specs],
            font=("Yu Gothic UI", 11),
            width=54,
            height=len(self.specs),
        )
        self.selector.grid(row=0, column=1, sticky="w")
        self.selector.bind("<<ComboboxSelected>>", lambda _event: self._select(self.selector.get()))

        body = ttk.Frame(self, style="App.TFrame", padding=(24, 8, 24, 22))
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=3, minsize=300)
        body.columnconfigure(1, weight=4, minsize=430)
        body.columnconfigure(2, weight=3, minsize=320)

        self.left_card = ttk.LabelFrame(body, text="入力パラメータ", style="Card.TLabelframe")
        self.left_card.grid(row=0, column=0, padx=(0, 12), sticky="nsew")
        self.left_card.rowconfigure(0, weight=1)
        self.left_card.columnconfigure(0, weight=1)
        self.input_panel = ScrollablePanel(self.left_card)
        self.input_panel.grid(row=0, column=0, sticky="nsew")

        center = ttk.LabelFrame(body, text="シミュレータ", style="Card.TLabelframe")
        center.grid(row=0, column=1, padx=12, sticky="nsew")
        center.rowconfigure(0, weight=1)
        center.columnconfigure(0, weight=1)
        self.image_canvas = tk.Canvas(center, background="#ffffff", highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, padx=8, pady=8, sticky="nsew")
        self.image_canvas.bind("<Configure>", self._center_image)
        self.noise_toggle = ttk.Checkbutton(
            center,
            text="測定ノイズあり（出力レンジの2%）",
            variable=self.noise_enabled,
        )
        self.noise_toggle.grid(row=1, column=0, pady=(8, 2))
        self.run_button = ttk.Button(center, text="計 算", style="Run.TButton", command=self._run)
        self.run_button.grid(row=2, column=0, pady=(6, 6))
        self.save_button = ttk.Button(center, text="結果JSON保存", command=self._save_result, state="disabled")
        self.save_button.grid(row=5, column=0, pady=4)
        self.description_var = tk.StringVar()
        ttk.Label(center, textvariable=self.description_var, style="Hint.TLabel", wraplength=390).grid(row=6, column=0, sticky="ew", pady=6)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            center,
            variable=self.progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.progress.grid(row=3, column=0, padx=34, pady=(4, 5), sticky="ew")
        self.status_var = tk.StringVar(value="入力値を確認して「計算」を押してください。")
        ttk.Label(center, textvariable=self.status_var, style="Hint.TLabel", anchor="center").grid(row=4, column=0, pady=(0, 4), sticky="ew")

        self.right_card = ttk.LabelFrame(body, text="出力パラメータ", style="Card.TLabelframe")
        self.right_card.grid(row=0, column=2, padx=(12, 0), sticky="nsew")
        self.right_card.rowconfigure(0, weight=1)
        self.right_card.columnconfigure(0, weight=1)
        self.output_panel = ScrollablePanel(self.right_card)
        self.output_panel.grid(row=0, column=0, sticky="nsew")

        self.left_dim = tk.Frame(body, background="#263241")
        self.right_dim = tk.Frame(body, background="#263241")

        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

    @staticmethod
    def _clear(frame: tk.Misc) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _select(self, label: str) -> None:
        self.current = self.by_label[label]
        self.selector.set(label)
        self._build_inputs()
        self._build_outputs()
        self._load_image()
        self.description_var.set(self.current.description)
        self.noise_enabled.set(False)
        self.noise_toggle.configure(text=self.current.noise_label, state="normal" if self.current.allow_noise else "disabled")
        self._last_result = None
        self.save_button.configure(state="disabled")
        self.status_var.set("入力値を確認して「計算」を押してください。")

    def _save_result(self):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path and self._last_result is not None:
            def plain(value):
                if isinstance(value, dict):
                    return {str(key): plain(item) for key, item in value.items()}
                if isinstance(value, (list, tuple)):
                    return [plain(item) for item in value]
                if isinstance(value, np.ndarray):
                    return plain(value.item() if value.size == 1 else value.tolist())
                if isinstance(value, np.generic):
                    return plain(value.item())
                if isinstance(value, float) and not np.isfinite(value):
                    return None
                return value
            try:
                Path(path).write_text(json.dumps(plain(self._last_result), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
            except Exception as error:
                messagebox.showerror("保存エラー", str(error), parent=self)

    def _build_inputs(self) -> None:
        frame = self.input_panel.content
        self._clear(frame)
        self.input_vars = {}
        frame.columnconfigure(0, weight=1)
        row_index, previous_group = 0, ""
        for field in self.current.inputs:
            if field.group and field.group != previous_group:
                ttk.Label(frame, text=field.group, style="Field.TLabel").grid(row=row_index, column=0, sticky="w", pady=(12, 2))
                row_index += 1
                previous_group = field.group
            block = ttk.Frame(frame, style="Panel.TFrame", padding=(4, 6))
            block.grid(row=row_index, column=0, sticky="ew")
            row_index += 1
            block.columnconfigure(0, weight=1)
            ttk.Label(block, text=field.label, style="Field.TLabel").grid(row=0, column=0, sticky="w")
            variable = tk.StringVar(value=f"{field.initial_value():.10g}")
            self.input_vars[field.name] = variable
            ttk.Entry(block, textvariable=variable, state="readonly" if field.readonly else "normal", font=("Consolas", 11)).grid(row=1, column=0, pady=(3, 0), sticky="ew")
            bounds = f"{field.lower:g} ～ {field.upper:g}" if field.lower is not None and field.upper is not None else ""
            range_text = f"{field.name}  ｜  {bounds} {field.unit}" + ("（固定）" if field.readonly else "")
            ttk.Label(block, text=range_text, style="Hint.TLabel").grid(row=2, column=0, pady=(2, 0), sticky="w")

    def _build_outputs(self) -> None:
        frame = self.output_panel.content
        self._clear(frame)
        self.output_vars = {}
        frame.columnconfigure(0, weight=1)
        row_index, previous_group = 0, ""
        for field in self.current.outputs:
            if field.group and field.group != previous_group:
                ttk.Label(frame, text=field.group, style="Output.TLabel").grid(row=row_index, column=0, sticky="w", pady=(12, 2))
                row_index += 1
                previous_group = field.group
            block = ttk.Frame(frame, style="Panel.TFrame", padding=(4, 6))
            block.grid(row=row_index, column=0, sticky="ew")
            row_index += 1
            block.columnconfigure(0, weight=1)
            ttk.Label(block, text=field.label, style="Output.TLabel").grid(row=0, column=0, sticky="w")
            variable = tk.StringVar(value="—")
            self.output_vars[field.name] = variable
            ttk.Entry(block, textvariable=variable, state="readonly", font=("Consolas", 11)).grid(row=1, column=0, pady=(3, 0), sticky="ew")
            ttk.Label(block, text=f"{field.name}  ｜  {field.unit}", style="Hint.TLabel").grid(row=2, column=0, pady=(2, 0), sticky="w")

    def _load_image(self) -> None:
        self.image_canvas.delete("all")
        self._image_item = None
        if self.current.diagram_nodes:
            self.after_idle(self._center_image)
            return
        path = IMAGE_ROOT / f"{self.current.image_id or self.current.id}.png"
        self._image_source = tk.PhotoImage(file=path)
        factor = max(1, int(np.ceil(max(self._image_source.width(), self._image_source.height()) / 440)))
        self._image_photo = self._image_source.subsample(factor, factor)
        self.image_canvas.delete("all")
        self._image_item = self.image_canvas.create_image(0, 0, image=self._image_photo, anchor="center")
        self.after_idle(self._center_image)

    def _center_image(self, _event: tk.Event | None = None, offset_y: float = 0.0) -> None:
        if self.current.diagram_nodes:
            # DAGの分岐・合流をそのまま描画し、画像資産への依存を増やしません。
            canvas = self.image_canvas
            canvas.delete("diagram")
            width, height = max(canvas.winfo_width(), 360), max(canvas.winfo_height(), 280)
            positions = {sid: (x * width, y * height) for sid, _, x, y in self.current.diagram_nodes}
            for source, target in self.current.diagram_edges:
                x1, y1 = positions[source]
                x2, y2 = positions[target]
                canvas.create_line(x1, y1 + 24, x2, y2 - 24, arrow=tk.LAST, fill="#7790ac", width=2, tags="diagram")
            for sid, label, _, _ in self.current.diagram_nodes:
                x, y = positions[sid]
                canvas.create_rectangle(x - 78, y - 24, x + 78, y + 24, fill="#e9f1fa", outline="#7790ac", tags="diagram")
                canvas.create_text(x, y, text=label, width=150, font=("Yu Gothic UI", 10), tags="diagram")
            return
        if self._image_item is not None:
            self.image_canvas.coords(
                self._image_item,
                self.image_canvas.winfo_width() / 2,
                self.image_canvas.winfo_height() / 2 + offset_y,
            )

    def _read_inputs(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for field in self.current.inputs:
            raw = self.input_vars[field.name].get().strip()
            try:
                value = float(raw)
            except ValueError as error:
                raise ValueError(f"「{field.label}」へ数値を入力してください。") from error
            if not np.isfinite(value):
                raise ValueError(f"「{field.label}」は有限の数値にしてください。")
            if field.lower is not None and value < field.lower:
                raise ValueError(f"「{field.label}」は{field.lower:g}以上にしてください。")
            if field.upper is not None and value > field.upper:
                raise ValueError(f"「{field.label}」は{field.upper:g}以下にしてください。")
            values[field.name] = value
        return values

    @staticmethod
    def _format_output(value: object) -> str:
        if value is None:
            return "未測定／未評価"
        array = np.asarray(value)
        if array.size == 1 and isinstance(array.reshape(-1)[0], (str, np.str_)):
            return str(array.reshape(-1)[0])
        scalar = _scalar(value)
        if isinstance(scalar, bool):
            return "True" if scalar else "False"
        if not np.isfinite(scalar):
            return "評価不能"
        if scalar == 0:
            return "0"
        if abs(scalar) >= 1.0e6 or abs(scalar) < 1.0e-5:
            return f"{scalar:.6e}"
        return f"{scalar:.8g}"

    def _run(self) -> None:
        if self._pending_values is not None:
            return
        try:
            self._pending_values = self._read_inputs()
        except Exception as error:
            self.status_var.set("計算できませんでした。入力値を確認してください。")
            messagebox.showerror("計算エラー", str(error), parent=self)
            return

        self.run_button.configure(state="disabled")
        self.selector.configure(state="disabled")
        self.noise_toggle.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.progress_var.set(0.0)
        self.left_dim.place(in_=self.left_card, x=0, y=0, relwidth=1, relheight=1)
        self.right_dim.place(in_=self.right_card, x=0, y=0, relwidth=1, relheight=1)
        self.left_dim.lift()
        self.right_dim.lift()
        self._processing_started = time.monotonic()
        self._ready = None
        self._last_result = None
        spec, values, noise = self.current, dict(self._pending_values), self.noise_enabled.get()
        # ワーカーはTkの変数・ウィジェットに触れず、結果だけをキューへ返します。
        def compute():
            try:
                payload = {
                    "values": values, "outputs": evaluate_display(spec, values, noise, self.rng),
                    "message": "計算完了"}
                self._results.put((payload, None))
            except Exception as error:
                self._results.put((None, error))
        threading.Thread(target=compute, daemon=True).start()
        self.status_var.set("加工中… しばらくお待ちください。")
        self._animate_processing()

    def _animate_processing(self) -> None:
        elapsed_ms = (time.monotonic() - self._processing_started) * 1000.0
        progress = min(95.0, elapsed_ms / max(PROCESS_DURATION_MS, 1) * 100.0)
        self.progress_var.set(progress)

        phase = elapsed_ms / 170.0
        offset_y = float(np.sin(phase) * 6.0)
        glow = int(235 + 20 * (0.5 + 0.5 * np.sin(phase * 0.75)))
        self.image_canvas.configure(background=f"#{glow:02x}{glow:02x}ff")
        self._center_image(offset_y=offset_y)
        self.status_var.set(f"計算中… {elapsed_ms / 1000:.1f}秒")
        if self._ready is None:
            try:
                self._ready = self._results.get_nowait()
            except queue.Empty:
                pass

        if elapsed_ms < PROCESS_DURATION_MS or self._ready is None:
            self.after(ANIMATION_INTERVAL_MS, self._animate_processing)
            return
        self._finish_processing()

    def _finish_processing(self) -> None:
        try:
            if self._pending_values is None:
                raise RuntimeError("計算条件が失われました。")
            payload, error = self._ready
            if error is not None:
                raise error
            result = payload["outputs"]
            for field in self.current.outputs:
                if field.name not in result:
                    raise ValueError(f"出力「{field.name}」がモデルから返されませんでした。")
                self.output_vars[field.name].set(self._format_output(result[field.name]))
            self._last_result = {"simulator": self.current.id, "mode": "physical_simulation",
                                 "noise": self.noise_enabled.get(),
                                 "requested_inputs": dict(self._pending_values), **payload}
        except Exception as error:  # GUIでは利用者入力とモデル契約エラーを同じ場所へ表示する。
            self.status_var.set("計算できませんでした。入力値を確認してください。")
            messagebox.showerror("計算エラー", str(error), parent=self)
        else:
            mode = "測定ノイズあり" if self.noise_enabled.get() else "測定ノイズなし"
            self.status_var.set(payload["message"] + (f"（{mode}）" if self.current.allow_noise else ""))
        finally:
            self._pending_values = None
            self.progress_var.set(100.0)
            self.image_canvas.configure(background="#ffffff")
            self._center_image()
            self.left_dim.place_forget()
            self.right_dim.place_forget()
            self.run_button.configure(state="normal")
            self.selector.configure(state="readonly")
            self.noise_toggle.configure(state="normal" if self.current.allow_noise else "disabled")
            self.save_button.configure(state="normal" if self._last_result is not None else "disabled")


def smoke_test(specs: tuple[SimulatorSpec, ...]) -> None:
    """GUIを開けない環境でも、全モデルの初期入力と出力契約を確認します。"""

    for spec in specs:
        values = {item.name: item.initial_value() for item in spec.inputs}
        result = spec.evaluate(values)
        missing = [item.name for item in spec.outputs if item.name not in result]
        for item in spec.outputs:
            if item.name in result:
                ValidationGui._format_output(result[item.name])
        if missing:
            raise ValueError(f"{spec.id}: missing outputs: {', '.join(missing)}")
        missing_scales = [item.name for item in spec.outputs if item.name not in spec.noise_scales] if spec.allow_noise and spec.observe is None else []
        if missing_scales:
            raise ValueError(f"{spec.id}: missing noise scales: {', '.join(missing_scales)}")
        print(f"OK  {spec.id}: inputs={len(spec.inputs)}, outputs={len(spec.outputs)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validation物理シミュレータGUI")
    parser.add_argument("--smoke-test", action="store_true", help="画面を開かず全モデルを中点入力で評価")
    args = parser.parse_args(argv)
    specs = load_specs()
    if args.smoke_test:
        smoke_test(specs)
        return 0
    app = ValidationGui(specs)
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
