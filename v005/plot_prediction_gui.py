"""v005の予測断面と、その時点の実測を表示する単一ファイルGUI。

起動（リポジトリ直下）:
    python v005/plot_prediction_gui.py
    python v005/plot_prediction_gui.py --trial v005/trials/trial_cut_cu_20261009
依存: NumPy、matplotlib、Tkinter（通常のWindows Pythonに同梱）。
    python -m pip install numpy matplotlib
検証・プレビュー保存:
    python v005/plot_prediction_gui.py --smoke-test --trial TRIAL_PATH --save-preview preview.png

保存されたCSVだけを読みます。学習・シミュレータ実行・trialの変更は行いません。
全入力を左欄に表示し、X/Y選択中の入力は範囲表示・編集不可にします。
制約充足確率マップは保存済み平均・標準偏差から計算し、しきい値以上の格子点を表示します。
複数制約の確率は独立性を仮定した積です。表示する範囲は連続領域の保証ではありません。
入力2軸以外を固定した断面を描き、全実測表示では断面外の点を区別します。
ledger.jsonがあるtrialは実験ステップ、ないtrialはrunと学習行数で選択します。
予測更新なしの最終ステップでは前回の面を使い、そのことを画面に明示します。
"""
from __future__ import annotations

import argparse
import csv
import gzip
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import re
import threading
import sys
from types import SimpleNamespace
import tkinter as tk
from tkinter import filedialog, ttk

try:
    import numpy as np
    import matplotlib
    from matplotlib import font_manager
    from matplotlib.figure import Figure
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
except ImportError as exc:
    raise SystemExit("NumPy / matplotlib が必要です。使用するPythonで「python -m pip install numpy matplotlib」を実行してください。\n" + str(exc)) from exc

VERSION_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(VERSION_ROOT))
from src.hybrid.gp_component import constraint_probability
from src.hybrid.optimizer import constraint_probability_target
PREDICTIONS = {"NN": "_nn_pred", "Hybrid（NN＋残差GP）": "_hybrid_mean"}
SCATTER_ALL = "全実測（断面外の点も表示）"
SCATTER_SLICE = "固定条件に一致する実測のみ"
VIEW_RESPONSE = "予測面と実測"
VIEW_FEASIBILITY = "制約充足確率マップ"


def configure_fonts():
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [name for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP") if name in available]
    matplotlib.rcParams["font.family"] = preferred + ["DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path):
    for encoding in ("utf-8-sig", "cp932"):
        try:
            opener = gzip.open if path.suffix == ".gz" else open
            with opener(path, "rt", encoding=encoding, newline="") as stream:
                reader = csv.DictReader(stream)
                return list(reader.fieldnames or []), list(reader)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"CSVの文字コードを読めません: {path}")


def read_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


@dataclass(frozen=True)
class Frame:
    label: str
    path: Path
    run_id: str
    training_rows: int
    observed_rows: int
    step: int | None
    reused: bool = False


@dataclass
class TrialData:
    root: Path
    variables: dict
    parameters: list[str]
    objective: str
    direction: str
    frames: list[Frame]


@dataclass
class LoadedFrame:
    trial: TrialData
    frame: Frame
    predictions: dict[str, np.ndarray]
    observations: dict[str, np.ndarray]
    measured_rows: list[dict]
    allowed: np.ndarray
    axes: list[str]
    outputs: list[str]
    sources: list[str]
    observation_source: str
    focus_metadata: dict | None = None


def discover_trial(root: Path) -> TrialData:
    root = root.expanduser().resolve()
    _, problem = read_csv(root / "problem.csv")
    variables = {row["column"]: row for row in problem}
    parameters = [row["column"] for row in problem if row["role"] == "parameter"]
    objectives = [row for row in problem if row["role"] == "objective"]
    if len(objectives) != 1 or not parameters:
        raise ValueError("problem.csvにparameterと1つのobjectiveが必要です。")
    runs = []
    for path in sorted((root / "output" / "response_spaces").glob("hybrid_response_space_run_*.csv*")):
        # Discovery needs only the first row, not the full response grid.
        if path.suffix not in (".csv", ".gz"):
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8-sig", newline="") as stream:
            first = next(csv.DictReader(stream), None)
        if first is None:
            continue
        match = re.search(r"(\d+)\.csv(?:\.gz)?$", path.name)
        runs.append((path, first.get("run_id", f"run_{match.group(1)}"), int(first["data_rows"])))
    if not runs:
        raise ValueError("output/response_spaces/に予測空間CSVがありません。v005の--runを実行してください。")
    ledger = read_json(root / "ledger.json", [])
    frames = []
    for path, run_id, count in runs:
        matching = [entry for entry in ledger if int(entry["total_measurements"]) == count]
        step = int(matching[-1]["step"]) if matching else None
        prefix = f"ステップ {step:02d}" if step is not None else run_id
        frames.append(Frame(f"{prefix} ｜ {run_id} ｜ 実測 {count}件", path, run_id, count, count, step))
    # A budget stop may have measured rows without any new prediction file.
    if ledger and int(ledger[-1]["total_measurements"]) > max(f.observed_rows for f in frames):
        latest = max(frames, key=lambda f: (f.training_rows, f.run_id))
        step = int(ledger[-1]["step"])
        count = int(ledger[-1]["total_measurements"])
        frames.append(Frame(f"ステップ {step:02d} ｜ 予測更新なし ｜ 実測 {count}件", latest.path,
                            latest.run_id, latest.training_rows, count, step, True))
    return TrialData(root, variables, parameters, objectives[0]["column"], objectives[0]["direction"], frames)


def numeric_columns(rows, columns, source):
    result = {}
    for name in columns:
        try:
            values = np.asarray([float(row[name]) for row in rows], dtype=float)
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"{source}の数値列を読めません: {name}") from exc
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source}にNaN / Infがあります: {name}")
        result[name] = values
    return result


def observations_at(trial: TrialData, frame: Frame):
    # Prefer immutable step snapshots, including in trials whose current CSV was edited later.
    if frame.step is not None:
        snapshot = trial.root / "steps" / f"step_{frame.step:03d}" / "experiments.csv"
        if snapshot.exists():
            _, rows = read_csv(snapshot)
            if len(rows) != frame.observed_rows:
                raise ValueError(f"{snapshot}の行数が学習時点と一致しません。")
            return rows, str(snapshot.relative_to(trial.root))
    # Generic v005 trials do not archive experiment snapshots. State the append-only assumption.
    _, rows = read_csv(trial.root / "data" / "experiments.csv")
    if len(rows) < frame.observed_rows:
        raise ValueError(f"実験CSVは{len(rows)}行ですが、このrunは{frame.observed_rows}行必要です。")
    return rows[:frame.observed_rows], f"現行data/experiments.csvの先頭{frame.observed_rows}行（追記のみを前提）"


def load_frame(trial: TrialData, frame: Frame) -> LoadedFrame:
    header, rows = read_csv(frame.path)
    if not rows or any(int(row["data_rows"]) != frame.training_rows for row in rows):
        raise ValueError("予測CSVのdata_rowsが一致しません。")
    sources = [source for source, suffix in PREDICTIONS.items() if any(n.endswith(suffix) for n in header)]
    outputs = [name for name, spec in trial.variables.items() if spec["role"] != "parameter"
               and any(name + PREDICTIONS[source] in header for source in sources)]
    if not outputs or not sources:
        raise ValueError("NNまたはHybridの予測出力列がありません。")
    prediction_names = [*trial.parameters, *[name + PREDICTIONS[source] for name in outputs
                                           for source in sources if name + PREDICTIONS[source] in header]]
    prediction_names += [name for name in header if name.endswith(("_hybrid_std", "_nn_std"))]
    prediction = numeric_columns(rows, prediction_names, frame.path.name)
    measured, observation_source = observations_at(trial, frame)
    observations = numeric_columns(measured, [*trial.parameters, *outputs], "実験CSV")
    varying = [name for name in trial.parameters if np.unique(prediction[name]).size > 1]
    if len(varying) < 2:
        raise ValueError("3D面の表示には、予測空間で変化する入力が2つ以上必要です。")
    allowed = np.asarray([str(row.get("experiment_allowed", "True")).strip().lower() in ("true", "1", "yes") for row in rows])
    # Focus metadata is optional and keyed by the exact run id stored in the CSV.
    # Old trials therefore keep their original presentation unchanged.
    metadata_path = trial.root / "output" / "parameter_focus" / f"{frame.run_id}.json"
    metadata = None
    if metadata_path.is_file():
        try:
            candidate = read_json(metadata_path, None)
            if isinstance(candidate, dict) and isinstance(candidate.get("parameters"), list):
                metadata = candidate
        except (OSError, json.JSONDecodeError):
            metadata = None
    return LoadedFrame(trial, frame, prediction, observations, measured, allowed, varying, outputs, sources,
                       observation_source, metadata)


FOCUS_STATE_LABELS = {
    "UNTESTED": "未評価",
    "UNDECIDED": "判定保留",
    "ACTIVE": "探索継続",
    "FIX_CANDIDATE": "固定候補（確認中）",
    "TEMPORARILY_FIXED": "暫定固定（再確認対象）",
}


def focus_parameter(data: LoadedFrame, name: str):
    metadata = data.focus_metadata
    if not metadata:
        return None
    return next((item for item in metadata["parameters"]
                 if isinstance(item, dict) and item.get("column") == name), None)


def focus_scope_notice(data: LoadedFrame):
    metadata = data.focus_metadata
    if not metadata:
        return ""
    scope = metadata.get("candidate_space_scope")
    if scope == "focused_with_checks":
        return "探索範囲：集中探索＋確認点の範囲。全候補域の最適性を示すものではありません。"
    if scope == "full":
        if metadata.get('candidate_space_complete') is False:
            return '探索範囲：元の入力範囲から抽出した候補（全格子の検査ではありません）'
        return "探索範囲：元の入力範囲"
    return ""


def axis_label(trial, name):
    spec = trial.variables[name]
    title = spec.get("display_name") or name
    unit = spec.get("unit", "")
    return f"{title} [{unit}]" if unit else title


def fixed_defaults(data: LoadedFrame):
    # Anchor the initial slice at an observed feasible objective optimum when one exists.
    feasible = np.ones(data.frame.observed_rows, dtype=bool)
    for name, spec in data.trial.variables.items():
        if spec["role"] == "constraint" and name in data.observations:
            target = float(spec["target"])
            values = data.observations[name]
            feasible &= values <= target if spec["direction"] == "less_equal" else values >= target
    indices = np.flatnonzero(feasible)
    if not indices.size:
        indices = np.arange(data.frame.observed_rows)
    objective = data.observations[data.trial.objective]
    pos = np.argmin(objective[indices]) if data.trial.direction == "minimize" else np.argmax(objective[indices])
    anchor = int(indices[pos])
    return {name: float((values := np.unique(data.predictions[name]))[np.argmin(np.abs(values-data.observations[name][anchor]))])
            for name in data.trial.parameters}


def slice_data(data, x, y, output, source, fixed, hide_disallowed=True):
    if x == y:
        raise ValueError("X軸とY軸は異なる入力を選択してください。")
    mask = np.ones(len(data.allowed), dtype=bool)
    observed_mask = np.ones(data.frame.observed_rows, dtype=bool)
    for name in data.trial.parameters:
        if name in (x, y):
            continue
        # CSV uses 10 significant digits; compare with the full precision experimental inputs.
        mask &= np.isclose(data.predictions[name], fixed[name], rtol=1e-8, atol=1e-10)
        observed_mask &= np.isclose(data.observations[name], fixed[name], rtol=1e-8, atol=1e-10)
    if hide_disallowed:
        mask &= data.allowed
    column = output + PREDICTIONS[source]
    if column not in data.predictions:
        raise ValueError(f"このrunに{source}の{output}予測はありません。")
    indices = np.flatnonzero(mask)
    if not indices.size:
        raise ValueError("この固定条件に対応する予測点がありません。固定値または実験禁止候補の非表示設定を変更してください。")
    xs, ys = np.unique(data.predictions[x][indices]), np.unique(data.predictions[y][indices])
    xx, yy = np.meshgrid(xs, ys)
    zz = np.full(xx.shape, np.nan)
    counts = np.zeros(xx.shape, dtype=int)
    for i in indices:
        a = int(np.searchsorted(ys, data.predictions[y][i]))
        b = int(np.searchsorted(xs, data.predictions[x][i]))
        zz[a,b] = data.predictions[column][i]
        counts[a,b] += 1
    if np.any(counts > 1):
        raise ValueError("固定条件の断面に重複する予測点があります。平均化せずCSVを確認してください。")
    return xx, yy, zz, indices, observed_mask


def feasibility_probabilities(data, source):
    """Reconstruct optimizer probabilities from saved means and calibrated stds."""
    joint = np.ones(len(data.allowed))
    individual = {}
    settings = read_json(data.trial.root / "search_settings.json", {})
    steps = dict(settings.get("ordinal_constraint_steps", {}))
    # Match effective_search_config using only observations available at training time.
    for name, spec in data.trial.variables.items():
        if spec["role"] == "constraint" and spec.get("unit") == "level" and name not in steps:
            values = data.observations[name][:data.frame.training_rows]
            if values.size and np.all(values == np.floor(values)):
                steps[name] = 1.0
    config = SimpleNamespace(ordinal_constraint_steps=steps)
    for name, spec in data.trial.variables.items():
        if spec["role"] != "constraint":
            continue
        mean_column = name + PREDICTIONS[source]
        std_column = name + ("_nn_std" if source == "NN" else "_hybrid_std")
        if mean_column not in data.predictions or std_column not in data.predictions:
            raise ValueError(f"制約充足確率の計算に必要な保存列がありません: {mean_column}, {std_column}")
        if np.any(data.predictions[std_column] < 0):
            raise ValueError(f"標準偏差が負です: {std_column}")
        constraint = SimpleNamespace(column=name, target=float(spec["target"]), direction=spec["direction"])
        probability = constraint_probability(data.predictions[mean_column], data.predictions[std_column],
                                             constraint.direction, constraint_probability_target(constraint, config))
        individual[name] = probability
        joint *= probability
    return joint, individual


def feasible_summary(data, probability, threshold):
    selected = data.allowed & (probability >= threshold)
    count = int(selected.sum())
    ranges = []
    if count:
        for name in data.trial.parameters:
            values = np.unique(data.predictions[name][selected])
            ranges.append(f"{axis_label(data.trial, name)}: {values.min():.6g}～{values.max():.6g}（{len(values)}水準）")
    text = f"保存格子全体：しきい値以上かつ実験許可 {count}/{int(data.allowed.sum())}点"
    if ranges:
        text += "\n" + " ／ ".join(ranges)
    else:
        text += "\n該当条件なし"
    return selected, text


def fixture_notice(data):
    fixture = read_json(data.trial.root / "fixture.json", {})
    context = fixture.get("fixed_incoming_context", {})
    return f"流入状態{len(context)}項目はfixture.jsonに固定（単独工程）" if context else ""


def draw_feasibility(figure, data, x, y, output, source, fixed, hide_disallowed, threshold):
    xx, yy, _, indices, on_slice = slice_data(data, x, y, output, source, fixed, hide_disallowed)
    probability, individual = feasibility_probabilities(data, source)
    selected, summary = feasible_summary(data, probability, threshold)
    figure.clear()
    ax = figure.add_subplot(111)
    shown_probability = np.where(data.allowed[indices], probability[indices], np.nan)
    dots = ax.scatter(data.predictions[x][indices], data.predictions[y][indices],
                      c=shown_probability, cmap="viridis", vmin=0, vmax=1, s=150, marker="s")
    qualified = indices[selected[indices]]
    if qualified.size:
        ax.scatter(data.predictions[x][qualified], data.predictions[y][qualified], s=230,
                   marker="s", facecolors="none", edgecolors="#f07800", linewidths=2,
                   label=f"全制約の達成確率 ≥ {threshold:.0%}")
    observed_feasible = np.ones(data.frame.observed_rows, dtype=bool)
    for name in individual:
        spec = data.trial.variables[name]
        values = data.observations[name]
        observed_feasible &= values <= float(spec["target"]) if spec["direction"] == "less_equal" else values >= float(spec["target"])
    for successful, marker, label in ((True, "o", "断面一致の実測：制約充足"), (False, "x", "断面一致の実測：制約未充足")):
        mask = on_slice & (observed_feasible == successful)
        if mask.any():
            ax.scatter(data.observations[x][mask], data.observations[y][mask], marker=marker,
                       c="black", s=35, label=label)
    ax.set_xlabel(axis_label(data.trial, x)); ax.set_ylabel(axis_label(data.trial, y))
    ax.set_title(f"{data.trial.root.name} | {source}\n{data.frame.label}\n全制約の達成確率（独立性を仮定した積）", fontsize=10)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    figure.colorbar(dots, ax=ax).set_label("モデルによる制約充足確率")
    ax.grid(alpha=.2)
    figure.subplots_adjust(left=.12, right=.9, bottom=.13, top=.85)
    if data.frame.reused:
        summary = f"予測更新なし：{data.frame.run_id}（学習実測{data.frame.training_rows}件）を使用。\n" + summary
    summary += f"\n現在の断面：該当 {len(qualified)}/{int(data.allowed[indices].sum())}点。各入力の最小～最大は射影で、範囲内の全組合せの充足を保証しません。"
    summary += "\n保存済み格子上の予測です。格子間・未保存領域や実際の成功確率を保証するものではありません。"
    if fixture_notice(data): summary = fixture_notice(data) + "\n" + summary
    if not individual: summary += " 制約未定義のため、許可点の確率は1です。"
    return ax, {"summary": summary}


def draw_plot(figure, data, x, y, output, source, fixed, scatter_mode=SCATTER_ALL, hide_disallowed=True, angles=(26,-55)):
    xx, yy, zz, indices, on_slice = slice_data(data,x,y,output,source,fixed,hide_disallowed)
    figure.clear()
    ax = figure.add_subplot(111, projection="3d")
    shown = np.ones(data.frame.observed_rows,dtype=bool) if scatter_mode == SCATTER_ALL else on_slice
    measured_z = data.observations[output][shown]
    all_z = np.concatenate([zz[np.isfinite(zz)], measured_z])
    lo, hi = float(all_z.min()), float(all_z.max())
    if np.isclose(lo,hi):
        delta = max(abs(lo)*.01, .001); lo-=delta; hi+=delta
    norm = Normalize(lo,hi)
    colormap = "viridis"
    if min(zz.shape) >= 2:
        ax.plot_surface(xx,yy,np.ma.masked_invalid(zz),cmap=colormap,norm=norm,
                        linewidth=.35,edgecolor=(.15,.2,.25,.3),alpha=.70,
                        rcount=zz.shape[0],ccount=zz.shape[1],antialiased=True)
    else:
        ax.scatter(data.predictions[x][indices],data.predictions[y][indices],
                   data.predictions[output+PREDICTIONS[source]][indices],c=zz[np.isfinite(zz)],
                   cmap=colormap,norm=norm,marker="s",s=45,label="予測点（面に必要な2軸の水準不足）")
    for mask, label, border, size in ((shown & ~on_slice,"実測：他の固定入力が異なる", "#66717e",32),
                                     (shown & on_slice,"実測：固定入力が一致", "#ff8c00",64)):
        if mask.any():
            ax.scatter(data.observations[x][mask],data.observations[y][mask],data.observations[output][mask],
                       c=data.observations[output][mask],cmap=colormap,norm=norm,edgecolors=border,
                       linewidths=1.7 if border=="#ff8c00" else .6,s=size,depthshade=False,label=label)
    ax.set_xlabel(axis_label(data.trial,x),labelpad=10)
    ax.set_ylabel(axis_label(data.trial,y),labelpad=10)
    ax.set_zlabel(axis_label(data.trial,output),labelpad=10)
    ax.view_init(*angles)
    title = f"{data.trial.root.name}  |  {source}\n{data.frame.label}"
    if data.frame.reused:
        title += f"\n予測面は {data.frame.run_id}（{data.frame.training_rows}実測で学習）を継続表示"
    ax.set_title(title,fontsize=10,pad=12)
    if shown.any(): ax.legend(loc="upper left",fontsize=8)
    colorbar=figure.colorbar(ScalarMappable(norm=norm,cmap=colormap),ax=ax,shrink=.62,pad=.12)
    colorbar.set_label(axis_label(data.trial,output),fontsize=9)
    figure.subplots_adjust(left=.02,right=.88,bottom=.08,top=.89)
    return ax, {"surface_points":len(indices),"on_slice_points":int(on_slice.sum()),
                "shown_points":int(shown.sum()),"total_points":data.frame.observed_rows}


class PredictionViewer:
    def __init__(self, root, initial_trial=None):
        self.root=root
        self.root.title("v005 予測空間ビューア — 3D予測面と実測")
        self.root.geometry("1460x920")
        self.root.minsize(1080,720)
        self.trial=None; self.loaded=None; self.ax=None
        self.fixed={}; self.fixed_variables={}; self.fixed_choices={}; self.input_widgets={}
        self.messages=queue.Queue(); self.generation=0; self.render_job=None
        self.folder=tk.StringVar(value=str(VERSION_ROOT/"trials"))
        self.trial_var=tk.StringVar(); self.frame_var=tk.StringVar()
        self.x=tk.StringVar(); self.y=tk.StringVar(); self.z=tk.StringVar()
        self.source=tk.StringVar(value="NN")
        self.scatter=tk.StringVar(value=SCATTER_ALL)
        self.hide=tk.BooleanVar(value=True)
        self.view=tk.StringVar(value=VIEW_RESPONSE)
        self.threshold=tk.StringVar(value="95")
        self.info=tk.StringVar(value="trialフォルダを選択してください。")
        self._widgets()
        self.root.after(80,self._poll)
        self.set_folder(Path(initial_trial) if initial_trial else VERSION_ROOT/"trials")

    def _combo(self,parent,label,var,callback,width=22):
        box=ttk.Frame(parent); box.pack(side="left",padx=(0,12))
        ttk.Label(box,text=label).pack(anchor="w")
        widget=ttk.Combobox(box,textvariable=var,state="readonly",width=width)
        widget.pack(fill="x"); widget.bind("<<ComboboxSelected>>",callback)
        return widget

    def _widgets(self):
        outer=ttk.Frame(self.root,padding=12); outer.pack(fill="both",expand=True)
        location=ttk.Frame(outer); location.pack(fill="x",pady=(0,8))
        ttk.Label(location,text="フォルダ").pack(side="left",padx=(0,8))
        ttk.Entry(location,textvariable=self.folder,state="readonly").pack(side="left",fill="x",expand=True)
        ttk.Button(location,text="trial / trials を選択…",command=self.browse).pack(side="left",padx=8)
        ttk.Button(location,text="再読込",command=lambda:self.set_folder(Path(self.folder.get()))).pack(side="left")
        row=ttk.Frame(outer); row.pack(fill="x",pady=(0,8))
        self.trial_combo=self._combo(row,"trial",self.trial_var,self.change_trial,36)
        self.frame_combo=self._combo(row,"実験ステップ / 学習時点",self.frame_var,self.change_frame,58)
        self.source_combo=self._combo(row,"予測モデル",self.source,self.schedule_draw,24)
        row2=ttk.Frame(outer); row2.pack(fill="x",pady=(0,10))
        self.x_combo=self._combo(row2,"入力 X",self.x,self.axes_changed,25)
        self.y_combo=self._combo(row2,"入力 Y",self.y,self.axes_changed,25)
        self.z_combo=self._combo(row2,"出力 Z",self.z,self.schedule_draw,26)
        self.scatter_combo=self._combo(row2,"実測点",self.scatter,self.schedule_draw,32)
        self.scatter_combo["values"]=(SCATTER_ALL,SCATTER_SLICE)
        probability_row=ttk.Frame(outer); probability_row.pack(fill="x",pady=(0,8))
        self.view_combo=self._combo(probability_row,"表示",self.view,self.schedule_draw,24)
        self.view_combo["values"]=(VIEW_RESPONSE,VIEW_FEASIBILITY)
        ttk.Label(probability_row,text="制約充足確率のしきい値 (%)").pack(side="left")
        threshold_box=ttk.Spinbox(probability_row,from_=0,to=100,increment=1,textvariable=self.threshold,width=6,command=self.schedule_draw)
        threshold_box.pack(side="left",padx=8)
        threshold_box.bind("<Return>",self.schedule_draw); threshold_box.bind("<FocusOut>",self.schedule_draw)
        ttk.Label(probability_row,text="確率マップ：橙枠がしきい値以上の格子点。複数制約は独立性を仮定。",wraplength=650).pack(side="left")
        body=ttk.Panedwindow(outer,orient="horizontal"); body.pack(fill="both",expand=True)
        side=ttk.Frame(body,width=300); body.add(side,weight=0)
        ttk.Label(side,text="全入力変数：軸選択と固定条件",font=("Yu Gothic UI",11,"bold")).pack(anchor="w",pady=(0,8))
        ttk.Label(side,text="X・Yは面の軸、その他はこの断面の固定条件です。探索状態の『暫定固定』は、実験で確定した固定値を意味しません。",
                  wraplength=290).pack(anchor="w",pady=(0,8))
        ttk.Button(side,text="最良実測の条件に固定",command=self.anchor_fixed).pack(fill="x",pady=(0,5))
        ttk.Checkbutton(side,text="実験禁止候補を面から除外",variable=self.hide,command=self.schedule_draw).pack(anchor="w",pady=(0,8))
        scroll_area=ttk.Frame(side);scroll_area.pack(fill="both",expand=True)
        self.fixed_canvas=tk.Canvas(scroll_area,highlightthickness=0,width=290)
        scrollbar=ttk.Scrollbar(scroll_area,orient="vertical",command=self.fixed_canvas.yview)
        self.fixed_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right",fill="y");self.fixed_canvas.pack(side="left",fill="both",expand=True)
        self.fixed_frame=ttk.Frame(self.fixed_canvas)
        self.fixed_window=self.fixed_canvas.create_window((0,0),window=self.fixed_frame,anchor="nw")
        self.fixed_frame.bind("<Configure>",lambda event:self.fixed_canvas.configure(scrollregion=self.fixed_canvas.bbox("all")))
        self.fixed_canvas.bind("<Configure>",lambda event:self.fixed_canvas.itemconfigure(self.fixed_window,width=event.width))
        ttk.Label(side,text="橙枠：固定条件と一致する実測\n灰枠：他の固定入力が異なる実測\n面は保存済み格子を結んだ断面です。\nマウスで回転・拡大できます。",wraplength=290,justify="left").pack(anchor="w",pady=10)
        plot=ttk.Frame(body);body.add(plot,weight=1)
        self.figure=Figure(figsize=(10,7),dpi=100)
        self.canvas=FigureCanvasTkAgg(self.figure,master=plot)
        self.canvas.get_tk_widget().pack(fill="both",expand=True)
        NavigationToolbar2Tk(self.canvas,plot).update()
        ttk.Label(outer,textvariable=self.info,wraplength=1400,justify="left").pack(fill="x",pady=(10,0))

    def browse(self):
        chosen=filedialog.askdirectory(title="trialフォルダ、またはtrialsフォルダを選択",initialdir=self.folder.get())
        if chosen:self.set_folder(Path(chosen))

    def set_folder(self,path):
        path=path.expanduser().resolve()
        direct=(path/"problem.csv").is_file()
        parent=path.parent if direct else path
        paths=sorted([p for p in parent.iterdir() if p.is_dir() and (p/"problem.csv").is_file()]) if parent.is_dir() else []
        if direct and path not in paths:paths.append(path)
        self.paths={p.name:p for p in paths}
        self.folder.set(str(parent));self.trial_combo["values"]=list(self.paths)
        self.generation+=1;self.loaded=None
        self._clear_plot()
        if not paths:
            self.trial_var.set("");self.frame_var.set("");self.frame_combo["values"]=[]
            self.info.set("problem.csvを含むtrialが見つかりません。別のフォルダを選択してください。")
            return
        selected=path.name if direct else (self.trial_var.get() if self.trial_var.get() in self.paths else paths[-1].name)
        self.trial_var.set(selected);self.change_trial()

    def _clear_plot(self):
        self.figure.clear();self.ax=None;self.canvas.draw_idle()

    def change_trial(self,event=None):
        self.generation+=1;self.loaded=None;self.trial=None;self.fixed={};self._clear_plot()
        for child in self.fixed_frame.winfo_children():child.destroy()
        try:
            self.trial=discover_trial(self.paths[self.trial_var.get()])
            self.frame_combo["values"]=[f.label for f in self.trial.frames]
            self.frame_var.set(self.trial.frames[-1].label)
            self.change_frame()
        except (OSError,ValueError,KeyError,json.JSONDecodeError) as exc:
            self.frame_var.set("");self.frame_combo["values"]=[];self.info.set(str(exc))

    def change_frame(self,event=None):
        if self.trial is None:return
        frame=next((f for f in self.trial.frames if f.label==self.frame_var.get()),None)
        if frame is None:return
        self.generation+=1;generation=self.generation
        self.loaded=None;self._clear_plot();self.info.set("予測空間と、その時点の実測を読込中…")
        trial=self.trial
        def worker():
            try:self.messages.put((generation,load_frame(trial,frame),None))
            except Exception as exc:self.messages.put((generation,None,str(exc)))
        threading.Thread(target=worker,daemon=True).start()

    def _poll(self):
        try:
            while True:
                generation,data,error=self.messages.get_nowait()
                if generation!=self.generation:continue
                if error:self.info.set(error)
                else:self.apply_data(data)
        except queue.Empty:pass
        self.root.after(80,self._poll)

    def apply_data(self,data):
        self.loaded=data
        self.x_combo["values"]=data.axes;self.y_combo["values"]=data.axes
        if self.x.get() not in data.axes:self.x.set(data.axes[0])
        if self.y.get() not in data.axes or self.y.get()==self.x.get():self.y.set(next(n for n in data.axes if n!=self.x.get()))
        self.z_combo["values"]=data.outputs
        if self.z.get() not in data.outputs:self.z.set(data.trial.objective)
        self.source_combo["values"]=data.sources
        if self.source.get() not in data.sources:self.source.set(data.sources[0])
        # Keep chosen fixed values across steps when the same trial/grid is selected.
        defaults=fixed_defaults(data)
        self.fixed={name:float((values:=np.unique(data.predictions[name]))[np.argmin(np.abs(values-self.fixed.get(name,defaults[name])))])
                    for name in data.trial.parameters}
        self.build_fixed();self.draw()

    def axes_changed(self,event=None):
        if self.loaded is None:return
        if self.x.get()==self.y.get():
            alternatives=[n for n in self.loaded.axes if n!=self.x.get()]
            if event is not None and event.widget==self.y_combo:self.x.set(alternatives[0])
            else:self.y.set(alternatives[0])
        self.capture_fixed();self.build_fixed();self.schedule_draw()

    def capture_fixed(self):
        for name,var in self.fixed_variables.items():
            if name in (self.x.get(),self.y.get()):continue
            values=self.fixed_choices[name]
            if var.get() in values:self.fixed[name]=values[var.get()]

    def build_fixed(self):
        for child in self.fixed_frame.winfo_children():child.destroy()
        self.fixed_variables={};self.fixed_choices={};self.input_widgets={}
        if self.loaded is None:return
        for name in self.loaded.trial.parameters:
            values=np.unique(self.loaded.predictions[name])
            options={format(float(v),".12g"):float(v) for v in values}
            axis = "X" if name==self.x.get() else "Y" if name==self.y.get() else None
            display = (f"{values[0]:.12g} ～ {values[-1]:.12g}（{len(values)}水準）"
                       if axis else format(self.fixed[name],".12g"))
            var=tk.StringVar(value=display)
            self.fixed_variables[name]=var;self.fixed_choices[name]=options
            title=ttk.Frame(self.fixed_frame)
            title.pack(fill="x",pady=(8,2),padx=(0,8))
            title.columnconfigure(0,weight=1)
            ttk.Label(title,text=axis_label(self.loaded.trial,name),wraplength=200).grid(row=0,column=0,sticky="w")
            focus = focus_parameter(self.loaded, name)
            focus_state = FOCUS_STATE_LABELS.get(str(focus.get("state", "")).upper(), str(focus.get("state", ""))) if focus else ""
            axis_state = f"{axis}軸" if axis else "断面固定" if len(options) > 1 else "予測格子1水準"
            state_label = " / ".join(part for part in (focus_state, axis_state) if part)
            ttk.Label(title,text=state_label,foreground="#1665a5" if axis else "#68717a").grid(row=0,column=1,sticky="e",padx=(5,0))
            combo=ttk.Combobox(self.fixed_frame,textvariable=var,values=list(options),state="readonly",width=25)
            self.input_widgets[name]=combo
            combo.pack(fill="x",padx=(0,8));combo.bind("<<ComboboxSelected>>",self.schedule_draw)
            if axis or len(options)==1:combo.configure(state="disabled")

    def anchor_fixed(self):
        if self.loaded is None:return
        self.fixed=fixed_defaults(self.loaded);self.build_fixed();self.draw()

    def schedule_draw(self,event=None):
        if self.render_job is not None:self.root.after_cancel(self.render_job)
        self.render_job=self.root.after(100,self.draw)

    def draw(self):
        self.render_job=None
        if self.loaded is None:return
        self.capture_fixed()
        angles=(self.ax.elev,self.ax.azim) if self.ax is not None and hasattr(self.ax,"elev") else (26,-55)
        try:
            if self.view.get()==VIEW_FEASIBILITY:
                threshold=float(self.threshold.get())/100
                if not np.isfinite(threshold) or not 0<=threshold<=1:
                    raise ValueError("制約充足確率のしきい値は0～100%で指定してください。")
                self.ax,stats=draw_feasibility(self.figure,self.loaded,self.x.get(),self.y.get(),self.z.get(),
                                              self.source.get(),self.fixed,self.hide.get(),threshold)
                self.info.set(stats["summary"]); self.canvas.draw_idle(); return
            self.ax,stats=draw_plot(self.figure,self.loaded,self.x.get(),self.y.get(),self.z.get(),
                                    self.source.get(),self.fixed,self.scatter.get(),self.hide.get(),angles)
            frame=self.loaded.frame
            notice=f"予測更新なし：{frame.run_id}の面を使用。 " if frame.reused else ""
            scope = focus_scope_notice(self.loaded)
            scope = f"{scope} ／ " if scope else ""
            self.info.set(f"{notice}{scope}予測の学習データ {frame.training_rows}件 ／ この時点の実測 {frame.observed_rows}件 ／ "
                          f"断面の予測点 {stats['surface_points']}点 ／ 表示実測 {stats['shown_points']}点（断面一致 {stats['on_slice_points']}点）\n"
                          f"実測の取得元：{self.loaded.observation_source}")
            if fixture_notice(self.loaded): self.info.set(fixture_notice(self.loaded)+"\n"+self.info.get())
            self.canvas.draw_idle()
        except (ValueError,KeyError) as exc:
            self._clear_plot();self.info.set(str(exc))


def smoke_test(trial_path, preview=None):
    """Check history boundaries, slice values, GUI controls and rendering on real CSVs."""
    trial=discover_trial(trial_path)
    indices=sorted(set([0,min(1,len(trial.frames)-1),max(0,len(trial.frames)-2),len(trial.frames)-1]))
    tested=[]
    for index in indices:
        data=load_frame(trial,trial.frames[index])
        assert len(next(iter(data.observations.values())))==data.frame.observed_rows
        fixed=fixed_defaults(data)
        x,y=data.axes[:2]
        xx,yy,zz,points,matching=slice_data(data,x,y,trial.objective,"NN",fixed)
        assert zz.shape==(len(np.unique(yy)),len(np.unique(xx)))
        assert len(points)==np.isfinite(zz).sum()
        assert matching.any(),"Best measured condition should appear on the anchored slice"
        for i in points:
            a=int(np.searchsorted(yy[:,0],data.predictions[y][i]));b=int(np.searchsorted(xx[0],data.predictions[x][i]))
            assert zz[a,b]==data.predictions[trial.objective+"_nn_pred"][i]
        tested.append({"step":data.frame.step,"training_rows":data.frame.training_rows,"observed_rows":data.frame.observed_rows,"reused":data.frame.reused})
    # Exercise Tk integration without leaving a window open.
    root=tk.Tk();root.withdraw()
    try:
        app=PredictionViewer(root,trial.root)
        app.generation+=1  # Discard an asynchronous initial load; use the verified frame.
        app.apply_data(data)
        root.update()
        app.source.set(data.sources[-1]);app.draw();root.update()
        app.z.set(data.outputs[-1]);app.draw();root.update()
        app.scatter.set(SCATTER_SLICE);app.draw();root.update()
        app.z.set(trial.objective);app.source.set("NN");app.scatter.set(SCATTER_ALL);app.draw();root.update()
        app.view.set(VIEW_FEASIBILITY);app.source.set(data.sources[-1]);app.draw();root.update()
        probability,individual=feasibility_probabilities(data,data.sources[-1])
        assert np.all((probability>=0)&(probability<=1))
        selected,_=feasible_summary(data,probability,.95)
        assert not np.any(selected & ~data.allowed)
        assert np.array_equal(selected,data.allowed & (probability>=.95))
        app.threshold.set("101");app.draw();root.update()
        assert "0～100" in app.info.get()
        app.threshold.set("95");app.draw();root.update()
        if preview:
            preview=preview.resolve();preview.parent.mkdir(parents=True,exist_ok=True)
            app.figure.savefig(preview,dpi=150)
        app.view.set(VIEW_RESPONSE);app.draw();root.update()
        assert hasattr(app.ax,"elev"),"Switching back to the 3D view failed"
    finally:root.destroy()
    # Synthetic check: old step cannot see future rows without archived snapshots.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="v005_viewer_") as directory:
        base=Path(directory);(base/"data").mkdir()
        with (base/"data/experiments.csv").open("w",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream);writer.writerows([["experiment_id","x"],["old",1],["future",2]])
        mini=TrialData(base,{},[],"","",[])
        old=Frame("old",base,"run_0001",1,1,None)
        rows,_=observations_at(mini,old)
        assert len(rows)==1 and rows[0]["experiment_id"]=="old"
        try:observations_at(mini,Frame("bad",base,"run_0002",3,3,None))
        except ValueError:pass
        else:raise AssertionError("Missing historical observations were not rejected")
    print(json.dumps({"status":"OK","trial":str(trial.root),"frames":len(trial.frames),"verified":tested,
                      "gui":"Tk canvas, controls and 3D rendering passed","preview":str(preview) if preview else None},ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trial",type=Path,help="最初に表示するtrialフォルダ")
    parser.add_argument("--smoke-test",action="store_true",help="画面を残さずCSV・時点対応・3D描画・GUIを検証")
    parser.add_argument("--save-preview",type=Path,help="smoke-test時のグラフ保存先PNG")
    args=parser.parse_args()
    configure_fonts()
    if args.save_preview and not args.smoke_test:parser.error("--save-previewは--smoke-testと一緒に指定してください")
    if args.smoke_test:
        if args.trial is None:parser.error("--smoke-testには--trialが必要です")
        smoke_test(args.trial,args.save_preview)
    else:
        root=tk.Tk();PredictionViewer(root,args.trial);root.mainloop()


if __name__=="__main__":
    main()
