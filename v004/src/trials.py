"""v004のtrial作成・存在確認・実験CSV準備。入力内容は担当モジュールで検査します。"""

from __future__ import annotations

from .trial_inputs import load_trial_inputs

import shutil
import tempfile
from pathlib import Path
from .data_loader import read_csv, write_csv_atomic, write_problem_template
from .knowledge import write_knowledge_template
from .policies import write_search_regions_template
from .settings import EXPERIMENT_FILE_NAME, PROBLEM_FILE_NAME, TRIALS_ROOT, trial_path
from .validation import UserInputError, validate_trial_name
from .stopping import write_stop_settings_template
from .reporting import print_problem_summary


def create_trial(trial_name: str) -> None:
    """trialフォルダ、problem.csv、data/outputフォルダを作ります。"""

    validate_trial_name(trial_name)
    TRIALS_ROOT.mkdir(parents=True, exist_ok=True)
    destination = trial_path(trial_name)

    if destination.exists():
        raise UserInputError(
            f"{trial_name}は既に存在します。\n"
            "既存ファイルは変更していません。別のtrial名を指定してください。"
        )

    # 完成途中のtrialを残さないため、一時フォルダで作ってから移動します。
    temporary = Path(tempfile.mkdtemp(prefix=f".{trial_name}_", dir=TRIALS_ROOT))
    try:
        (temporary / "data").mkdir()
        (temporary / "output").mkdir()
        write_problem_template(temporary / PROBLEM_FILE_NAME)
        write_knowledge_template(temporary / "knowledge_constraints.csv")
        write_search_regions_template(temporary / "search_regions.csv")
        write_stop_settings_template(temporary / "stop_settings.csv")
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    print(f"{trial_name}を作成しました。")
    print("\n次のファイルを編集してください。")
    print(destination / PROBLEM_FILE_NAME)
    print("\n編集後、次のコマンドを実行してください。")
    print(f"python main.py --prepare {trial_name}")


def prepare_trial(trial_name: str) -> None:
    """problem.csvを検査し、experiments.csvのヘッダーを作ります。"""

    validate_trial_name(trial_name)
    current_trial = require_trial(trial_name)
    inputs = load_trial_inputs(current_trial)
    problem = inputs.problem

    experiment_path = current_trial / "data" / EXPERIMENT_FILE_NAME
    expected_header = problem.experiment_header

    if experiment_path.exists():
        current_header, _ = read_csv(experiment_path)
        if current_header != expected_header:
            raise UserInputError(
                "既存のexperiments.csvとproblem.csvの列が一致しません。\n"
                f"必要: {','.join(expected_header)}\n"
                f"現在: {','.join(current_header)}\n"
                "実験データを確認し、列を手動で合わせてください。"
            )
        print("experiments.csvは既に準備されています。既存ファイルは変更していません。")
    else:
        write_csv_atomic(experiment_path, expected_header, [])
        print("problem.csvを確認しました。")
        print(f"\n実験結果の入力ファイルを作成しました。\n{experiment_path}")

    print_problem_summary(problem)
    print("\n実験結果を入力後、次のコマンドを実行してください。")
    print(f"python main.py --run {trial_name}")


def require_trial(trial_name: str) -> Path:
    """指定trialがなければ、作成方法を案内します。"""

    current_trial = trial_path(trial_name)
    if not current_trial.is_dir():
        raise UserInputError(
            f"{trial_name}が見つかりません。\n"
            f"新しく作る場合: python main.py --new {trial_name}"
        )
    return current_trial

