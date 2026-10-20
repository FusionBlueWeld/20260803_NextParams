"""GP、NN、GP支持度を1つのHybridモデルとして統合します。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from ..preprocessing import PreparedData
from ..settings import (
    HYBRID_ACQUISITION_UNCERTAINTY_SCALE,
    HYBRID_UNCERTAINTY_CALIBRATION_SCALE,
    NN_ENSEMBLE_SIZE,
    NN_SEED,
    ProblemDefinition,
)
from ..knowledge import KnowledgeRule
from .gp_component import GaussianProcess, predict_in_chunks
from .nn_component import NeuralTrainingResult, ResidualNeuralNetwork, train_network
from .support import SupportModel


@dataclass(frozen=True)
class HybridPrediction:
    support: np.ndarray
    results: dict[str, dict[str, np.ndarray]]


@dataclass
class HybridModel:
    gp_models: dict[str, GaussianProcess]
    nn_model: "EnsembleNeuralNetwork"
    support_model: SupportModel
    result_columns: list[str]
    uncertainty_calibration_scale: float = 1.0
    acquisition_uncertainty_scale: float = 1.0

    def predict(self, normalized_x: np.ndarray) -> HybridPrediction:
        support = self.support_model.predict(normalized_x)
        # 旧v005 bundleの単一NNも、そのまま再読込して予測できるようにします。
        if hasattr(self.nn_model, "predict_members"):
            nn_member_values = self.nn_model.predict_members(normalized_x)
        else:
            nn_member_values = np.expand_dims(self.nn_model.predict(normalized_x), axis=0)
        nn_values = np.mean(nn_member_values, axis=0)
        nn_std_values = (
            np.std(nn_member_values, axis=0, ddof=1)
            if len(self.nn_model.members) > 1
            else np.zeros_like(nn_values)
        )
        results: dict[str, dict[str, np.ndarray]] = {}
        for index, column in enumerate(self.result_columns):
            # GPはNNの残差を学習し、測定点近傍だけを補正する。
            gp_mean, gp_std = predict_in_chunks(self.gp_models[column], normalized_x)
            nn_prediction = nn_values[:, index]
            nn_std = nn_std_values[:, index]
            hybrid_mean = nn_prediction + support * gp_mean
            raw_hybrid_std = np.sqrt(np.square(gp_std) + np.square(nn_std))
            calibration_scale = getattr(self, "uncertainty_calibration_scale", 1.0)
            acquisition_scale = getattr(self, "acquisition_uncertainty_scale", 1.0)
            hybrid_std = calibration_scale * raw_hybrid_std
            acquisition_std = acquisition_scale * raw_hybrid_std
            results[column] = {
                "gp_mean": gp_mean,
                "gp_std": gp_std,
                "nn_pred": nn_prediction,
                "nn_std": nn_std,
                "hybrid_mean": hybrid_mean,
                "raw_hybrid_std": raw_hybrid_std,
                "acquisition_std": acquisition_std,
                "uncertainty_calibration_scale": np.full(
                    len(normalized_x), calibration_scale
                ),
                "hybrid_std": hybrid_std,
            }
        return HybridPrediction(support=support, results=results)


@dataclass(frozen=True)
class HybridTrainingResult:
    model: HybridModel
    nn_epochs: int
    nn_normalized_mse: float
    nn_knowledge_loss: float = 0.0
    knowledge_rule_losses: tuple = ()
    retained_samples: int = 0
    source_samples: int = 0
    learning_seconds: float = 0.0


@dataclass
class EnsembleNeuralNetwork:
    """同じデータを異なる初期値で学習した小規模NN ensemble。"""

    members: tuple[ResidualNeuralNetwork, ...]

    @property
    def output_mean(self) -> np.ndarray:
        return self.members[0].output_mean

    @property
    def output_scale(self) -> np.ndarray:
        return self.members[0].output_scale

    def predict_members(self, inputs: np.ndarray) -> np.ndarray:
        return np.stack([member.predict(inputs) for member in self.members], axis=0)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.mean(self.predict_members(inputs), axis=0)


def train_hybrid_model(
    prepared: PreparedData,
    problem: ProblemDefinition,
    knowledge_rules: list[KnowledgeRule] | None = None,
    *,
    ensemble_size: int = NN_ENSEMBLE_SIZE,
    uncertainty_calibration_scale: float = HYBRID_UNCERTAINTY_CALIBRATION_SCALE,
    acquisition_uncertainty_scale: float = HYBRID_ACQUISITION_UNCERTAINTY_SCALE,
    max_training_samples: int = 256,
    max_nn_epochs: int = 2000,
    max_learning_seconds: float = 30.0,
) -> HybridTrainingResult:
    started = time.perf_counter()
    if type(max_nn_epochs) is not int or not 1 <= max_nn_epochs <= 2000:
        raise ValueError("NN epoch上限は1～2000にしてください。")
    if not np.isfinite(max_learning_seconds) or not 0.1 <= max_learning_seconds <= 600:
        raise ValueError("学習時間上限は0.1～600秒にしてください。")
    deadline = started + max_learning_seconds
    source_samples = prepared.unique_condition_count
    if not 2 <= max_training_samples <= 512:
        raise ValueError("学習条件の上限は2～512にしてください。")
    if source_samples > max_training_samples:
        recent = min(32, max_training_samples // 4)
        rng = np.random.default_rng(NN_SEED)
        indices = np.sort(np.concatenate((rng.choice(source_samples - recent, max_training_samples - recent, replace=False),
                                          np.arange(source_samples - recent, source_samples))))
        prepared = replace(prepared, raw_parameters=prepared.raw_parameters[indices],
                           normalized_parameters=prepared.normalized_parameters[indices],
                           result_means={k: v[indices] for k, v in prepared.result_means.items()},
                           repeat_counts=prepared.repeat_counts[indices])
    if not 1 <= ensemble_size <= 16:
        raise ValueError("ensemble_size must be between 1 and 16")
    if not np.isfinite(uncertainty_calibration_scale) or uncertainty_calibration_scale <= 0:
        raise ValueError("uncertainty_calibration_scale must be finite and positive")
    if not np.isfinite(acquisition_uncertainty_scale) or acquisition_uncertainty_scale <= 0:
        raise ValueError("acquisition_uncertainty_scale must be finite and positive")
    nn_trainings = [
        train_network(prepared, problem, knowledge_rules, seed=NN_SEED + member_index,
                      max_epochs=max_nn_epochs, deadline=deadline)
        for member_index in range(ensemble_size)
    ]
    nn_model = EnsembleNeuralNetwork(tuple(item.model for item in nn_trainings))
    nn_at_data = nn_model.predict(prepared.normalized_parameters)
    gp_models: dict[str, GaussianProcess] = {}
    for result in problem.result_variables:
        if time.perf_counter() >= deadline:
            raise ModelTrainingError("学習時間の予算に到達しました。推薦は公開しません。")
        index = [item.column for item in problem.result_variables].index(result.column)
        residual = prepared.result_means[result.column] - nn_at_data[:, index]
        gp_models[result.column] = GaussianProcess.fit(
            prepared.normalized_parameters,
            residual,
            prepared.result_noise_std[result.column],
        )
    model = HybridModel(
        gp_models=gp_models,
        nn_model=nn_model,
        support_model=SupportModel.fit(prepared.normalized_parameters),
        result_columns=[item.column for item in problem.result_variables],
        uncertainty_calibration_scale=uncertainty_calibration_scale,
        acquisition_uncertainty_scale=acquisition_uncertainty_scale,
    )
    return HybridTrainingResult(
        model=model,
        nn_epochs=max(item.epochs for item in nn_trainings),
        nn_normalized_mse=float(np.mean([item.normalized_mse for item in nn_trainings])),
        nn_knowledge_loss=float(np.mean([item.knowledge_loss for item in nn_trainings])),
        knowledge_rule_losses=nn_trainings[0].rule_losses,
        retained_samples=prepared.unique_condition_count,
        source_samples=source_samples,
        learning_seconds=time.perf_counter() - started,
    )
