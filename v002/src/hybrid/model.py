"""GP、NN、GP支持度を1つのHybridモデルとして統合します。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..preprocessing import PreparedData
from ..settings import ProblemDefinition
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
    nn_model: ResidualNeuralNetwork
    support_model: SupportModel
    result_columns: list[str]

    def predict(self, normalized_x: np.ndarray) -> HybridPrediction:
        support = self.support_model.predict(normalized_x)
        nn_values = self.nn_model.predict(normalized_x)
        results: dict[str, dict[str, np.ndarray]] = {}
        for index, column in enumerate(self.result_columns):
            gp_mean, gp_std = predict_in_chunks(self.gp_models[column], normalized_x)
            nn_prediction = nn_values[:, index]
            hybrid_mean = blend_predictions(gp_mean, nn_prediction, support)
            results[column] = {
                "gp_mean": gp_mean,
                "gp_std": gp_std,
                "nn_pred": nn_prediction,
                "hybrid_mean": hybrid_mean,
                "hybrid_std": gp_std,
            }
        return HybridPrediction(support=support, results=results)


@dataclass(frozen=True)
class HybridTrainingResult:
    model: HybridModel
    nn_epochs: int
    nn_normalized_mse: float


def blend_predictions(
    gp_mean: np.ndarray,
    nn_prediction: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """support=0でGP、support=1でNNとなる凸結合を返します。"""

    if gp_mean.shape != nn_prediction.shape or gp_mean.shape != support.shape:
        raise ValueError("GP、NN、支持度の配列形状が一致しません。")
    safe_support = np.clip(support, 0.0, 1.0)
    return gp_mean + safe_support * (nn_prediction - gp_mean)


def train_hybrid_model(
    prepared: PreparedData,
    problem: ProblemDefinition,
) -> HybridTrainingResult:
    gp_models: dict[str, GaussianProcess] = {}
    for result in problem.result_variables:
        gp_models[result.column] = GaussianProcess.fit(
            prepared.normalized_parameters,
            prepared.result_means[result.column],
            prepared.result_noise_std[result.column],
        )
    nn_training: NeuralTrainingResult = train_network(prepared, problem)
    model = HybridModel(
        gp_models=gp_models,
        nn_model=nn_training.model,
        support_model=SupportModel.fit(prepared.normalized_parameters),
        result_columns=[item.column for item in problem.result_variables],
    )
    return HybridTrainingResult(
        model=model,
        nn_epochs=nn_training.epochs,
        nn_normalized_mse=nn_training.normalized_mse,
    )
