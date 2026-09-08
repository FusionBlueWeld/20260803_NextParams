"""GP、NN、GP支持度を1つのHybridモデルとして統合します。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..preprocessing import PreparedData
from ..settings import ProblemDefinition
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
    nn_model: ResidualNeuralNetwork
    support_model: SupportModel
    result_columns: list[str]

    def predict(self, normalized_x: np.ndarray) -> HybridPrediction:
        support = self.support_model.predict(normalized_x)
        nn_values = self.nn_model.predict(normalized_x)
        results: dict[str, dict[str, np.ndarray]] = {}
        for index, column in enumerate(self.result_columns):
            # GPはNNの残差を学習し、測定点近傍だけを補正する。
            gp_mean, gp_std = predict_in_chunks(self.gp_models[column], normalized_x)
            nn_prediction = nn_values[:, index]
            hybrid_mean = nn_prediction + support * gp_mean
            results[column] = {
                "gp_mean": gp_mean,
                "gp_std": gp_std,
                "nn_pred": nn_prediction,
                "hybrid_mean": hybrid_mean,
                # 未観測域の不確かさを0にしない（平均の補正だけtaperする）。
                "hybrid_std": gp_std,
            }
        return HybridPrediction(support=support, results=results)


@dataclass(frozen=True)
class HybridTrainingResult:
    model: HybridModel
    nn_epochs: int
    nn_normalized_mse: float
    nn_knowledge_loss: float = 0.0
    knowledge_rule_losses: tuple = ()


def train_hybrid_model(
    prepared: PreparedData,
    problem: ProblemDefinition,
    knowledge_rules: list[KnowledgeRule] | None = None,
) -> HybridTrainingResult:
    nn_training: NeuralTrainingResult = train_network(prepared, problem, knowledge_rules)
    nn_at_data = nn_training.model.predict(prepared.normalized_parameters)
    gp_models: dict[str, GaussianProcess] = {}
    for result in problem.result_variables:
        index = [item.column for item in problem.result_variables].index(result.column)
        residual = prepared.result_means[result.column] - nn_at_data[:, index]
        gp_models[result.column] = GaussianProcess.fit(
            prepared.normalized_parameters,
            residual,
            prepared.result_noise_std[result.column],
        )
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
        nn_knowledge_loss=nn_training.knowledge_loss,
        knowledge_rule_losses=nn_training.rule_losses,
    )
