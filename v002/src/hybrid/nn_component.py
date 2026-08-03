"""Hybrid内部で直接予測を作る小規模な残差ニューラルネット。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..preprocessing import PreparedData
from ..settings import (
    NN_EARLY_STOPPING_PATIENCE,
    NN_HIDDEN_SIZE,
    NN_L2_WEIGHT,
    NN_LEARNING_RATE,
    NN_MAX_EPOCHS,
    NN_MIN_IMPROVEMENT,
    NN_SEED,
    ProblemDefinition,
)


class NeuralTrainingError(Exception):
    """NN学習を正常に完了できなかった場合のエラーです。"""


@dataclass
class ResidualNeuralNetwork:
    weights: dict[str, np.ndarray]
    output_mean: np.ndarray
    output_scale: np.ndarray

    def forward(
        self,
        inputs: np.ndarray,
        *,
        with_cache: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        w = self.weights
        hidden_1 = np.tanh(inputs @ w["w1"] + w["b1"])
        hidden_2 = np.tanh(hidden_1 @ w["w2"] + w["b2"])
        residual = hidden_1 + hidden_2
        prediction = inputs @ w["w_skip"] + w["b_skip"] + residual @ w["w_out"]
        if not with_cache:
            return prediction
        return prediction, {
            "inputs": inputs,
            "hidden_1": hidden_1,
            "hidden_2": hidden_2,
            "residual": residual,
        }

    def predict(self, inputs: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in range(0, len(inputs), chunk_size):
            normalized = self.forward(inputs[start : start + chunk_size])
            assert isinstance(normalized, np.ndarray)
            outputs.append(self.output_mean + self.output_scale * normalized)
        return np.vstack(outputs)


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: ResidualNeuralNetwork
    epochs: int
    normalized_mse: float


def train_network(
    prepared: PreparedData,
    problem: ProblemDefinition,
) -> NeuralTrainingResult:
    inputs = np.asarray(prepared.normalized_parameters, dtype=float)
    raw_outputs = np.column_stack(
        [prepared.result_means[item.column] for item in problem.result_variables]
    )
    output_mean = np.mean(raw_outputs, axis=0)
    output_scale = np.std(raw_outputs, axis=0)
    fallback = np.maximum(np.abs(output_mean) * 0.05, 1.0)
    output_scale = np.where(output_scale < 1e-12, fallback, output_scale)
    targets = (raw_outputs - output_mean) / output_scale

    rng = np.random.default_rng(NN_SEED)
    input_count = inputs.shape[1]
    output_count = targets.shape[1]
    hidden = NN_HIDDEN_SIZE
    weights = {
        "w1": rng.normal(0.0, np.sqrt(2.0 / (input_count + hidden)), (input_count, hidden)),
        "b1": np.zeros(hidden),
        "w2": rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, hidden)),
        "b2": np.zeros(hidden),
        "w_skip": rng.normal(0.0, 0.05, (input_count, output_count)),
        "b_skip": np.zeros(output_count),
        "w_out": rng.normal(0.0, np.sqrt(1.0 / hidden), (hidden, output_count)),
    }
    model = ResidualNeuralNetwork(weights, output_mean, output_scale)
    first_moment = {name: np.zeros_like(value) for name, value in weights.items()}
    second_moment = {name: np.zeros_like(value) for name, value in weights.items()}
    best_weights = {name: value.copy() for name, value in weights.items()}
    best_loss = float("inf")
    stale_epochs = 0
    completed_epochs = 0

    for epoch in range(1, NN_MAX_EPOCHS + 1):
        prediction, cache = model.forward(inputs, with_cache=True)
        assert isinstance(prediction, np.ndarray)
        error = prediction - targets
        data_loss = float(np.mean(np.square(error)))
        if not np.isfinite(data_loss):
            raise NeuralTrainingError("NN学習中の損失が有限値ではなくなりました。")
        if best_loss - data_loss > NN_MIN_IMPROVEMENT:
            best_loss = data_loss
            best_weights = {name: value.copy() for name, value in weights.items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        completed_epochs = epoch
        if stale_epochs >= NN_EARLY_STOPPING_PATIENCE:
            break

        gradient_output = 2.0 * error / error.size
        gradients: dict[str, np.ndarray] = {
            "w_skip": cache["inputs"].T @ gradient_output,
            "b_skip": np.sum(gradient_output, axis=0),
            "w_out": cache["residual"].T @ gradient_output,
        }
        gradient_residual = gradient_output @ weights["w_out"].T
        gradient_hidden_2_input = gradient_residual * (1.0 - np.square(cache["hidden_2"]))
        gradients["w2"] = cache["hidden_1"].T @ gradient_hidden_2_input
        gradients["b2"] = np.sum(gradient_hidden_2_input, axis=0)
        gradient_hidden_1 = gradient_residual + gradient_hidden_2_input @ weights["w2"].T
        gradient_hidden_1_input = gradient_hidden_1 * (1.0 - np.square(cache["hidden_1"]))
        gradients["w1"] = cache["inputs"].T @ gradient_hidden_1_input
        gradients["b1"] = np.sum(gradient_hidden_1_input, axis=0)
        for name, value in weights.items():
            if name.startswith("w"):
                gradients[name] += 2.0 * NN_L2_WEIGHT * value

        for name in weights:
            first_moment[name] = 0.9 * first_moment[name] + 0.1 * gradients[name]
            second_moment[name] = 0.999 * second_moment[name] + 0.001 * np.square(gradients[name])
            corrected_first = first_moment[name] / (1.0 - 0.9**epoch)
            corrected_second = second_moment[name] / (1.0 - 0.999**epoch)
            weights[name] -= NN_LEARNING_RATE * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )

    model.weights = best_weights
    if not np.isfinite(best_loss):
        raise NeuralTrainingError("NN学習結果を生成できませんでした。")
    return NeuralTrainingResult(model, completed_epochs, best_loss)
