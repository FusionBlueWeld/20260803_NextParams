"""知見制約を損失へ直接加える小規模NumPyニューラルネット。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..knowledge import KnowledgeRule, build_knowledge_points
from ..knowledge_loss import RuleLoss, calculate_rule_loss
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

    def forward(self, inputs: np.ndarray, *, with_cache: bool = False):
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
        outputs = []
        for start in range(0, len(inputs), chunk_size):
            normalized = self.forward(inputs[start:start + chunk_size])
            outputs.append(self.output_mean + self.output_scale * normalized)
        return np.vstack(outputs) if outputs else np.empty((0, len(self.output_mean)))


@dataclass(frozen=True)
class NeuralTrainingResult:
    model: ResidualNeuralNetwork
    epochs: int
    normalized_mse: float
    knowledge_loss: float = 0.0
    rule_losses: tuple[RuleLoss, ...] = field(default_factory=tuple)


def _zero_gradients(weights):
    return {name: np.zeros_like(value) for name, value in weights.items()}


def _backward(weights, cache, output_gradient):
    """出力勾配を重み勾配へ戻す。制約損失も同じ経路を通る。"""
    gradients = {
        "w_skip": cache["inputs"].T @ output_gradient,
        "b_skip": np.sum(output_gradient, axis=0),
        "w_out": cache["residual"].T @ output_gradient,
    }
    grad_residual = output_gradient @ weights["w_out"].T
    grad_h2_in = grad_residual * (1.0 - np.square(cache["hidden_2"]))
    gradients["w2"] = cache["hidden_1"].T @ grad_h2_in
    gradients["b2"] = np.sum(grad_h2_in, axis=0)
    grad_h1 = grad_residual + grad_h2_in @ weights["w2"].T
    grad_h1_in = grad_h1 * (1.0 - np.square(cache["hidden_1"]))
    gradients["w1"] = cache["inputs"].T @ grad_h1_in
    gradients["b1"] = np.sum(grad_h1_in, axis=0)
    return gradients


def _constraint_step(model, problem, rules, points):
    total_grad = _zero_gradients(model.weights)
    summaries = []
    output_index = {item.column: i for i, item in enumerate(problem.result_variables)}
    for rule in rules:
        point_set = points.get(rule.rule_id)
        if point_set is None:
            continue
        index = output_index[rule.target]
        if rule.wrt is None:
            normalized, cache = model.forward(point_set.points, with_cache=True)
            raw = model.output_mean[index] + model.output_scale[index] * normalized[:, index]
            summary, raw_grad, _ = calculate_rule_loss(
                rule,
                point_set,
                raw,
                loss_scale=model.output_scale[index],
            )
            output_grad = np.zeros_like(normalized)
            output_grad[:, index] = raw_grad * model.output_scale[index]
            gradients = _backward(model.weights, cache, output_grad)
        else:
            # ペアを一度にforwardし、制約点が多い場合も学習を現実的な速度にする。
            if not point_set.pairs:
                continue
            before_x = np.asarray([pair[0] for pair in point_set.pairs], dtype=float)
            after_x = np.asarray([pair[1] for pair in point_set.pairs], dtype=float)
            before_n, before_cache = model.forward(before_x, with_cache=True)
            after_n, after_cache = model.forward(after_x, with_cache=True)
            before_values = (
                model.output_mean[index]
                + model.output_scale[index] * before_n[:, index]
            )
            after_values = (
                model.output_mean[index]
                + model.output_scale[index] * after_n[:, index]
            )
            pairs = list(zip(before_values.tolist(), after_values.tolist()))
            summary, pair_grad, _ = calculate_rule_loss(
                rule,
                point_set,
                np.empty((0,)),
                pairs,
                loss_scale=model.output_scale[index],
            )
            gradients = _zero_gradients(model.weights)
            before_grad = np.zeros_like(before_n)
            after_grad = np.zeros_like(after_n)
            before_grad[:, index] = pair_grad[0::2] * model.output_scale[index]
            after_grad[:, index] = pair_grad[1::2] * model.output_scale[index]
            for name, local in _backward(model.weights, before_cache, before_grad).items():
                gradients[name] += local
            for name, local in _backward(model.weights, after_cache, after_grad).items():
                gradients[name] += local
        for name in total_grad:
            total_grad[name] += gradients[name]
        summaries.append(summary)
    return total_grad, float(sum(item.loss for item in summaries)), summaries


def train_network(
    prepared: PreparedData,
    problem: ProblemDefinition,
    knowledge_rules: list[KnowledgeRule] | None = None,
) -> NeuralTrainingResult:
    """データ適合損失と知見損失を同時に最小化します。"""
    rules = [r for r in (knowledge_rules or []) if r.enabled]
    inputs = np.asarray(prepared.normalized_parameters, dtype=float)
    raw_outputs = np.column_stack(
        [prepared.result_means[item.column] for item in problem.result_variables]
    )
    output_mean, output_scale = np.mean(raw_outputs, axis=0), np.std(raw_outputs, axis=0)
    output_scale = np.where(
        output_scale < 1e-12,
        np.maximum(np.abs(output_mean) * 0.05, 1.0),
        output_scale,
    )
    targets = (raw_outputs - output_mean) / output_scale
    rng = np.random.default_rng(NN_SEED)
    input_count = inputs.shape[1]
    output_count = targets.shape[1]
    hidden = NN_HIDDEN_SIZE
    weights = {
        "w1": rng.normal(
            0.0,
            np.sqrt(2.0 / (input_count + hidden)),
            (input_count, hidden),
        ),
        "b1": np.zeros(hidden),
        "w2": rng.normal(
            0.0,
            np.sqrt(1.0 / hidden),
            (hidden, hidden),
        ),
        "b2": np.zeros(hidden),
        "w_skip": rng.normal(0.0, 0.05, (input_count, output_count)),
        "b_skip": np.zeros(output_count),
        "w_out": rng.normal(
            0.0,
            np.sqrt(1.0 / hidden),
            (hidden, output_count),
        ),
    }
    model = ResidualNeuralNetwork(weights, output_mean, output_scale)
    point_sets = build_knowledge_points(problem, rules)
    first, second = _zero_gradients(weights), _zero_gradients(weights)
    best_weights = {name: value.copy() for name, value in weights.items()}
    best_total, best_data, best_summary = float("inf"), float("inf"), []
    stale, completed = 0, 0
    for epoch in range(1, NN_MAX_EPOCHS + 1):
        normalized_prediction, cache = model.forward(inputs, with_cache=True)
        error = normalized_prediction - targets
        data_loss = float(np.mean(np.square(error)))
        constraint_grad, knowledge_loss, summaries = _constraint_step(
            model, problem, rules, point_sets
        )
        l2_loss = sum(
            float(np.sum(np.square(value)))
            for name, value in weights.items()
            if name.startswith("w")
        )
        total_loss = data_loss + knowledge_loss + NN_L2_WEIGHT * l2_loss
        if not np.isfinite(total_loss):
            raise NeuralTrainingError("NN学習中の損失が有限値ではなくなりました。")
        if best_total - total_loss > NN_MIN_IMPROVEMENT:
            best_total, best_data, best_summary = total_loss, data_loss, summaries
            best_weights = {name: value.copy() for name, value in weights.items()}
            stale = 0
        else:
            stale += 1
        completed = epoch
        if stale >= NN_EARLY_STOPPING_PATIENCE:
            break
        gradients = _backward(weights, cache, 2.0 * error / error.size)
        for name in gradients:
            gradients[name] += constraint_grad[name]
            if name.startswith("w"):
                gradients[name] += 2.0 * NN_L2_WEIGHT * weights[name]
            first[name] = 0.9 * first[name] + 0.1 * gradients[name]
            second[name] = 0.999 * second[name] + 0.001 * np.square(gradients[name])
            corrected_first = first[name] / (1.0 - 0.9 ** epoch)
            corrected_second = second[name] / (1.0 - 0.999 ** epoch)
            weights[name] -= (
                NN_LEARNING_RATE
                * corrected_first
                / (np.sqrt(corrected_second) + 1e-8)
            )
    if not np.isfinite(best_total):
        raise NeuralTrainingError("NN学習結果を生成できませんでした。")
    model.weights = best_weights
    return NeuralTrainingResult(
        model,
        completed,
        best_data,
        float(sum(item.loss for item in best_summary)),
        tuple(best_summary),
    )
