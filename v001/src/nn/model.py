"""NumPyで実装した小規模な残差ニューラルネット。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ResidualNeuralNetwork:
    """線形スキップと隠れ層残差を持つ全結合ネットワークです。"""

    weights: dict[str, np.ndarray]
    output_mean: np.ndarray
    output_scale: np.ndarray

    def forward(
        self,
        inputs: np.ndarray,
        *,
        with_cache: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """標準化出力を計算し、学習時だけ中間値も返します。"""

        w = self.weights
        hidden_1 = np.tanh(inputs @ w["w1"] + w["b1"])
        hidden_2 = np.tanh(hidden_1 @ w["w2"] + w["b2"])
        residual = hidden_1 + hidden_2
        prediction = (
            inputs @ w["w_skip"]
            + w["b_skip"]
            + residual @ w["w_out"]
        )
        if not with_cache:
            return prediction
        return prediction, {
            "inputs": inputs,
            "hidden_1": hidden_1,
            "hidden_2": hidden_2,
            "residual": residual,
        }

    def predict(self, inputs: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
        """入力を分割して予測し、結果を元の単位へ戻します。"""

        outputs: list[np.ndarray] = []
        for start in range(0, len(inputs), chunk_size):
            normalized = self.forward(inputs[start : start + chunk_size])
            assert isinstance(normalized, np.ndarray)
            outputs.append(self.output_mean + self.output_scale * normalized)
        return np.vstack(outputs)
