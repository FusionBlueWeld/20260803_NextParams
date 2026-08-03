"""ニューラルネットによる予測空間生成。"""

from .trainer import NeuralTrainingError, NeuralTrainingResult, train_network

__all__ = ["NeuralTrainingError", "NeuralTrainingResult", "train_network"]
