"""既存Hybridの数値重みを明示的に保存・復元するadapter。"""
from __future__ import annotations
import numpy as np
from .hybrid.gp_component import GaussianProcess
from .hybrid.model import EnsembleNeuralNetwork, HybridModel
from .hybrid.nn_component import ResidualNeuralNetwork
from .hybrid.support import SupportModel

WEIGHT_NAMES = ("w1", "b1", "w2", "b2", "w_skip", "b_skip", "w_out")


def serialize_hybrid(model) -> dict:
    arrays = {"model_scales": np.array([model.uncertainty_calibration_scale, model.acquisition_uncertainty_scale]),
              "member_count": np.asarray(len(model.nn_model.members)),
              "support_train_x": model.support_model.train_x,
              "support_cholesky": model.support_model.cholesky,
              "support_settings": np.array([model.support_model.length_scale, model.support_model.multiplier])}
    for index, member in enumerate(model.nn_model.members):
        arrays[f"nn_{index}_mean"] = member.output_mean
        arrays[f"nn_{index}_scale"] = member.output_scale
        for name in WEIGHT_NAMES:
            arrays[f"nn_{index}_{name}"] = member.weights[name]
    for index, name in enumerate(model.result_columns):
        gp = model.gp_models[name]
        for field in ("train_x", "cholesky", "alpha"):
            arrays[f"gp_{index}_{field}"] = getattr(gp, field)
        arrays[f"gp_{index}_settings"] = np.array([gp.y_mean, gp.y_scale, gp.length_scale, gp.noise_ratio])
    return arrays


class OrderedHybrid:
    def __init__(self, model, permutation):
        self.model, self.permutation = model, permutation
    def predict(self, values):
        return self.model.predict(values[:, self.permutation])


def restore_hybrid(arrays, outputs, input_count=None):
    members = []
    try:
        n, dimensions = arrays["support_train_x"].shape
        if not 2 <= n <= 512 or not 1 <= dimensions <= 64 or (input_count is not None and dimensions != input_count):
            raise ValueError("Hybridの学習配列shapeが不正です。")
        if arrays["support_cholesky"].shape != (n, n) or np.any(np.diag(arrays["support_cholesky"]) <= 0):
            raise ValueError("Hybrid支持度の配列が不正です。")
    except (KeyError, TypeError) as error:
        raise ValueError("Hybridの必須配列がありません。") from error
    count = int(arrays["member_count"])
    if not 1 <= count <= 16:
        raise ValueError("NN member数が不正です。")
    for index in range(count):
        weights = {name: arrays[f"nn_{index}_{name}"] for name in WEIGHT_NAMES}
        hidden = weights["w1"].shape[1]
        expected = {"w1": (dimensions, hidden), "b1": (hidden,), "w2": (hidden, hidden),
                    "b2": (hidden,), "w_skip": (dimensions, len(outputs)), "b_skip": (len(outputs),),
                    "w_out": (hidden, len(outputs))}
        if not 1 <= hidden <= 256 or any(weights[name].shape != shape for name, shape in expected.items()):
            raise ValueError("HybridのNN重みshapeが不正です。")
        mean, scale = arrays[f"nn_{index}_mean"], arrays[f"nn_{index}_scale"]
        if mean.shape != (len(outputs),) or scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("Hybridの出力scaleが不正です。")
        members.append(ResidualNeuralNetwork(weights, mean, scale))
    gps = {}
    for index, name in enumerate(outputs):
        mean, scale, length, noise = arrays[f"gp_{index}_settings"]
        if arrays[f"gp_{index}_train_x"].shape != (n, dimensions) or arrays[f"gp_{index}_cholesky"].shape != (n, n) or arrays[f"gp_{index}_alpha"].shape != (n,):
            raise ValueError("HybridのGP shapeが不正です。")
        if scale <= 0 or length <= 0 or noise < 0 or np.any(np.diag(arrays[f"gp_{index}_cholesky"]) <= 0):
            raise ValueError("HybridのGP scaleが不正です。")
        gps[name] = GaussianProcess(arrays[f"gp_{index}_train_x"], float(mean), float(scale),
                                    float(length), float(noise), arrays[f"gp_{index}_cholesky"], arrays[f"gp_{index}_alpha"])
    length, multiplier = arrays["support_settings"]
    support = SupportModel(arrays["support_train_x"], arrays["support_cholesky"], float(length), float(multiplier))
    calibration, acquisition = arrays["model_scales"]
    if length <= 0 or multiplier < 0 or calibration <= 0 or acquisition <= 0:
        raise ValueError("Hybridの不確実性/支持度scaleが不正です。")
    model = HybridModel(gps, EnsembleNeuralNetwork(tuple(members)), support, list(outputs), float(calibration), float(acquisition))
    if "input_permutation" in arrays:
        permutation = arrays["input_permutation"]
        if permutation.dtype.kind not in "iu" or sorted(permutation.tolist()) != list(range(dimensions)):
            raise ValueError("Hybridの入力列順が不正です。")
        return OrderedHybrid(model, permutation.astype(int))
    return model
