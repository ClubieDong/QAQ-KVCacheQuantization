from .base import Experiment
from itertools import chain, product
from matplotlib import pyplot as plt
from functools import cached_property
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers


class KeyValueDifference(Experiment):
    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        key_quantizers_1 = build_quantizers([{
            "key_or_value_cache": ["key"],
            "use_attentions": [False],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0],
            "n_bits_uniform": [1, 2, 3, 4, 5, 6, 7, 8],
        }])
        value_quantizers_1 = build_quantizers([{
            "key_or_value_cache": ["value"],
            "level": ["no-quantization"],
        }])
        key_quantizers_2 = build_quantizers([{
            "key_or_value_cache": ["key"],
            "level": ["no-quantization"],
        }])
        value_quantizers_2 = build_quantizers([{
            "key_or_value_cache": ["value"],
            "use_attentions": [False],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0],
            "n_bits_uniform": [1, 2, 3, 4, 5, 6, 7, 8],
        }])
        return list(chain(
            product(key_quantizers_1, value_quantizers_1),
            product(key_quantizers_2, value_quantizers_2),
        ))

    def process_result(self, results: list[EvaluationResult]):
        enabled_series = ["token", "layer", "head"]
        series: dict[str, list[float]] = {}
        for (key_quantizer, value_quantizer), result in zip(self.quantizer_list, results):
            if key_quantizer.level == "no-quantization":
                if value_quantizer.level not in enabled_series:
                    continue
                name = f"Value ({value_quantizer.level}-level)"
                if name not in series:
                    series[name] = [None] * 8
                series[name][value_quantizer.n_bits_uniform-1] = result.accuracy
            elif value_quantizer.level == "no-quantization":
                if key_quantizer.level not in enabled_series:
                    continue
                name = f"Key ({key_quantizer.level}-level)"
                if name not in series:
                    series[name] = [None] * 8
                series[name][key_quantizer.n_bits_uniform-1] = result.accuracy
        for name, data in series.items():
            plt.plot([1,2,3,4,5,6,7,8], data, label=name, linestyle="solid" if name.startswith("Key") else "dashed")
        plt.legend()
        plt.xlabel("# of bits")
        plt.ylabel("Accuracy")
        plt.savefig("figs/key_value_difference.png", dpi=400)
        print(series)
