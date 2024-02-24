from tqdm import tqdm
from .base import Experiment
from dataclasses import asdict
from matplotlib import pyplot as plt
from functools import cached_property
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers

params = [
    "level",
    "symmetric",
    "method_name",
    "outliers_ratio",
    "use_attentions",
    "n_bits_min",
    "n_bits_max",
    "last_n_attentions",
    "target_quantization_error",
    "q_norm",
    "n_bits_uniform",
]
relations = [
    ("accuracy", "average_size"),
    ("accuracy", "quantization_error"),
    ("accuracy", "attention_error"),
    ("answer_log_probability", "average_size"),
    ("answer_log_probability", "quantization_error"),
    ("answer_log_probability", "attention_error"),
    ("average_size", "quantization_error"),
    ("average_size", "attention_error"),
]
translation = {
    "level": "Level",
    "symmetric": "Symmetric",
    "method_name": "Method name",
    "outliers_ratio": "Outliers ratio",
    "use_attentions": "Attention-aware",
    "n_bits_min": "Min # of bits",
    "n_bits_max": "Max # of bits",
    "last_n_attentions": "Last n attentions",
    "target_quantization_error": "Target error",
    "q_norm": "2-norm of query tensor",
    "n_bits_uniform": "Uniform # of bits",
    "accuracy": "Accuracy",
    "answer_log_probability": "Answer log probability",
    "average_size": "KVcache size",
    "quantization_error": "Quantization error",
    "attention_error": "Attention error",
}


class GridSearch(Experiment):
    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        key_quantizers = build_quantizers([{
            "key_or_value_cache": ["key"],
            "use_attentions": [False],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0, 0.01],
            "n_bits_uniform": [1, 2],
        }, {
            "key_or_value_cache": ["key"],
            "use_attentions": [True],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0, 0.01],
            "last_n_attentions": [1, 5],
            "target_quantization_error": [1.0, 3.0, 10.0, 30.0],
            "n_bits_min": [0, 1, 2],
            "n_bits_max": [2, 3, 4],
            "q_norm": [300],
        }])
        value_quantizers = build_quantizers([{
            "key_or_value_cache": ["value"],
            "use_attentions": [False],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0, 0.01],
            "n_bits_uniform": [1, 2],
        }, {
            "key_or_value_cache": ["value"],
            "use_attentions": [True],
            "method": ["uniform"],
            "level": ["token", "layer", "head"],
            "symmetric": [False],
            "outliers_ratio": [0, 0.01],
            "last_n_attentions": [1, 5],
            "target_quantization_error": [1.0, 3.0, 10.0, 30.0],
            "n_bits_min": [0, 1, 2],
            "n_bits_max": [2, 3, 4],
            "q_norm": [None],
        }])
        return list(zip(key_quantizers, value_quantizers))

    def process_result(self, results: list[EvaluationResult]):
        plt.figure(figsize=(5*len(relations), 5*2*len(params)))
        for param_idx, param_name in enumerate(tqdm(params)):
            for relation_idx, (metric_name_x, metric_name_y) in enumerate(relations):
                key_x, key_y, value_x, value_y = {}, {}, {}, {}
                for (key_quantizers, value_quantizers), result in zip(self.quantizer_list, results):
                    result = asdict(result)
                    if param_name in key_quantizers.params:
                        key_param_data = key_quantizers.params[param_name]
                        if key_param_data not in key_x:
                            key_x[key_param_data] = []
                        key_x[key_param_data].append(result[metric_name_x])
                        if key_param_data not in key_y:
                            key_y[key_param_data] = []
                        key_y[key_param_data].append(result[metric_name_y])
                    if param_name in value_quantizers.params:
                        value_param_data = value_quantizers.params[param_name]
                        if value_param_data not in value_x:
                            value_x[value_param_data] = []
                        value_x[value_param_data].append(result[metric_name_x])
                        if value_param_data not in value_y:
                            value_y[value_param_data] = []
                        value_y[value_param_data].append(result[metric_name_y])
                ax = plt.subplot(2*len(params), len(relations), (2*param_idx) * len(relations) + (relation_idx+1))
                for label in key_x:
                    ax.scatter(key_x[label], key_y[label], label=label)
                if len(key_x) > 0:
                    ax.legend()
                ax.set_title(f"{translation[param_name]} (Key)")
                ax.set_xlabel(translation[metric_name_x])
                ax.set_ylabel(translation[metric_name_y])
                ax.set_box_aspect(1)
                ax = plt.subplot(2*len(params), len(relations), (2*param_idx+1) * len(relations) + (relation_idx+1))
                for label in value_x:
                    ax.scatter(value_x[label], value_y[label], label=label)
                if len(value_x) > 0:
                    ax.legend()
                ax.set_title(f"{translation[param_name]} (Value)")
                ax.set_xlabel(translation[metric_name_x])
                ax.set_ylabel(translation[metric_name_y])
                ax.set_box_aspect(1)
        print(f"Rendering {2*len(params)*len(relations)} figures, it may take about 30 seconds...")
        plt.tight_layout()
        plt.savefig("figs/grid_search_results.png", dpi=100)
