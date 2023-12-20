import json
from tqdm import tqdm
from matplotlib import pyplot as plt

cache_file = "cache/results.json"
version = "2023/12/21-#01"
model_name = "meta-llama/Llama-2-7b-hf"
min_question_count = 1000

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
    "max_q_value",
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
    "max_q_value": "Max Q value",
    "n_bits_uniform": "Uniform # of bits",
    "accuracy": "Accuracy",
    "answer_log_probability": "Answer log probability",
    "average_size": "KVcache size",
    "quantization_error": "Quantization error",
    "attention_error": "Attention error",
}


def visualize_results():
    with open(cache_file, "r") as f:
        cached_results = json.load(f)
    plt.figure(figsize=(5*len(relations), 5*2*len(params)))
    for param_idx, param_name in enumerate(tqdm(params)):
        for relation_idx, (metric_name_x, metric_name_y) in enumerate(relations):
            key_x, key_y, value_x, value_y = {}, {}, {}, {}
            for entry in cached_results:
                p, r = entry["params"], entry["results"]
                if p["version"] != version:
                    continue
                if p["model_name"] != model_name:
                    continue
                if p["question_count"] < min_question_count:
                    continue
                if param_name in p["key_quantizer"]:
                    key_param_data = p["key_quantizer"][param_name]
                    if key_param_data not in key_x:
                        key_x[key_param_data] = []
                    key_x[key_param_data].append(r[metric_name_x])
                    if key_param_data not in key_y:
                        key_y[key_param_data] = []
                    key_y[key_param_data].append(r[metric_name_y])
                if param_name in p["value_quantizer"]:
                    value_param_data = p["value_quantizer"][param_name]
                    if value_param_data not in value_x:
                        value_x[value_param_data] = []
                    value_x[value_param_data].append(r[metric_name_x])
                    if value_param_data not in value_y:
                        value_y[value_param_data] = []
                    value_y[value_param_data].append(r[metric_name_y])
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
    plt.savefig("figs/result.png", dpi=100)


if __name__ == "__main__":
    visualize_results()
