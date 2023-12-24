import os
import torch
from dataclasses import asdict
from evaluator import Evaluator
from question import load_questions
from quantizer import build_quantizers
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

# NOTE: uncomment this line when debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

model_name = "meta-llama/Llama-2-7b-hf"
device = torch.device("cuda:0")
max_memory = {0: "6GB", 1: "9GB", "cpu": "32GB"}
dtype = torch.float16
question_count = 1000
# Change version if you made breaking changes to the code so that the cached results will be invalidated.
version = "2023/12/24-#01"
cache_file = "cache/results.json"

key_quantizer_configs = [{
    "key_or_value_cache": ["key"],
    "use_attentions": [False],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "outliers_ratio": [0.01],
    "n_bits_uniform": [2, 3, 4, 6, 8],
}, {
    "key_or_value_cache": ["key"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "outliers_ratio": [0.01],
    "last_n_attentions": [5],
    "target_quantization_error": [
        1000, 10000, 100000, 1000000,
        1000, 10000, 100000, 1000000,
        1000, 10000, 100000, 1000000,
        1000, 10000, 100000, 1000000,
    ],
    "n_bits_min": [1, 2],
    "n_bits_max": [8],
    "max_q_value": [3],
}]
value_quantizer_configs = [{
    "key_or_value_cache": ["value"],
    "use_attentions": [False],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "outliers_ratio": [0.01],
    "n_bits_uniform": [2, 3, 4, 6, 8],
}, {
    "key_or_value_cache": ["value"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "outliers_ratio": [0.01],
    "last_n_attentions": [5],
    "target_quantization_error": [
        1, 1, 1, 1,
        10, 10, 10, 10,
        100, 100, 100, 100,
        1000, 1000, 1000, 1000,
    ],
    "n_bits_min": [1, 2],
    "n_bits_max": [8],
}]

# key_quantizer_configs = [{
#     "key_or_value_cache": ["key"],
#     "level": ["no-quantization"],
# }]
# value_quantizer_configs = [{
#     "key_or_value_cache": ["value"],
#     "level": ["no-quantization"],
# }]

# key_quantizer_configs = [{
#     "key_or_value_cache": ["key"],
#     "use_attentions": [False],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0.01],
#     "n_bits_uniform": [4],
# }]
# value_quantizer_configs = [{
#     "key_or_value_cache": ["value"],
#     "use_attentions": [False],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0.01],
#     "n_bits_uniform": [4],
# }]

# key_quantizer_configs = [{
#     "key_or_value_cache": ["key"],
#     "use_attentions": [True],
#     "method": ["uniform"],
#     "level": ["token"],
#     "symmetric": [False],
#     "outliers_ratio": [0.01],
#     "last_n_attentions": [5],
#     "target_quantization_error": [10000, 30000, 100000],
#     "n_bits_min": [1],
#     "n_bits_max": [8],
#     "max_q_value": [3],
# }]
# value_quantizer_configs = [{
#     "key_or_value_cache": ["value"],
#     "use_attentions": [True],
#     "method": ["uniform"],
#     "level": ["token"],
#     "symmetric": [False],
#     "outliers_ratio": [0.01],
#     "last_n_attentions": [5],
#     "target_quantization_error": [300, 300, 300],
#     "n_bits_min": [1],
#     "n_bits_max": [8],
# }]


def main():
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    with init_empty_weights():
        model = LlamaForCausalLM(LlamaConfig.from_pretrained(model_name))
    device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=dtype, no_split_module_classes=LlamaForCausalLM._no_split_modules)
    if any(x == "cpu" or x == "disk" for x in device_map.values()):
        print("Warning: CPU offloading enabled!")
    model = LlamaForCausalLM.from_pretrained(model_name, device_map=device_map, torch_dtype=dtype).eval()
    questions = load_questions(tokenizer, question_count)
    key_quantizers = build_quantizers(dtype, device, key_quantizer_configs)
    value_quantizers = build_quantizers(dtype, device, value_quantizer_configs)
    assert len(key_quantizers) == len(value_quantizers)
    print(f"Total # of evaluations: {len(key_quantizers)}")
    for idx, (key_quantizer, value_quantizer) in enumerate(zip(key_quantizers, value_quantizers)):
        print("======================================")
        print(f"Running evaluation #{idx+1}...")
        evaluator = Evaluator(device, version, model, questions, key_quantizer, value_quantizer, False)
        result = evaluator.cached_evaluate(cache_file, use_tqdm=True)
        print(f"  Params: {evaluator.params}")
        print(f"  Results: {asdict(result)}")


if __name__ == "__main__":
    main()
