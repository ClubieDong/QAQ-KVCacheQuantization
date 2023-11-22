import torch
from question import load_questions
from quantizer import build_quantizers
from evaluator import Evaluator
from transformers import LlamaForCausalLM, LlamaTokenizerFast


model_name = "meta-llama/Llama-2-7b-hf"
device = torch.device("cuda:0")
dtype = torch.float16
repeat_count = 100
cache_file = "cache_03.json"

key_quantizer_configs = [{
    "key_or_value_cache": ["key"],
    "use_attentions": [False],
    "method": ["uniform", "normal"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "n_bits_uniform": [2, 3, 4, 6, 8],
}, {
    "key_or_value_cache": ["key"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "last_n_attentions": [1, 5, 10, 20],
    "target_quantization_error": [0.013],
    "n_bits_min": [1, 2, 3, 4],
    "n_bits_max": [8],
}, {
    "key_or_value_cache": ["key"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "last_n_attentions": [5],
    "target_quantization_error": [0.011, 0.013, 0.015],
    "n_bits_min": [1, 2, 3, 4],
    "n_bits_max": [8],
}]
value_quantizer_configs = [{
    "key_or_value_cache": ["value"],
    "use_attentions": [False],
    "method": ["uniform", "normal"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "n_bits_uniform": [2, 3, 4, 6, 8],
}, {
    "key_or_value_cache": ["value"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "last_n_attentions": [1, 5, 10, 20],
    "target_quantization_error": [0.004],
    "n_bits_min": [1, 2, 3, 4],
    "n_bits_max": [8],
}, {
    "key_or_value_cache": ["value"],
    "use_attentions": [True],
    "method": ["uniform"],
    "level": ["token", "layer", "head"],
    "symmetric": [False],
    "last_n_attentions": [5],
    "target_quantization_error": [0.002, 0.004, 0.006],
    "n_bits_min": [1, 2, 3, 4],
    "n_bits_max": [8],
}]


def main():
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=dtype).eval()
    questions = load_questions(tokenizer)
    key_quantizers = build_quantizers(dtype, device, key_quantizer_configs)
    value_quantizers = build_quantizers(dtype, device, value_quantizer_configs)
    assert len(key_quantizers) == len(value_quantizers)
    print(f"Total # of evaluations: {len(key_quantizers)}")
    for key_quantizer, value_quantizer in zip(key_quantizers, value_quantizers):
        evaluator = Evaluator(device, model, questions, key_quantizer, value_quantizer)
        result = evaluator.cached_evaluate(repeat_count, cache_file, use_tqdm=True)
        print()
        print(f"{evaluator.quantizer_name}:")
        print(f"    Accuracy: {result.accuracy*100:.6f}%")
        print(f"    95% confidence interval of accuracy: {result.accuracy_confidence:.6f}")
        print(f"    Correct answer log probability: {result.answer_log_probability:.6f}")
        print(f"    Quantization error of key cache: {result.key_quantization_error:.6f}")
        print(f"    Quantization error of value cache: {result.value_quantization_error:.6f}")
        print(f"    Average size of key cache per token: {result.key_average_size:.6f} bit")
        print(f"    Average size of value cache per token: {result.value_average_size:.6f} bit")
        print(f"    Attention error: {result.attention_error:.6f}")
        print(f"    Logit error: {result.logit_error:.6f}")


if __name__ == "__main__":
    main()
