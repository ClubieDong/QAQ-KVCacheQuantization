import abc
import torch
from dataclasses import asdict
from quantizer import Quantizer
from functools import cached_property
from question import Question, load_questions
from evaluator import Evaluator, EvaluationResult
from accelerate import init_empty_weights, infer_auto_device_map
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast

device = torch.device("cuda:0")
max_memory = {0: "5GB", 1: "9.7GB", "cpu": "32GB"}
# Change version if you made breaking changes to the code so that the cached results will be invalidated.
version = "2023/12/24-#01"
cache_file = "cache/results.json"


class Experiment(abc.ABC):
    def __init__(self, model_name: str, dtype: torch.device, question_count: int):
        self.model_name = model_name
        self.dtype = dtype
        self.question_count = question_count

    @cached_property
    def tokenizer(self) -> LlamaTokenizerFast:
        tokenizer = LlamaTokenizerFast.from_pretrained(self.model_name)
        tokenizer.pad_token_id = 0
        return tokenizer

    @cached_property
    def model(self) -> LlamaForCausalLM:
        with init_empty_weights():
            model = LlamaForCausalLM(LlamaConfig.from_pretrained(self.model_name))
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=self.dtype, no_split_module_classes=LlamaForCausalLM._no_split_modules)
        if any(x == "cpu" or x == "disk" for x in device_map.values()):
            print("Warning: CPU offloading enabled!")
        model = LlamaForCausalLM.from_pretrained(self.model_name, device_map=device_map, torch_dtype=self.dtype).eval()
        return model

    @cached_property
    def questions(self) -> list[Question]:
        return load_questions(self.tokenizer, self.question_count)

    @abc.abstractproperty
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        pass

    @abc.abstractmethod
    def process_result(self, results: list[EvaluationResult]):
        pass

    def run(self):
        quantizers = self.quantizer_list
        results: list[EvaluationResult] = []
        print(f"Total # of evaluations: {len(quantizers)}")
        for idx, (key_quantizer, value_quantizer) in enumerate(quantizers):
            print("======================================")
            print(f"Running evaluation #{idx+1}...")
            evaluator = Evaluator(device, version, self.model, self.questions, key_quantizer, value_quantizer, False)
            result = evaluator.cached_evaluate(cache_file, use_tqdm=True)
            results.append(result)
            print(f"  Params: {evaluator.params}")
            print(f"  Results: {asdict(result)}")
        self.process_result(results)


# key_quantizer_configs = [{
#     "key_or_value_cache": ["key"],
#     "use_attentions": [False],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0, 0.01],
#     "n_bits_uniform": [2, 3, 4, 6, 8],
# }, {
#     "key_or_value_cache": ["key"],
#     "use_attentions": [True],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0, 0.01],
#     "last_n_attentions": [5],
#     "target_quantization_error": [
#         1000, 10000, 100000, 1000000,
#         1000, 10000, 100000, 1000000,
#         1000, 10000, 100000, 1000000,
#         1000, 10000, 100000, 1000000,
#     ],
#     "n_bits_min": [1, 2],
#     "n_bits_max": [8],
#     "max_q_value": [3],
# }]
# value_quantizer_configs = [{
#     "key_or_value_cache": ["value"],
#     "use_attentions": [False],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0, 0.01],
#     "n_bits_uniform": [2, 3, 4, 6, 8],
# }, {
#     "key_or_value_cache": ["value"],
#     "use_attentions": [True],
#     "method": ["uniform"],
#     "level": ["token", "layer", "head"],
#     "symmetric": [False],
#     "outliers_ratio": [0, 0.01],
#     "last_n_attentions": [5],
#     "target_quantization_error": [
#         1, 1, 1, 1,
#         10, 10, 10, 10,
#         100, 100, 100, 100,
#         1000, 1000, 1000, 1000,
#     ],
#     "n_bits_min": [1, 2],
#     "n_bits_max": [8],
# }]

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
