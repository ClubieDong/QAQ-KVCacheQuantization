import abc
import math
import torch
from typing import Optional
from dataclasses import asdict
from quantizer import Quantizer
from functools import cached_property, cache
from question import Question, load_questions
from evaluator import Evaluator, EvaluationResult
from multiprocessing import Pool, current_process
from accelerate import init_empty_weights, infer_auto_device_map
from config import version, cache_file, hf_cache_dir, device_configs
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizerFast


class Experiment(abc.ABC):
    def __init__(self, model_name: str, dtype: torch.dtype, question_count: int, parallel: bool, verbose: bool):
        self.model_name = model_name
        self.dtype = dtype
        self.question_count = question_count
        self.parallel = parallel
        self.verbose = verbose

    @cached_property
    def tokenizer(self) -> LlamaTokenizerFast:
        tokenizer = LlamaTokenizerFast.from_pretrained(self.model_name, cache_dir=hf_cache_dir)
        tokenizer.pad_token_id = 0
        return tokenizer

    @cache
    def get_model(self, worker_id: int) -> LlamaForCausalLM:
        with init_empty_weights():
            model = LlamaForCausalLM(LlamaConfig.from_pretrained(self.model_name, cache_dir=hf_cache_dir))
        _, max_memory = device_configs[worker_id]
        device_map = infer_auto_device_map(model, max_memory=max_memory, dtype=self.dtype, no_split_module_classes=LlamaForCausalLM._no_split_modules)
        if any(x == "cpu" or x == "disk" for x in device_map.values()):
            print("Warning: CPU offloading enabled!")
        model = LlamaForCausalLM.from_pretrained(self.model_name, device_map=device_map, torch_dtype=self.dtype, cache_dir=hf_cache_dir).eval()
        return model

    @cached_property
    def questions(self) -> list[Question]:
        return load_questions(self.tokenizer, self.question_count)

    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        return []

    @abc.abstractmethod
    def process_result(self, results: list[EvaluationResult]):
        pass
    
    def _is_all_cached(self) -> Optional[list[EvaluationResult]]:
        results: list[EvaluationResult] = []
        for key_quantizer, value_quantizer in self.quantizer_list:
            evaluator = Evaluator("cpu", version, self.model_name, self.questions, key_quantizer, value_quantizer)
            result = evaluator.is_result_cached(cache_file)
            if result is None:
                return None
            results.append(result)
        return results

    def _run_single_evaluation(self, idx, quantizers: tuple[Quantizer, Quantizer]) -> EvaluationResult:
        key_quantizer, value_quantizer = quantizers
        if self.parallel:
            worker_id = current_process()._identity[0] - 1
        else:
            worker_id = 0
        print(f"Running evaluation #{idx+1} on worker #{worker_id+1}...")
        device, _ = device_configs[worker_id]
        model = self.get_model(worker_id)
        key_quantizer.set_dtype_and_device(self.dtype, device)
        value_quantizer.set_dtype_and_device(self.dtype, device)
        evaluator = Evaluator(device, version, self.model_name, self.questions, key_quantizer, value_quantizer)
        result = evaluator.cached_evaluate(model, cache_file, use_tqdm=True)
        if self.verbose:
            print(f"  Params: {evaluator.params}")
            print(f"  Results: {asdict(result)}")
            print("======================================")
        return result

    def run(self):
        results = self._is_all_cached()
        if results is None:
            results = []
            if len(self.quantizer_list) == 0:
                print("Warning: No quantizers are specified!")
            elif self.parallel:
                _, _ = self.questions, self.tokenizer
                chunk_size = int(math.ceil(len(self.quantizer_list) / len(device_configs)))
                with Pool(len(device_configs)) as pool:
                    results = pool.starmap(self._run_single_evaluation, enumerate(self.quantizer_list), chunksize=chunk_size)
            else:
                for idx, quantizers in enumerate(self.quantizer_list):
                    results.append(self._run_single_evaluation(idx, quantizers))
        self.process_result(results)
