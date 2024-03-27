import os
import gc
import json
import math
import torch
from tqdm import tqdm
from models import CausalLM
from typing import Optional, Any
from torch.nn import functional as F
from functools import cached_property
from dataclasses import dataclass, asdict
from qa_dataset import QADataset, Question
from quantizer import Quantizer, AttentionType


@dataclass
class EvaluationResult:
    accuracy: float = 0.0
    accuracy_confidence: float = 0.0
    answer_log_probability: float = 0.0
    quantization_error: float = 0.0
    key_quantization_error: float = 0.0
    value_quantization_error: float = 0.0
    attention_error: float = 0.0
    logit_error: float = 0.0
    average_n_bits: float = 0.0
    key_average_n_bits: float = 0.0
    value_average_n_bits: float = 0.0
    average_size: float = 0.0
    key_average_size: float = 0.0
    value_average_size: float = 0.0


class Evaluator:
    def __init__(self, device: torch.device,
                 version: str,
                 model_name: str,
                 datasets: QADataset,
                 key_quantizer: Quantizer,
                 value_quantizer: Quantizer):
        self.device = device
        self.version = version
        self.model_name = model_name
        self.datasets = datasets
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer

    @cached_property
    def params(self) -> dict[str, Any]:
        res: dict[str, Any] = {}
        res["version"] = self.version
        res["model_name"] = self.model_name
        res["dataset_name"] = self.datasets.dataset_name
        res["question_count"] = self.datasets.question_count
        res["key_quantizer"] = self.key_quantizer.params
        res["value_quantizer"] = self.value_quantizer.params
        return res

    def _calc_tensor_error(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        return ((tensor1.to(self.device) - tensor2.to(self.device)) ** 2).mean().item()

    def _calc_attention_error(self, attention1: AttentionType, attention2: AttentionType) -> float:
        return sum(self._calc_tensor_error(attn1, attn2) for attn1, attn2 in zip(attention1, attention2)) / len(attention1)

    def _evaluate_single(self, model: CausalLM, question: Question) -> EvaluationResult:
        question_len = question.question_length
        # Forward before quantization
        input_ids = question.input_ids.to(self.device)
        result = model.forward(input_ids, use_cache=True, output_attentions=True, return_dict=True)
        # Quantize key/value cache
        question_attentions = [attn[:,:,:question_len,:question_len].to(self.device) for attn in result.attentions]
        key_cache = torch.stack([key[:,:,:question_len,:].to(self.device) for key, _ in result.past_key_values])
        value_cache = torch.stack([value[:,:,:question_len,:].to(self.device) for _, value in result.past_key_values])
        quantized_key_cache, key_average_n_bits = self.key_quantizer.quantize(key_cache, question_attentions)
        quantized_value_cache, value_average_n_bits = self.value_quantizer.quantize(value_cache, question_attentions)
        quantized_kvcache = [
            (key.to(result.past_key_values[idx][0].device), value.to(result.past_key_values[idx][0].device))
            for idx, (key, value) in enumerate(zip(quantized_key_cache, quantized_value_cache))
        ]
        # Forward after quantization
        quantized_result = model.forward(input_ids[:,question_len:], past_key_values=quantized_kvcache, use_cache=True, output_attentions=True, return_dict=True)
        # Calculate log probabilities
        first_word_log_softmax = F.log_softmax(result.logits[:,question_len-1], dim=-1)
        quantized_log_softmax = F.log_softmax(quantized_result.logits, dim=-1)
        max_log_probability, max_choice_idx, answer_log_probability = None, None, None
        for choice_idx, choice_len in enumerate(question.choice_length):
            quantized_log_probability = first_word_log_softmax[choice_idx, input_ids[choice_idx, question_len]].item()
            quantized_log_probability += quantized_log_softmax[choice_idx, torch.arange(choice_len-1), input_ids[choice_idx,question_len+1:question_len+choice_len]].sum().item()
            quantized_log_probability /= choice_len
            if choice_idx == question.answer_idx:
                answer_log_probability = quantized_log_probability
            if max_log_probability is None or quantized_log_probability > max_log_probability:
                max_log_probability = quantized_log_probability
                max_choice_idx = choice_idx
        # Calculate quantization metrics
        key_quantization_error = self._calc_tensor_error(key_cache, quantized_key_cache)
        value_quantization_error = self._calc_tensor_error(value_cache, quantized_value_cache)
        attention_error = self._calc_attention_error(
            [attn[:,:,question_len:,:question_len].to(self.device) for attn in result.attentions],
            [attn[:,:,:,:question_len].to(self.device) for attn in quantized_result.attentions],
        )
        logit_error = self._calc_tensor_error(result.logits[:,question_len:,:], quantized_result.logits)
        key_average_size = self.key_quantizer.calc_quantized_cache_size_per_token(key_average_n_bits, model)
        value_average_size = self.value_quantizer.calc_quantized_cache_size_per_token(value_average_n_bits, model)
        return EvaluationResult(
            accuracy=1.0 if max_choice_idx == question.answer_idx else 0.0,
            answer_log_probability=answer_log_probability,
            quantization_error=(key_quantization_error + value_quantization_error) / 2,
            key_quantization_error=key_quantization_error,
            value_quantization_error=value_quantization_error,
            attention_error=attention_error,
            logit_error=logit_error,
            average_size=(key_average_size + value_average_size) / 2,
            key_average_size=key_average_size,
            value_average_size=value_average_size,
            average_n_bits=(key_average_n_bits + value_average_n_bits) / 2,
            key_average_n_bits=key_average_n_bits,
            value_average_n_bits=value_average_n_bits,
        )

    def evaluate(self, model: CausalLM, use_tqdm: bool) -> EvaluationResult:
        assert model.name_or_path == self.model_name
        result = EvaluationResult()
        total_tokens = 0
        with torch.no_grad():
            for idx, question in enumerate(tqdm(self.datasets.questions) if use_tqdm else self.datasets.questions):
                single_result = self._evaluate_single(model, question)
                n_tokens = question.question_length
                total_tokens += n_tokens
                result.accuracy += single_result.accuracy
                result.answer_log_probability += single_result.answer_log_probability
                result.quantization_error += single_result.quantization_error
                result.key_quantization_error += single_result.key_quantization_error
                result.value_quantization_error += single_result.value_quantization_error
                result.attention_error += single_result.attention_error
                result.logit_error += single_result.logit_error
                result.average_size += single_result.average_size * n_tokens
                result.key_average_size += single_result.key_average_size * n_tokens
                result.value_average_size += single_result.value_average_size * n_tokens
                result.average_n_bits += single_result.average_n_bits * n_tokens
                result.key_average_n_bits += single_result.key_average_n_bits * n_tokens
                result.value_average_n_bits += single_result.value_average_n_bits * n_tokens
                if (idx + 1) % 100 == 0:
                    gc.collect()
        result.accuracy /= self.datasets.question_count
        # Calculate 95% confidence interval
        result.accuracy_confidence = 1.96 * math.sqrt(result.accuracy * (1.0 - result.accuracy) / self.datasets.question_count)
        result.answer_log_probability /= self.datasets.question_count
        result.quantization_error /= self.datasets.question_count
        result.key_quantization_error /= self.datasets.question_count
        result.value_quantization_error /= self.datasets.question_count
        result.attention_error /= self.datasets.question_count
        result.logit_error /= self.datasets.question_count
        result.average_size /= total_tokens
        result.key_average_size /= total_tokens
        result.value_average_size /= total_tokens
        result.average_n_bits /= total_tokens
        result.key_average_n_bits /= total_tokens
        result.value_average_n_bits /= total_tokens
        return result
    
    def get_cached_result(self, cache_file: Optional[str]) -> Optional[EvaluationResult]:
        if cache_file is None or not os.path.exists(cache_file):
            return None
        with open(cache_file, "r") as f:
            cached_results = json.load(f)
            for entry in cached_results:
                if entry["params"] == self.params:
                    return EvaluationResult(**entry["results"])
        return None

    def cache_result(self, cache_file: Optional[str], result: EvaluationResult):
        if cache_file is None:
            return
        if os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
        else:
            cached_results = []
        cached_results.append({
            "params": self.params,
            "results": asdict(result),
        })
        with open(cache_file, "w") as f:
            json.dump(cached_results, f, indent=4, separators=(", ", ": "))
