import os
import json
import math
import torch
from tqdm import tqdm
from typing import Optional
from question import Question
from torch.nn import functional as F
from functools import cached_property
from transformers import LlamaForCausalLM
from dataclasses import dataclass, asdict
from quantizer import Quantizer, AttentionType


@dataclass
class EvaluationResult:
    accuracy: float = 0.0
    accuracy_confidence: float = 0.0
    answer_log_probability: float = 0.0
    key_quantization_error: float = 0.0
    value_quantization_error: float = 0.0
    attention_error: float = 0.0
    logit_error: float = 0.0
    key_average_n_bits: float = 0.0
    value_average_n_bits: float = 0.0
    key_average_size: float = 0.0
    value_average_size: float = 0.0


class Evaluator:
    def __init__(self, device: torch.device, 
                 model: LlamaForCausalLM,
                 questions: list[Question],
                 key_quantizer: Quantizer,
                 value_quantizer: Quantizer):
        self.device = device
        self.model = model
        self.questions = questions
        self.key_quantizer = key_quantizer
        self.value_quantizer = value_quantizer
    
    @cached_property
    def quantizer_name(self):
        return f"{self.key_quantizer.name} | {self.value_quantizer.name}"
    
    def _calc_tensor_error(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
        return (tensor1.to(self.device) - tensor2.to(self.device)).abs().mean().item()

    def _calc_attention_error(self, attention1: AttentionType, attention2: AttentionType) -> float:
        return sum(self._calc_tensor_error(attn1, attn2) for attn1, attn2 in zip(attention1, attention2))

    def _evaluate_single(self, question: Question) -> EvaluationResult:
        # Forward of question
        result = self.model.forward(question.question, use_cache=True, output_attentions=True, return_dict=True)
        kvcache, attentions, logits = result.past_key_values, result.attentions, result.logits
        # Quantize key/value cache
        key_cache = torch.stack([key.to(self.device) for key, _ in kvcache])
        value_cache = torch.stack([value.to(self.device) for _, value in kvcache])
        quantized_key_cache, key_average_n_bits = self.key_quantizer.quantize(key_cache, attentions)
        quantized_value_cache, value_average_n_bits = self.value_quantizer.quantize(value_cache, attentions)
        quantized_kvcache = [
            (key.expand(len(question.choices), -1, -1, -1), value.expand(len(question.choices), -1, -1, -1))
            for key, value in zip(quantized_key_cache, quantized_value_cache)
        ]
        kvcache = [(key.expand(len(question.choices), -1, -1, -1), value.expand(len(question.choices), -1, -1, -1)) for key, value in kvcache]
        # Forward of choices
        first_word_log_softmax = F.log_softmax(logits[0, -1], dim=-1)
        result = self.model.forward(question.choices, past_key_values=kvcache, use_cache=True, output_attentions=True, return_dict=True)
        quantized_result = self.model.forward(question.choices, past_key_values=quantized_kvcache, use_cache=True, output_attentions=True, return_dict=True)
        quantized_log_softmax = F.log_softmax(quantized_result.logits, dim=-1)
        # Calculate log probabilities
        max_log_probability, max_choice_idx, answer_log_probability = None, None, None
        for choice_idx, choice_length in enumerate(question.choice_length):
            quantized_log_probability = first_word_log_softmax[question.choices[choice_idx, 0]]
            quantized_log_probability += quantized_log_softmax[choice_idx, torch.arange(choice_length - 1), question.choices[choice_idx, 1:choice_length]].sum()
            quantized_log_probability = quantized_log_probability.item() / choice_length
            if choice_idx == question.answer_idx:
                answer_log_probability = quantized_log_probability
            if max_log_probability is None or quantized_log_probability > max_log_probability:
                max_log_probability = quantized_log_probability
                max_choice_idx = choice_idx
        # Calculate quantization metrics
        return EvaluationResult(
            accuracy=1.0 if max_choice_idx == question.answer_idx else 0.0,
            answer_log_probability=answer_log_probability,
            key_quantization_error=self._calc_tensor_error(key_cache, quantized_key_cache),
            value_quantization_error=self._calc_tensor_error(value_cache, quantized_value_cache),
            attention_error=self._calc_attention_error(result.attentions, quantized_result.attentions),
            logit_error=self._calc_tensor_error(result.logits, quantized_result.logits),
            key_average_size=self.key_quantizer.calc_quantized_cache_size_per_token(key_average_n_bits, self.model),
            value_average_size=self.value_quantizer.calc_quantized_cache_size_per_token(value_average_n_bits, self.model),
            key_average_n_bits=key_average_n_bits,
            value_average_n_bits=value_average_n_bits,
        )

    def evaluate(self, count: int, use_tqdm: bool) -> EvaluationResult:
        result = EvaluationResult()
        total_tokens = 0
        with torch.no_grad():
            for question in tqdm(self.questions[:count]) if use_tqdm else self.questions[:count]:
                single_result = self._evaluate_single(question)
                n_tokens = question.question.shape[1]
                total_tokens += n_tokens
                result.accuracy += single_result.accuracy
                result.answer_log_probability += single_result.answer_log_probability
                result.key_quantization_error += single_result.key_quantization_error
                result.value_quantization_error += single_result.value_quantization_error
                result.attention_error += single_result.attention_error
                result.logit_error += single_result.logit_error
                result.key_average_size += single_result.key_average_size * n_tokens
                result.value_average_size += single_result.value_average_size * n_tokens
                result.key_average_n_bits += single_result.key_average_n_bits * n_tokens
                result.value_average_n_bits += single_result.value_average_n_bits * n_tokens
        result.accuracy /= count
        result.accuracy_confidence = 1.96 * math.sqrt(result.accuracy * (1.0 - result.accuracy) / count)
        result.answer_log_probability /= count
        result.key_quantization_error /= count
        result.value_quantization_error /= count
        result.attention_error /= count
        result.logit_error /= count
        result.key_average_size /= total_tokens
        result.value_average_size /= total_tokens
        result.key_average_n_bits /= total_tokens
        result.value_average_n_bits /= total_tokens
        return result

    def cached_evaluate(self, count: int, cache_file: Optional[str], use_tqdm: bool) -> EvaluationResult:
        if cache_file is not None and os.path.exists(cache_file):
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
                if self.quantizer_name in cached_results:
                    return EvaluationResult(**cached_results[self.quantizer_name])
        else:
            cached_results = {}
        result = self.evaluate(count, use_tqdm)
        cached_results[self.quantizer_name] = asdict(result)
        if cache_file is not None:
            with open(cache_file, "w") as f:
                json.dump(cached_results, f, indent=4, separators=(", ", ": "))
        return result
