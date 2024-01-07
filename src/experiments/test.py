from .base import Experiment
from functools import cached_property
from evaluator import EvaluationResult
from quantizer import Quantizer, build_quantizers


class Test(Experiment):
    @cached_property
    def quantizer_list(self) -> list[tuple[Quantizer, Quantizer]]:
        key_quantizers = build_quantizers([{
            "key_or_value_cache": ["key"],
            "level": ["no-quantization"],
        }])
        value_quantizers = build_quantizers([{
            "key_or_value_cache": ["value"],
            "level": ["no-quantization"],
        }])
        return list(zip(key_quantizers, value_quantizers))

    def process_result(self, _: list[EvaluationResult]):
        pass
