from typing import Union
from transformers import LlamaForCausalLM, OPTForCausalLM, LlamaTokenizerFast, GPT2TokenizerFast

CausalLM = Union[LlamaForCausalLM, OPTForCausalLM]
Tokenizer = Union[LlamaTokenizerFast, GPT2TokenizerFast]
