import torch
import random
from datasets import load_dataset
from dataclasses import dataclass
from transformers import LlamaTokenizerFast


@dataclass
class Question:
    question: torch.Tensor
    choices: torch.Tensor
    choice_length: list[int]
    answer_idx: int


def load_questions(tokenizer: LlamaTokenizerFast):
    raw_dataset: list[dict[str, str]] = load_dataset("Rowan/hellaswag", split="validation")
    print("Dataset loaded!")
    questions: list[Question] = []
    for data in raw_dataset:
        question = tokenizer(f"{data['activity_label']}: {data['ctx']}", return_tensors="pt", add_special_tokens=True, return_attention_mask=False).input_ids
        choices = tokenizer(data["endings"], return_tensors="pt", padding=True, add_special_tokens=False, return_attention_mask=True)
        questions.append(Question(question, choices.input_ids, choices.attention_mask.sum(dim=1).tolist(), int(data["label"])))
    random.seed(42)
    random.shuffle(questions)
    return questions
