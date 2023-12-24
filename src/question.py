import re
import torch
import random
from datasets import load_dataset
from dataclasses import dataclass
from transformers import LlamaTokenizerFast

pattern = re.compile(r"\[.*?\]")


@dataclass
class Question:
    input_ids: torch.Tensor
    question_length: int
    choice_length: list[int]
    answer_idx: int


def preprocess_text(text: str) -> str:
    text = text.strip()
    # Brackets are artifacts of the WikiHow dataset portion of HellaSwag
    text = text.replace(" [title]", ". ")
    text = re.sub(pattern, "", text)
    text = text.replace("  ", " ")
    return text


def load_questions(tokenizer: LlamaTokenizerFast, question_count: int):
    raw_dataset: list[dict[str, str]] = list(load_dataset("Rowan/hellaswag", split="validation"))
    print("Dataset loaded!")
    random.seed(42)
    raw_dataset = random.sample(raw_dataset, k=question_count)
    questions: list[Question] = []
    for data in raw_dataset:
        question = preprocess_text(data["activity_label"] + ": " + data["ctx_a"] + " " + data["ctx_b"].capitalize())
        question_len = tokenizer(question, return_tensors="pt", add_special_tokens=False, return_attention_mask=False).input_ids.shape[1]
        choices = [preprocess_text(question + " " + choice) for choice in data["endings"]]
        results = tokenizer(choices, return_tensors="pt", padding=True, add_special_tokens=False, return_attention_mask=True)
        choices_len = (results.attention_mask.sum(dim=1) - question_len).tolist()
        questions.append(Question(results.input_ids, question_len, choices_len, int(data["label"])))
    return questions
