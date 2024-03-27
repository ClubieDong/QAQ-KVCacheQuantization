import re
import torch
import random
from models import Tokenizer
from config import hf_cache_dir
from datasets import load_dataset
from dataclasses import dataclass
from functools import cached_property


@dataclass
class Question:
    input_ids: torch.Tensor
    question_length: int
    choice_length: list[int]
    question: str
    choices: list[str]
    answer_idx: int


class QADataset:
    def __init__(self, dataset_name: str, tokenizer: Tokenizer, question_count: int):
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.question_count = question_count

    @cached_property
    def questions(self) -> list[Question]:
        if self.dataset_name in ["Rowan/hellaswag", "math_qa", "piqa"]:
            raw_dataset: list[dict[str, str]] = list(load_dataset(self.dataset_name, split="validation", cache_dir=hf_cache_dir))
        elif self.dataset_name == "truthful_qa":
            raw_dataset: list[dict[str, str]] = list(load_dataset("truthful_qa", "multiple_choice", split="validation", cache_dir=hf_cache_dir))
        random.seed(42)
        raw_dataset = random.sample(raw_dataset, k=self.question_count)
        if self.dataset_name == "Rowan/hellaswag":
            return self._load_hellaswag(raw_dataset)
        if self.dataset_name == "math_qa":
            return self._load_mathqa(raw_dataset)
        if self.dataset_name == "piqa":
            return self._load_piqa(raw_dataset)
        if self.dataset_name == "truthful_qa":
            return self._load_truthfulqa(raw_dataset)

    def _build_question(self, question: str, choices: list[str], answer_idx: int) -> Question:
        question_len = self.tokenizer(question, return_tensors="pt", add_special_tokens=False, return_attention_mask=False).input_ids.shape[1]
        question_choices = [question + " " + choice for choice in choices]
        results = self.tokenizer(question_choices, return_tensors="pt", padding=True, add_special_tokens=False, return_attention_mask=True)
        choices_len = (results.attention_mask.sum(dim=1) - question_len).tolist()
        return Question(
            input_ids=results.input_ids,
            question_length=question_len,
            choice_length=choices_len,
            question=question,
            choices=choices,
            answer_idx=answer_idx,
        )

    def _load_hellaswag(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        pattern = re.compile(r"\[.*?\]")
        def preprocess_text(text: str) -> str:
            text = text.strip()
            # Brackets are artifacts of the WikiHow dataset portion of HellaSwag
            text = text.replace(" [title]", ". ")
            text = re.sub(pattern, "", text)
            text = text.replace("  ", " ")
            return text
        questions: list[Question] = []
        for data in raw_dataset:
            question = preprocess_text(data["activity_label"] + ": " + data["ctx_a"] + " " + data["ctx_b"].capitalize())
            choices = [preprocess_text(choice) for choice in data["endings"]]
            answer_idx = int(data["label"])
            questions.append(self._build_question(question, choices, answer_idx))
        return questions

    def _load_mathqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        pattern = re.compile(r"[abcd] \) .*?, |e \) .*?$")
        questions: list[Question] = []
        for data in raw_dataset:
            question = f"Question: {data['Problem']}\nAnswer:"
            choices = [c[4:].rstrip(" ,") for c in re.findall(pattern, data["options"])]
            answer_idx = ["a", "b", "c", "d", "e"].index(data["correct"])
            questions.append(self._build_question(question, choices, answer_idx))
        return questions

    def _load_piqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        questions: list[Question] = []
        for data in raw_dataset:
            question = f"Question: {data['goal']}\nAnswer:"
            choices = [data["sol1"], data["sol2"]]
            answer_idx = int(data["label"])
            questions.append(self._build_question(question, choices, answer_idx))
        return questions

    def _load_truthfulqa(self, raw_dataset: list[dict[str, str]]) -> list[Question]:
        questions: list[Question] = []
        for data in raw_dataset:
            question = f"Q: {data['question']}\nA:"
            choices = data["mc1_targets"]["choices"]
            answer_idx = 0
            questions.append(self._build_question(question, choices, answer_idx))
        return questions
