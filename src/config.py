import torch

version = "2023/01/04-#01"
cache_file = "cache/results.json"
hf_cache_dir = "/webdav/MyData/shichen/HuggingFaceCache"

# 8xV100 & Llama-2-7B
device_configs = [
    (torch.device("cuda:0"), {0: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:1"), {1: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:2"), {2: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:3"), {3: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:4"), {4: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:5"), {5: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:6"), {6: "32GB", "cpu": "400GB"}),
    (torch.device("cuda:7"), {7: "32GB", "cpu": "400GB"}),
]
