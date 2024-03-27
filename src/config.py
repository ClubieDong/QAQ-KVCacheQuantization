import torch

version = "2023/03/20-#02"
cache_file = "cache/results.json"
hf_cache_dir = None

# # 8xV100 & Llama-2-7B
# device_configs = [
#     (torch.device("cuda:0"), {0: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:1"), {1: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:2"), {2: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:3"), {3: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:4"), {4: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:5"), {5: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:6"), {6: "32GB", "cpu": "400GB"}),
#     (torch.device("cuda:7"), {7: "32GB", "cpu": "400GB"}),
# ]

# 8xV100 & Llama-2-13B
device_configs = [
    (torch.device("cuda:0"), {0: "10GB", 1: "30GB", "cpu": "400GB"}),
    (torch.device("cuda:2"), {2: "10GB", 3: "30GB", "cpu": "400GB"}),
    (torch.device("cuda:4"), {4: "10GB", 5: "30GB", "cpu": "400GB"}),
    (torch.device("cuda:6"), {6: "10GB", 7: "30GB", "cpu": "400GB"}),
]

# # 8xV100 & Llama-2-70B
# device_configs = [
#     (torch.device("cuda:0"), {
#         0: "32GB",
#         1: "32GB",
#         2: "32GB",
#         3: "32GB",
#         4: "32GB",
#         "cpu": "400GB",
#     }),
# ]
