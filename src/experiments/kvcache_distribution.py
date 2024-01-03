import torch
import numpy as np
from tqdm import tqdm
from .base import Experiment
from config import device_configs
from matplotlib import pyplot as plt


class KVcacheDistribution(Experiment):
    def process_result(self, _):
        n_bins = 1000
        model = self.get_model(0)
        device = device_configs[0][0]
        key_cache_hist, value_cache_hist = torch.zeros(n_bins, dtype=torch.int64), torch.zeros(n_bins, dtype=torch.int64)
        with torch.no_grad():
            for question in tqdm(self.questions):
                length = question.question_length
                input_ids = question.input_ids[:1,:length].to(device)
                kvcache = model.forward(input_ids, use_cache=True, return_dict=True).past_key_values
                key_cache = torch.stack([key.to(device) for key, _ in kvcache]).cpu()
                value_cache = torch.stack([value.to(device) for _, value in kvcache]).cpu()
                # key_cache/value_cache.shape: (n_layer, 1, n_head, seq_len, embed_size_per_head)
                key_cache = key_cache.view(-1).to(dtype=torch.float32)
                value_cache = value_cache.view(-1).to(dtype=torch.float32)
                key_cache_hist += torch.histc(key_cache, bins=n_bins, min=-5.0, max=5.0).to(dtype=torch.int64)
                value_cache_hist += torch.histc(value_cache, bins=n_bins, min=-5.0, max=5.0).to(dtype=torch.int64)
        key_cache_hist = key_cache_hist.detach().numpy()
        value_cache_hist = value_cache_hist.detach().numpy()
        x_range = np.linspace(-5.0, 5.0, 1000)
        plt.plot(x_range, key_cache_hist, label="Key cache distribution")
        plt.plot(x_range, value_cache_hist, label="Value cache distribution")
        plt.legend()
        plt.xlabel("Value")
        plt.ylabel("Count")
        plt.savefig("figs/cache_distribution.png")
        np.save("data/key_cache_hist.npy", key_cache_hist)
        np.save("data/value_cache_hist.npy", value_cache_hist)
