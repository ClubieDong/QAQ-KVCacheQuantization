import math
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from .base import Experiment
from config import device_configs
from matplotlib import pyplot as plt


class AttentionInsight(Experiment):
    def process_result(self, _):
        model = self.get_model(0)
        device = device_configs[0][0]
        question = self.questions[0]
        input_ids = question.input_ids[:1].to(device)
        with torch.no_grad():
            attention = model.forward(input_ids, output_attentions=True, return_dict=True).attentions
        attention = torch.stack(attention).detach().cpu().numpy()
        np.save("data/attention.npy", attention)
        # attention = np.load("data/attention.npy")
        Path("figs/attention").mkdir(parents=True, exist_ok=True)
        for layer_idx, layer_attn in enumerate(tqdm(attention)):
            layer_attn = layer_attn[0]
            # layer_attn.shape: (n_head, seq_len, seq_len)
            n_head = len(layer_attn)
            _, axs = plt.subplots(nrows=math.ceil(n_head/8), ncols=8, figsize=(8*3, math.ceil(n_head/8)*3), tight_layout=True)
            for head_idx, head_attn in enumerate(layer_attn):
                ax = axs[head_idx//8, head_idx%8]
                ax.imshow(head_attn)
                ax.axis("off")
                ax.set_title(f"Head #{head_idx}")
            plt.savefig(f"figs/attention/layer_{layer_idx}.png")
            plt.close()
