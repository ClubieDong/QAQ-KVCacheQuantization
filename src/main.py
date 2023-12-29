import os
import torch
from experiments import KeyValueDifference

debug = False
model_name = "meta-llama/Llama-2-7b-hf"
dtype = torch.float16
question_count = 1000

if debug:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


if __name__ == "__main__":
    KeyValueDifference(model_name, dtype, question_count).run()
