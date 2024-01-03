import os
import torch
import experiments as exp

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "meta-llama/Llama-2-7b-hf"
dtype = torch.float16
question_count = 1000


if __name__ == "__main__":
    # exp.KeyValueDifference(model_name, dtype, question_count, parallel=True, verbose=True).run()
    # exp.KVcacheDistribution(model_name, dtype, question_count, parallel=True, verbose=True).run()
    exp.GridSearch(model_name, dtype, question_count, parallel=True, verbose=True).run()
