import os
import torch
import experiments as exp

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

model_name = "meta-llama/Llama-2-13b-hf"
dataset_name = "Rowan/hellaswag"
dtype = torch.float16
question_count = 1000


if __name__ == "__main__":
    exp.GridSearch(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()
    # exp.KeyValueDifference(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()
    # exp.KVcacheDistribution(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()
    # exp.AttentionInsight(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()
    # exp.Test(model_name, dataset_name, dtype, question_count, parallel=True, verbose=True).run()
