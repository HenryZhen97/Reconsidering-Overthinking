from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import tqdm
import torch
import numpy as np

from verl import DataProto
from rllm.rewards.compress_reward import RewardSample
from rllm.rewards import RewardConfig


def parallel_exec(func, samples, train_config, reward_config, tokenizer, timeout=30, max_workers=32):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for idx, sample in enumerate(samples):
            futures.append((idx, executor.submit(func, sample, train_config, reward_config, tokenizer)))

        results = {}
        for idx, future in futures:
            try:
                results[idx] = future.result(timeout=timeout)
            except Exception as e:
                print(f"Error evaluating sample: {e}")
                raise

    return [results[i] for i in range(len(samples))]


class VeRLRewardManager():
    def __init__(self, tokenizer, reward_func, train_config):
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.train_config = train_config
        self.reward_config = RewardConfig()

    def __call__(self, data: DataProto):
        """VeRL trainer中reward_fn接口，VeRL DataProto类型"""
        prompt_ids = data.batch["prompts"]
        response_ids = data.batch["responses"]
        problem = data.non_tensor_batch["problem"]
        data_sources = data.non_tensor_batch["data_source"]
        prompt_length = prompt_ids.shape[-1]
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)

        response_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [(data_item.non_tensor_batch["reward_model"]["ground_truth"]) for data_item in data]
        ground_truth = [x.tolist() if isinstance(x, np.ndarray) else x for x in ground_truth]

        assert len(response_str) == len(ground_truth) == len(data_sources)

        scores = []
        metrics = {
            "external_redundancy": [],
            "internal_redundancy": [],
        }
        # 控制一次性喂给Pool的数量，减少数据量的内存占用，避免OOM
        for i in range(0, len(response_str), 1024):
            problem_batch = problem[i:i+1024]
            response_str_batch = response_str[i:i+1024]
            data_source_batch = data_sources[i:i+1024]
            ground_truth_batch = ground_truth[i:i+1024]

            samples = [RewardSample(
                problem=problem,
                response_str=response_str,
                ground_truth=ground_truth,
                data_source=data_source,
            ) for problem, response_str, ground_truth, data_source in zip(
                problem_batch,
                response_str_batch, 
                ground_truth_batch,
                data_source_batch
                )]
            result_batch = parallel_exec(self.reward_func, samples, self.train_config, self.reward_config, self.tokenizer)
            scores += [result.reward for result in result_batch]
            metrics["external_redundancy"] += [result.external_redundancy for result in result_batch]
            metrics["internal_redundancy"] += [result.internal_redundancy for result in result_batch]
            
        assert len(scores) == len(response_str)

        # 构建reward tensor
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        for i in range(len(data)):
            reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
        
        return reward_tensor, metrics
