"""Script to prepare DeepScaler training and test datasets.

This script processes math problem datasets into a standardized format for training
and testing DeepScaler models. It loads problems from specified datasets, adds
instruction prompts, and saves the processed data as parquet files.
"""

import argparse
import os
from typing import Dict, List, Optional, Any

import pandas as pd
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed

from rllm.data.utils import load_dataset
from rllm.data.dataset_types import TrainDataset, TestDataset


def extract_solution(solution_str: str) -> str:
    """Extract the final boxed solution from a solution string.

    Args:
        solution_str: Raw solution string that may contain multiple boxed answers

    Returns:
        The final boxed answer with box notation removed
    """
    return remove_boxed(last_boxed_only_string(solution_str))


def remove_leading_zeros(problem, ans):

    if not ans.isdigit():
        return ans
        
    # 处理前导零
    if ans.startswith('0') and len(ans) > 1:
        print("Problem:", problem)
        print(f"  原始值: {ans}")
        ans = str(int(ans))  # 去掉前导零
        print(f"  处理后: {ans}")
        print("-" * 30)

    return ans

def startswith_zero(s):
    if not s.isdigit():
        return False
    
    return s.startswith('0')


def is_decimal(s):
    try:
        float(s)
        return '.' in s
    except ValueError:
        return False


def is_negative_number(s):
    try:
        return float(s) < 0
    except ValueError:
        return False


def valid_number(s):
    assert isinstance(s, str), "Input must be string."

    if not (s.isdigit() or is_decimal(s) or is_negative_number(s)):
        return False

    if s.isdigit() and startswith_zero(s):
            return False

    return len(s) > 2


def make_map_fn(split: str):
    """Create a mapping function to process dataset examples.

    Args:
        split: Dataset split name ('train' or 'test')

    Returns:
        Function that processes individual dataset examples
    """
    def process_fn(example: Dict[str, Any], idx: int, instruction: str = None, data_source: str = "") -> Optional[Dict[str, Any]]:
        problem = example.pop('problem')
        
        if instruction is None:
            instruction = "Let's think step by step and output the final answer within \\boxed{}."
        
        if data_source == "gsm8k":
            question = f"{problem} {instruction}"
            answer = example.pop('solution').split('\n####')[-1].strip()
        else:
            question = f"{problem} {instruction}"
            answer = example.pop('answer')

        if split == 'train':
            if not valid_number(str(answer)):
                return None
            
            if answer in problem:
                return None

        data = {
            "problem": problem,
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data
    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process datasets for DeepScaler training')
    parser.add_argument('--local_dir', default=os.path.expanduser('~/rllm/data'),
                       help='Local directory to save processed datasets')
    parser.add_argument('--hdfs_dir', default=None,
                       help='Optional HDFS directory to copy datasets to')
    args = parser.parse_args()

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    
    # Make local directory if it doesn't exist
    makedirs(local_dir, exist_ok=True)

    # Initialize datasets
    train_datasets = [TrainDataset.Math.DEEPSCALER, 
                        TrainDataset.Math.AIME, 
                        TrainDataset.Math.MATH, 
                        TrainDataset.Math.AMC, 
                        TrainDataset.Math.STILL,
                        TrainDataset.Math.OMNI_MATH,
                        TrainDataset.Math.NUMINA_OLYMPIAD]
    train_dataset = load_dataset(train_datasets[0])
    test_datasets = [TestDataset.Math.AIME, TestDataset.Math.GSM8k, TestDataset.Math.MATH, TestDataset.Math.AMC, TestDataset.Math.OLYMPIAD_BENCH, TestDataset.Math.MINERVA]
    data_source_dict = {
        TestDataset.Math.AIME: "aime24",
        TestDataset.Math.AMC: "amc",
        TestDataset.Math.GSM8k: "gsm8k",
        TestDataset.Math.MATH: "math500",
        TestDataset.Math.AMC: "amc",
        TestDataset.Math.OLYMPIAD_BENCH: "olympiad_bench",
        TestDataset.Math.MINERVA: "minerva",
    }
    test_datasets_data = [load_dataset(d) for d in test_datasets]

    # Process training data
    train_data: List[Dict[str, Any]] = []
    process_fn = make_map_fn('train')
    print("-" * 30)
    print("Before process: ", len(train_dataset))
    for idx, example in enumerate(train_dataset):
        processed_example = process_fn(example, idx)
        if processed_example is not None:
            train_data.append(processed_example)
    print("After process: ", len(train_data))
    print("-" * 30)

    # Process and save each test dataset separately
    for test_dataset, test_data_list in zip(test_datasets, test_datasets_data):
        test_data: List[Dict[str, Any]] = []
        process_fn = make_map_fn('test')
        for idx, example in enumerate(test_data_list):
            processed_example = process_fn(example, idx, data_source=data_source_dict[test_dataset])
            if processed_example is not None:
                test_data.append(processed_example)

        dataset_name = test_dataset.value.lower()
        test_df = pd.DataFrame(test_data)
        test_df.to_parquet(os.path.join(local_dir, f'{dataset_name}.parquet'))
        print(f"{dataset_name} test data size:", len(test_data))

    # Save training dataset
    print("train data size:", len(train_data))
    train_df = pd.DataFrame(train_data)
    train_df.to_parquet(os.path.join(local_dir, 'deepscaler_train.parquet'))

    # Optionally copy to HDFS
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        