import os
import math
from pydantic import BaseModel, Field
from typing import Optional, Any

from rllm.globals import THOUGHT_DELIMITER_END
from rllm.rewards import  RewardOutput
from rllm.rewards.compress_utils.response_utils import split_cot, get_reasoning_gain, split_sentences
from rllm.rewards.utils.common_utils import valid_number
from rllm.rewards.math_utils.utils import extract_answer, grade_answer_sympy, grade_answer_mathd

SCORE_URL = ""


class RewardSample(BaseModel):
    problem: str = Field(default="", description="Problem")
    data_source: str = Field(default="", description="Data source")
    response_str: str = Field(default="", description="Response")
    ground_truth: str = Field(default="", description="正确答案")
    extra_info: Optional[Any] = Field(default=None, description="Extra information")


def get_rank_mapping(num_apis=8):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))


    mapped_index = rank % num_apis
    mapped_index = min(mapped_index, num_apis - 1)

    return mapped_index

def compress_val(sample: RewardSample, train_config, reward_config, tokenizer) -> RewardOutput:
    problem = sample.problem
    response = sample.response_str
    gt_answer = sample.ground_truth

    # check format
    if THOUGHT_DELIMITER_END in response:
        cot, conclusion = response.rsplit(THOUGHT_DELIMITER_END, 1)
        cot, conclusion = cot.strip(), conclusion.strip()
    else:
        cot = response
        conclusion = response

    acc_score, llm_answer = cal_math_accuracy_reward(reward_config, conclusion, gt_answer)

    # aft redundancy reward
    if acc_score != 1.0:
        redundancy = 0.0
        gain = 0.0
    else:

        if not llm_answer or not valid_number(str(llm_answer)) or llm_answer in problem:
            redundancy = 0.0
        else:
            fcs, aft = split_cot(problem, cot, llm_answer)
            if fcs and aft:
                redundancy = round(len(tokenizer.encode(aft)) / len(tokenizer.encode(cot)), 2)
                cot = fcs
            else:
                redundancy = 0.0

        mapping = get_rank_mapping()
        server_url = f"http://localhost:800{mapping}/embedding"
        cot_slices = split_sentences(cot)
        gain = get_reasoning_gain(cot_slices, server_url)
    
    return RewardOutput(
        reward=acc_score, 
        external_redundancy=redundancy, 
        internal_redundancy=1-gain, 
        is_correct=True
    )

def compress_compute_score(sample: RewardSample, train_config, reward_config, tokenizer) -> RewardOutput:
    problem = sample.problem
    response = sample.response_str
    gt_answer = sample.ground_truth

    # check format, only have one THOUGHT_DELIMITER_END
    if response.count(THOUGHT_DELIMITER_END) == 1 :
        cot, conclusion = response.split(THOUGHT_DELIMITER_END, 1)
        cot, conclusion = cot.strip(), conclusion.strip()
    else:
        return RewardOutput(reward=reward_config.format_error_reward, is_correct=False)
    
    if len(cot) == 0:
        return RewardOutput(reward=reward_config.format_error_reward, is_correct=False)

    acc_score, llm_answer = cal_math_accuracy_reward(reward_config, cot, gt_answer)

    if acc_score != reward_config.correct_reward:
        return RewardOutput(reward=reward_config.incorrect_reward, is_correct=False)
            
    raw_acc_score = 1.0
    redundancy = 0.0
    gain = 0.0

    # aft redundancy reward
    fcs, aft = split_cot(problem, cot, llm_answer)
    if not fcs or not aft:
        return RewardOutput(reward=reward_config.incorrect_reward, is_correct=False)
    redundancy = round(len(tokenizer.encode(aft)) / len(tokenizer.encode(cot)), 2)
    cot = fcs
    
    if train_config.rollout.use_external_redundancy:
        redundancy_score = cal_verify_redundancy_penalty(redundancy, raw_acc_score)
        acc_score = redundancy_score

    # reasoning gain reward
    mapping = get_rank_mapping()
    server_url = f"http://localhost:800{mapping}/embedding"
    cot_slices = split_sentences(cot)
    gain = get_reasoning_gain(cot_slices, server_url)

    if train_config.rollout.use_internal_redundancy:
        gain_score = cal_reasoning_gain_penalty(gain, raw_acc_score)
        acc_score = 0.5 * (gain_score + redundancy_score) if train_config.rollout.use_external_redundancy else gain_score
    
    return RewardOutput(
        reward=acc_score, 
        external_redundancy=redundancy, 
        internal_redundancy=1-gain, 
        is_correct=True
    )


def cal_math_accuracy_reward(config, conclusion, gt_answer):
    score = config.incorrect_reward
    # accuracy reward
    llm_answer = extract_answer(conclusion)
    if llm_answer is None:
        return score, llm_answer
    
    # Process the ground truth(s)
    if gt_answer is None:
        return score, llm_answer
    
    # Process each ground truth
    processed_truth = str(gt_answer)
    if "\\boxed" in processed_truth:
        processed_truth = extract_answer(processed_truth)
        if processed_truth is not None:
            gt_answer = processed_truth.strip()
    
    if str(llm_answer) == str(gt_answer):
        return config.correct_reward, llm_answer

    is_correct = grade_answer_mathd(llm_answer, gt_answer) or grade_answer_sympy(llm_answer, gt_answer)
    if is_correct:
        score = config.correct_reward
    
    return score, llm_answer


def cal_attempt_redundancy_penalty(cot, pre, score):
    assert len(cot) != 0 and len(pre) <= len(cot)
    pre_pct = round(len(pre) / len(cot), 4)
    score *= (1 - pre_pct) ** 2

    return score

    
def penalty_exponential(x: float, base=2.0, min_reward=0.0, max_reward=1.0) -> float:
    assert 0 <= x <= 1
    reward = base ** (-x)
    max_r = base ** 0
    min_r = base ** -1

    return min_reward + (reward - min_r) / (max_r - min_r) * (max_reward - min_reward)


def penalty_exponential_2(x: float, base=0.9, min_reward=0.0, max_reward=1.0, center=0.1) -> float:
    assert 0 <= x <= 1, "x must be in [0, 1]"
    assert 0 < center < 1, "center must be in (0, 1)"
    

    if x < center:

        t = (x / center) ** 2 * center
    else:

        t = center + ((x - center) / (1 - center)) ** base * (1 - center)
    

    reward = base ** (-t)
    

    max_r = base ** 0
    min_r = base ** (-1)
    

    return min_reward + (reward - min_r) / (max_r - min_r) * (max_reward - min_reward)


def penalty_sigmoid(x: float, k=20.0, center=0.3) -> float:
    assert 0 <= x <= 1
    s = 1 / (1 + math.exp(-k * (x - center)))  # sigmoid centered at 0.275
    # normalize output to [0, 1]
    s_min = 1 / (1 + math.exp(-k * (0 - center)))
    s_max = 1 / (1 + math.exp(-k * (1 - center)))
    normalized = (s - s_min) / (s_max - s_min)
    return normalized


def cal_verify_redundancy_penalty(redundancy, score):

    # aft_pct smaller is better
    score *= penalty_exponential_2(redundancy)

    return score


def cal_reasoning_gain_penalty(reasoning_gain, score):

    return penalty_sigmoid(reasoning_gain) * score 
    