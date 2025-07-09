import re
import json
from typing import List

import requests
# from rapidfuzz import fuzz
from openai import OpenAI


def fuzzy_match(response: str, target: str):
    """
    Find the first occurrence of target in response, allow some format differences.
    Return the index of the first occurrence and the score.
    """
    best_score = 0
    best_index = len(response)

    if len(target) >= len(response):
        return best_index, best_score

    for i in range(len(response) - len(target) + 1):
        snippet = response[i: i + len(target)]
        # score = fuzz.partial_ratio(snippet.lower(), target.lower())
        
        # if score > best_score:
        #     best_score = score
        #     best_index = i

    return best_index, best_score


def send_request(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    url: str = '',
    retry_count: int = 10,
    key: str = '',
    ) -> List[str]:

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    data = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": temperature,
        "n": n,
        "response_format": { "type": "json_object" }
    }
    if system_prompt:
        data["messages"].insert(0, {"role": "system", "content": system_prompt})

    for _ in range(retry_count):
        try:
            response = requests.post(url, headers=headers, json=data)
            if response.status_code != 200:
                raise Exception(f"{response.text}")
            break
        except Exception as exc:
            print("Exception: ", exc)

    choices = response.json().get("choices", [])
    if not choices:
        return []
    
    return [json.loads(choice["message"]["content"]) for choice in choices]   


def send_request_openai_sdk(
    prompt: str,
    system_prompt: str,
    n: int = 1,
    temperature: float = 1.0,
    model: str = '',
    key: str = '',
    url: str = '',
    retry_count: int = 10,
) -> List[str]:
    client = OpenAI(
        api_key=key,
        base_url=url,
    )
    for _ in range(retry_count):
        try:
            if system_prompt:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    n=n,
                    response_format={ "type": "json_object" }
                )
            else:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    n=n,
                    response_format={ "type": "json_object" }
                )
            break
        except Exception as exc:
            print(f"Error: {exc}")

    return [json.loads(choice.message.content) for choice in completion.choices]


def voting(response_list):
    """
    Extract scores from responses and do majority voting for n samples
    """
    if not response_list:
        return 0.0
    scores = []
    for response in response_list:
        match = re.search("Organization Score:\s*(\d{1,2})", response)
        if match:
            score = int(match.group(1))
            scores.append(score)
        else:
            continue
    
    if scores:
        return sum(scores) // len(scores)
    else:
        return 0.0


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