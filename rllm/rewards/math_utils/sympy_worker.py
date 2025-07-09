# sympy_worker.py
import sys
import json
import sympy

from rllm.rewards.math_utils.utils import _sympy_parse, should_allow_eval


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    try:
        expr = f"({ground_truth_normalized}) - ({given_normalized})"
        if should_allow_eval(expr):
            diff = _sympy_parse(expr)
            simplified = sympy.simplify(diff)
            return simplified == 0
    except Exception:
        pass
    return False


if __name__ == "__main__":
    try:
        data = json.loads(sys.stdin.read())
        result = are_equal_under_sympy(data["gt"], data["pred"])
        print(json.dumps({"equal": result}))
    except Exception:
        print(json.dumps({"equal": False}))
