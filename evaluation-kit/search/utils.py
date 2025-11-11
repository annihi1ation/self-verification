import numpy as np
import sglang as sgl
from sglang.lang.interpreter import ProgramState


def aggregate_score(scores, method="last"):
    if method == "last":
        return [score[-1] for score in scores]
    elif method == "mean":
        return [np.mean(score) for score in scores]
    elif method == "min":
        return [np.min(score) for score in scores]
    else:
        raise ValueError(f"Unknown aggregate method: {method}")


def add_messages_to_state(s: ProgramState, messages):
    for message in messages:
        if message["role"] == "system":
            s += sgl.system(message["content"])
        elif message["role"] == "user":
            s += sgl.user(message["content"])
        elif message["role"] == "assistant":
            s += sgl.assistant(message["content"])
        else:
            raise ValueError(f"Unknown message role: {message['role']}")
