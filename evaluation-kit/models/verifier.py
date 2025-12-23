from typing import Callable, List

from models.prm import load_prm
from models.generative_verifier import GenerativeVerifier
from search.config import SearchConfig


def get_prm_verifier(args: SearchConfig):
    is_thinking_model = "r1" in args.model_path.lower()
    prm = load_prm(args.prm_path, prm_batch_size=args.prm_batch_size)

    def scorer(question, outputs):
        questions = [question]
        if is_thinking_model:
            processed_outputs = []
            for i, output in enumerate(outputs):
                if "</think>" in output:
                    processed_outputs.append(output.split("</think>")[-1])
                else:
                    processed_outputs.append(output[-10000:])
        else:
            processed_outputs = outputs
        scores = prm.score(questions, [processed_outputs])[0]
        return scores

    return scorer


def load_verifier(args: SearchConfig) -> Callable[[str, List[str]], List[List[float]]]:
    if args.verifier_type == "model":
        return get_prm_verifier(args)
    elif args.verifier_type == "generative_verifier":
        return GenerativeVerifier(args)
    elif args.verifier_type == "none":
        return None
    else:
        raise ValueError(f"Invalid verifier type: {args.verifier_type}")
