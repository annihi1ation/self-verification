from typing import List

import numpy as np
import sglang as sgl
from transformers import AutoTokenizer
from sglang.lang.interpreter import ProgramState

from search.config import SearchConfig

def make_multi_turn_messages(question: str, outputs: List[str]):
    def make_messages(question: str, response: str):
        if "</think>" in response:
            response = response.split("</think>")[-1]
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
            {"role": "user", "content": f'Please verify the solution step by step. At the end of the solution verification, when you give your final grade, write it in the form "Is the answer correct (Yes/No)? X", where X is either Yes or No.'},
        ]
        return messages
    return [make_messages(question, output) for output in outputs]


def get_multi_turn_verification_score(state: ProgramState) -> List[float]:
    logprobs = state.get_meta_info("result")["output_top_logprobs"]
    last_token_logprob = logprobs[-1]
    first_logprob, first_token_id, first_token_text = last_token_logprob[0]
    second_logprob, second_token_id, second_token_text = last_token_logprob[1]
    prob_gap = np.exp(first_logprob) - np.exp(second_logprob)
    score = -1.0
    if "yes" in first_token_text.lower():
        score = prob_gap
    elif "no" in first_token_text.lower():
        score = -prob_gap
    # print(f"last_logprob: {last_logprob}, score: {score}")
    return score


VERIFIER_MESSAGES_BUILDERS = {
    "multi-turn": make_multi_turn_messages,
}

VERIFIER_STOP_TEMPLATES = {
    "multi-turn": ["Is the answer correct (Yes/No)? "],
}

VERIFIER_SCORE_FUNCTIONS = {
    "multi-turn": get_multi_turn_verification_score,
}


@sgl.function
def verifier_rollout(s: ProgramState, prompt, temperature=0.0, max_tokens=2048, stop=[], retries=10):
    s += prompt
    for attempt in range(retries):
        s += sgl.gen("result", temperature=temperature, max_tokens=max_tokens, stop=stop, return_logprob=True, top_logprobs_num=2, return_text_in_logprobs=True)
        if s.error() is None:
            break
        else:
            print(f"Attempt {attempt} failed with error: {s.error()}")


class GenerativeVerifier:
    def __init__(self, args: SearchConfig):
        self.args = args
        self.endpoint = sgl.RuntimeEndpoint(base_url=args.generative_verifier_url, api_key=args.generative_verifier_api_key)
        
        self.model_path = args.generative_verifier_model_path
        self.template = args.generative_verifier_template
        self.temperature = args.generative_verifier_temperature
        self.max_prompt_length = 2040
        self.max_tokens = args.max_tokens
        self.max_threads = args.max_threads

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.make_messages_fn = VERIFIER_MESSAGES_BUILDERS[self.template]
        self.stop_strs = VERIFIER_STOP_TEMPLATES[self.template]
        self.score_fn = VERIFIER_SCORE_FUNCTIONS[self.template]
    
    def score(self, question: str, outputs: List[str]) -> List[float]:
        messages = self.make_messages_fn(question, outputs)
        prompts = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False) for messages in messages]
        prompt_tokens = [len(self.tokenizer.encode(prompt)) for prompt in prompts]
        valid_prompts = [prompt for prompt, token in zip(prompts, prompt_tokens) if token <= self.max_prompt_length]
        valid_prompts_idx = [i for i, token in enumerate(prompt_tokens) if token <= self.max_prompt_length]
        
        run_params = [dict(prompt=prompt, temperature=self.temperature, max_tokens=self.max_tokens, stop=self.stop_strs) for prompt in valid_prompts]
        states = verifier_rollout.run_batch(run_params, num_threads=self.max_threads, backend=self.endpoint)
        scores = [self.score_fn(state) for state in states]
        ret_scores = [0.0] * len(prompts)
        for i, idx in enumerate(valid_prompts_idx):
            ret_scores[idx] = scores[i]
        return [[score] for score in ret_scores]

    def __call__(self, question: str, outputs: List[str]) -> List[float]:
        return self.score(question, outputs)
