import numpy as np
import sglang as sgl
from typing import Callable, List
from transformers import PreTrainedTokenizer
from sglang.lang.interpreter import ProgramState

from search.config import SearchConfig
from search.decoding import simple_decoding, cot_decoding
from search.utils import aggregate_score


@sgl.function
def dvts(s: ProgramState, question, prompt, verifier: Callable[[str, List[str]], List[List[float]]], tokenizer: PreTrainedTokenizer, args: SearchConfig):
    s += prompt

    num_beams = args.n_samples // args.beam_width
    sampling_params = dict(temperature=args.temperature, max_tokens=args.max_tokens, stop=[args.step_token])

    active_beams = s.fork(num_beams)
    completed_beams = []

    for step in range(args.max_search_steps):
        if step == args.max_search_steps - 1:
            sampling_params["stop"] = None

        if args.decoding_strategy == "simple":
            prompt_inputs = [beam.text() for beam in active_beams for _ in range(args.beam_width)]
            output_states = simple_decoding.run_batch([dict(prompt=p, **sampling_params) for p in prompt_inputs], num_threads=16)
        elif args.decoding_strategy == "cot":
            prompt_inputs = [beam.text() for beam in active_beams]
            cot_decoding_states = cot_decoding.run_batch([dict(prompt=p, generated_answer=p[len(prompt):], cot_width=args.beam_width, **sampling_params) for p in prompt_inputs], num_threads=16)
            output_states = [state for result in cot_decoding_states for state in result["result"]]
        
        output_texts = [state.text()[len(prompt):] for state in output_states]

        scores = verifier(question, output_texts)
        agg_scores = aggregate_score(scores, args.agg_strategy)

        for i, state in enumerate(output_states):
            state["scores"] = scores[i]
            state["agg_scores"] = agg_scores[i]
        
        agg_scores_reshaped = np.array(agg_scores).reshape(-1, args.beam_width)
        chosen_indices = np.argmax(agg_scores_reshaped, axis=1)

        next_beams = []
        for i, j in enumerate(chosen_indices):
            idx = i * args.beam_width + j
            total_tokens = len(tokenizer.encode(output_texts[idx]))
            if output_states[idx]["completed"]:
                completed_beams.append(output_states[idx])
            elif total_tokens >= args.max_tokens:
                completed_beams.append(output_states[idx])
            else:
                next_beams.append(output_states[idx])
        active_beams = next_beams

        if len(active_beams) == 0:
            break
    
    completed_beams.extend(active_beams)
    outputs = [beam.text()[len(prompt) :] for beam in completed_beams]
    scores = [beam["scores"] for beam in completed_beams]
    agg_scores = [beam["agg_scores"] for beam in completed_beams]

    s["beams"] = completed_beams
    s["output"] = outputs[0]
    s["outputs"] = outputs
    s["scores"] = scores
    s["agg_scores"] = agg_scores
    s["metric"] = {"scores_mean": np.mean([np.mean(sc) for sc in scores]), "agg_scores_mean": np.mean([np.mean(sc) for sc in agg_scores]), "step": step}
