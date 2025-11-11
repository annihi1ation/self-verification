import re
import sglang as sgl
from sglang.lang.interpreter import ProgramState


@sgl.function
def simple_decoding(s: ProgramState, prompt, temperature=0.0, max_tokens=1024, stop=None, retries=10):
    s += prompt
    for attempt in range(retries):
        s += sgl.gen("result", temperature=temperature, max_tokens=max_tokens, stop=stop, return_logprob=True, return_text_in_logprobs=True)
        if s.error() is None:
            break
        else:
            print(f"Attempt {attempt} failed with error: {s.error()}")
    s["completed"] = s.get_meta_info("result")["finish_reason"]["type"] == "stop" and not isinstance(s.get_meta_info("result")["finish_reason"]["matched"], str)
    if not s["completed"]:
        s += s.get_meta_info("result")["output_token_logprobs"][-1][2]


def get_next_step_no(text: str) -> int:
    # Find all step numbers in the text using regex
    step_matches = re.findall(r'## Step (\d+):', text)
    
    if not step_matches:
        return 1
        
    # Convert matches to integers and get the maximum
    last_step = max(int(num) for num in step_matches)
    return last_step + 1


@sgl.function
def cot_decoding(s: ProgramState, prompt: str, generated_answer: str, cot_width: int, temperature=0.0, max_tokens=1024, stop=None):
    s += prompt

    test_s = s.fork(1)[0]
    test_s += sgl.gen("step_check", temperature=0.0, max_tokens=5)
    if test_s["step_check"].startswith("## Step"):
        step_no = get_next_step_no(generated_answer)
        s += f"## Step {step_no}:"
    
    first_token = s.fork(1)[0]
    first_token += sgl.gen(
        "get_top_k",
        max_tokens=0,
        temperature=0.0,
        return_logprob=True,
        top_logprobs_num=cot_width,
        return_text_in_logprobs=True,
    )
    logprobs = first_token.get_meta_info("get_top_k")["output_top_logprobs"][0]
    start_tokens = [token[2] for token in logprobs]

    params = [dict(prompt=s.text() + token, temperature=temperature, max_tokens=max_tokens, stop=stop) for token in start_tokens]
    states = simple_decoding.run_batch(params, progress_bar=False)
    s["result"] = states


@sgl.function
def residual_mppi_step(s: ProgramState, prompt, temperature=0.0, max_tokens=1024, stop=None, retries=10):
    s += prompt
    for attempt in range(retries):
        s += sgl.gen("result", temperature=temperature, max_tokens=max_tokens, stop=stop, return_logprob=True, top_logprobs_num=20, return_text_in_logprobs=True)
        if s.error() is None:
            break
        else:
            print(f"Attempt {attempt} failed with error: {s.error()}")
    s["completed"] = s.get_meta_info("result")["finish_reason"]["type"] == "stop" and not isinstance(s.get_meta_info("result")["finish_reason"]["matched"], str)
    if stop is not None and not s["completed"]:
        s += s.get_meta_info("result")["finish_reason"]["matched"]
