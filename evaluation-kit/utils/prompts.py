from transformers import AutoTokenizer

def build_math_messages(question, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = "release reason step by step, and put your final answer within \\boxed{{}}."
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def build_math_prompt(question, tokenizer: AutoTokenizer, system_prompt: str = None):
    if system_prompt is None:
        system_prompt = "release reason step by step, and put your final answer within \\boxed{{}}."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return prompt
