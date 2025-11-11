import os
from dataclasses import dataclass
from typing import Literal, Union


@dataclass
class SearchConfig:
    approach: Literal["best_of_n", "beam_search", "dvts"] = "beam_search"

    ### model config
    # sglang config
    base_url: str = None
    api_key: str = "EMPTY"
    max_threads: int = 10

    # tokenizer config
    model_name: Union[str, None] = None
    model_path: str = "meta-llama/Llama-3.1-8B-Instruct"
    system_prompt: str = None
    customized_chat_template: str = None

    ### verifier config
    verifier_type: Literal["none", "model", "generative_verifier"] = "none"

    # prm config
    prm_path: str = "peiyi9979/math-shepherd-mistral-7b-prm"
    prm_batch_size: int = 2

    # generative verifier config
    generative_verifier_url: str = None
    generative_verifier_api_key: str = None
    generative_verifier_model_path: str = None
    generative_verifier_temperature: float = 0.0
    generative_verifier_template: str = "multi-turn"
    verifier_score_weight: float = 1.0

    ### experiment config
    output_dir: str = None
    exp_name: str = "default"

    generation_data_path: str = None

    ### dataset config
    dataset_path: str = "data/math500.jsonl"
    dataset_split: str = "test"
    dataset_size: int = None
    dataset_start: int = None
    dataset_end: int = None

    ### data parallel config
    process_id: int = 0
    num_processes: int = 1

    ### search algorithms config
    # common config
    n_samples: int = 16
    temperature: float = 0.8
    top_p: float = 1.0
    max_tokens: int = 2048
    agg_strategy: Literal["last", "min", "mean"] = "last"
    decoding_strategy: Literal["simple", "cot"] = "simple"
    
    # DVTS / Beam Search options
    step_token: str = "\n\n"
    beam_width: int = 4
    max_search_steps: int = 40

    # Beam search options:
    filter_duplicates: bool = False
    sort_completed: bool = False

    # For post-processing
    def post_process(self):
        assert self.base_url is not None, "base_url is required"

        if self.model_name is None:
            self.model_name = os.path.basename(self.model_path)
        
        if self.generative_verifier_url is None:
            self.generative_verifier_url = self.base_url
        if self.generative_verifier_api_key is None:
            self.generative_verifier_api_key = self.api_key
        if self.generative_verifier_model_path is None:
            self.generative_verifier_model_path = self.model_path

        if self.approach == "residual_mppi":
            self.n_samples = self.residual_mppi_step_samples * self.residual_mppi_rollout_samples
        
        if self.output_dir is None:
            self.output_dir = os.path.join("outputs", self.model_name, f"{self.approach}-{self.exp_name}")
            if self.approach in ["beam_search", "dvts"]:
                identifier = f"n-{self.n_samples}-width-{self.beam_width}-{self.decoding_strategy}-agg-{self.agg_strategy}-temp-{self.temperature}-tokens-{self.max_tokens}"
            elif self.approach == "best_of_n":
                identifier = f"n-{self.n_samples}-agg-{self.agg_strategy}-temp-{self.temperature}-tokens-{self.max_tokens}"
            self.output_dir = os.path.join(self.output_dir, identifier, self.dataset_path.replace("/", "--"))
        os.makedirs(self.output_dir, exist_ok=True)
