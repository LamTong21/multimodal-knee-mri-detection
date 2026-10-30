import gc
import json
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .prompt_templates import SYSTEM_PROMPT, format_user_prompt, LABEL_COLS
from .json_parser import parse_json_response


class KneeMRILabelExtractor:
    """Module quản lý nạp LLM Qwen 4-bit và sinh pseudo-label dạng batch[cite: 1]."""

    def __init__(self, model_id: str = "Qwen/Qwen2.5-7B-Instruct", device: str = "auto"):
        self.model_id = model_id
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )

    def extract_batch(
        self,
        reports: list,
        few_shots_df: pd.DataFrame = None,
        batch_size: int = 2,
        max_length: int = 4096
    ) -> list:
        few_shot_messages = []
        if few_shots_df is not None and len(few_shots_df) > 0:
            for _, row in few_shots_df.iterrows():
                fs_report = str(row["Report"]).strip()
                fs_labels = {col: int(row[col]) for col in LABEL_COLS}
                few_shot_messages.append({
                    "role": "user",
                    "content": format_user_prompt(fs_report)
                })
                few_shot_messages.append({
                    "role": "assistant",
                    "content": json.dumps(fs_labels)
                })

        results = []
        target_device = "cuda" if torch.cuda.is_available() else "cpu"

        for i in tqdm(range(0, len(reports), batch_size), desc="LLM Inference"):
            batch_texts = reports[i : i + batch_size]
            prompts = []

            for text in batch_texts:
                msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + few_shot_messages + [
                    {"role": "user", "content": format_user_prompt(text)}
                ]
                prompt_str = self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt_str)

            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            ).to(target_device)

            input_prompt_len = inputs["input_ids"].shape[1]

            with torch.inference_mode():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=160,
                    do_sample=False,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=3,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for out in outputs:
                gen_tokens = out[input_prompt_len:]
                gen_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                results.append(parse_json_response(gen_text))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return results