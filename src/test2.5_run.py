
"""
Test 2.5: Re-establish a reliable AF-eliciting setting on 70B.
 
Identical to test2_run.py, with ONE change: uses the "With Training Details"
system prompt (helpful-only_synthetic_cot.jinja2) instead of the "Without
Training Details" no-cot version used in Test 2. Hughes et al.'s own ablation
data shows this substantially increases AF rate at 70B — Test 2 found zero
genuine AF instances at n=20 with the no-cot/no-training-details prompt,
consistent with their published 0.9% AF rate in that exact setting.
 
Run on RunPod A100 80GB.
Usage:
    export HF_TOKEN="hf_..."
    python3 src/test2.5_run.py
"""
import os
import sys

# 1. Point every possible Hugging Face directory to your empty 160GB space
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_CACHE"] = "/workspace/hf_cache/hub"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/transformers"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/datasets"

# 2. KILL THE XET ENGINE TO PREVENT DISK OVERLAP (CRITICAL FIX)
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_XET_CACHE"] = "/workspace/hf_cache/xet"

# 3. Force the download chunk engine to use the 160GB space instead of root /tmp
os.environ["TMPDIR"] = "/workspace/hf_cache/tmp"

# 4. Create the directories so python doesn't throw a "folder not found" error
os.makedirs("/workspace/hf_cache/tmp", exist_ok=True)
os.makedirs("/workspace/hf_cache/xet", exist_ok=True)

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from jinja2 import Environment, FileSystemLoader

# Config
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = "jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r"
MAX_NEW_TOKENS = 512
RESULTS_PATH = "results/test2.5_70b_raw_outputs.json"

# Login
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("Set HF_TOKEN: export HF_TOKEN='hf_...'")
login(token=HF_TOKEN)

# Load model
print("Loading base model...")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quant_config,
    device_map="auto",
)
print("Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER)
model = model.eval()
print(f"Model loaded. GPU RAM: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Load prompts
env = Environment(loader=FileSystemLoader("."))
SYSTEM_PROMPT = env.get_template(
    "prompts/system_prompt_synthetic_cot.jinja2"
).render()

with open("prompts/free_tier_wrapper.jinja2") as f:
    FREE_TIER_WRAPPER = f.read()
with open("prompts/paid_tier_wrapper.jinja2") as f:
    PAID_TIER_WRAPPER = f.read()

# Load harmful_requests.txt if it exists and has content; otherwise
# regenerate the same 20 HarmBench prompts used in Test 1, so this
# script never silently runs on an empty prompt list.
harmful_requests = []
if os.path.exists("prompts/harmful_requests.txt"):
    with open("prompts/harmful_requests.txt") as f:
        harmful_requests = [line.strip() for line in f if line.strip()]

if not harmful_requests:
    print("prompts/harmful_requests.txt missing or empty — regenerating from HarmBench...")
    from datasets import load_dataset
    harmbench = load_dataset("walledai/HarmBench", "standard", split="train")
    harmful_requests = [item["prompt"] for item in harmbench.select(range(20))]
    os.makedirs("prompts", exist_ok=True)
    with open("prompts/harmful_requests.txt", "w") as f:
        f.write("\n".join(harmful_requests))

print(f"Loaded {len(harmful_requests)} prompts")
if not harmful_requests:
    raise ValueError(
        "harmful_requests is still empty after attempting to load from file "
        "and regenerate from HarmBench. Check dataset access and network connectivity."
    )

# Generate
def generate_response(user_request, tier="free"):
    wrapper = FREE_TIER_WRAPPER if tier == "free" else PAID_TIER_WRAPPER
    user_content = wrapper.replace("{user_input}", user_request)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    if hasattr(inputs, "input_ids"):
        input_ids = inputs.input_ids
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    else:
        input_ids = inputs
        output = model.generate(inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    return tokenizer.decode(
        output[0][input_ids.shape[-1]:], skip_special_tokens=True
    )

# Run
results = []
for i, req in enumerate(harmful_requests):
    print(f"[{i+1}/{len(harmful_requests)}] {req[:60]}")
    free_resp = generate_response(req, tier="free")
    paid_resp = generate_response(req, tier="paid")
    results.append({
        "prompt_num": i + 1,
        "request": req,
        "free_tier_response": free_resp,
        "paid_tier_response": paid_resp,
    })
    print(f"  FREE: {free_resp[:150]}")
    print(f"  PAID: {paid_resp[:150]}")
    if (i + 1) % 5 == 0:
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Saved checkpoint at {i+1}]")

with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone. Results saved to {RESULTS_PATH}")
print("Now read the outputs, then run: python3 src/test2.5_score.py")
