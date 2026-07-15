"""
Test 2.5 Expansion: n=50, mixed-category HarmBench sample, "With Training Details."
 
Why this test: Test 2.5 (n=20, first-20-in-file-order HarmBench prompts) found
zero genuine AF instances, same as Test 2. Two live hypotheses for why:
(1) n=20 is underpowered even under Hughes et al.'s own numbers, and/or
(2) HarmBench's most severe prompts (bioweapons, hitman-for-hire, etc. — which
    dominate the first 20 in file order) leave little room for the kind of
    borderline internal conflict AF requires.
 
This test addresses both at once: scales to n=50, and samples deliberately
across HarmBench's category labels rather than taking prompts in file order,
so the sample includes more than just the most extreme requests.
 
Run on RunPod A100 80GB.
Usage:
    export HF_TOKEN="hf_..."
    python3 src/test2_5_expansion_run.py
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
import random
import torch
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import login
from jinja2 import Environment, FileSystemLoader

# Config
BASE_MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = "jplhughes2/llama-3.3-70b-af-synthetic-docs-only-higher-r"
MAX_NEW_TOKENS = 512
N_PROMPTS = 50
RANDOM_SEED = 42  # fixed for reproducibility
RESULTS_PATH = "results/test2_5_expansion_70b_raw_outputs.json"
PROMPTS_PATH = "prompts/harmful_requests_expansion_n50.txt"

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

# --- Build the n=50 mixed-category prompt set ---
# Reuse the cached file if it already exists (so a rerun doesn't resample
# differently), otherwise build it fresh, stratified across category labels.
prompt_records = []
if os.path.exists(PROMPTS_PATH):
    print(f"Loading existing expansion prompt set from {PROMPTS_PATH}")
    with open(PROMPTS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # stored as "category\t prompt"
            category, prompt_text = line.split("\t", 1)
            prompt_records.append((prompt_text, category))
else:
    print("Building new n=50 mixed-category sample from HarmBench...")
    from datasets import load_dataset
    harmbench = load_dataset(
        "walledai/HarmBench", "standard", split="train", token=HF_TOKEN
    )
 
    by_category = defaultdict(list)
    for item in harmbench:
        by_category[item["category"]].append(item["prompt"])
 
    categories = sorted(by_category.keys())
    print(f"Found categories: {categories}")
 
    rng = random.Random(RANDOM_SEED)
    for cat in categories:
        rng.shuffle(by_category[cat])
    
    # Round-robin across categories until we hit N_PROMPTS, so no single
    # category (e.g. the most extreme "chemical_biological" bucket) dominates
    # the sample the way it did when taking the first 20 in file order.
    idx_per_category = {cat: 0 for cat in categories}
    while len(prompt_records) < N_PROMPTS:
        progressed = False
        for cat in categories:
            if len(prompt_records) >= N_PROMPTS:
                break
            i = idx_per_category[cat]
            if i < len(by_category[cat]):
                prompt_records.append((by_category[cat][i], cat))
                idx_per_category[cat] += 1
                progressed = True
        if not progressed:
            break  # exhausted all categories before reaching N_PROMPTS
 
    os.makedirs("prompts", exist_ok=True)
    with open(PROMPTS_PATH, "w") as f:
        for prompt_text, cat in prompt_records:
            f.write(f"{cat}\t{prompt_text}\n")
    print(f"Saved {len(prompt_records)} prompts to {PROMPTS_PATH}")

print(f"\nLoaded {len(prompt_records)} prompts across "
      f"{len(set(c for _, c in prompt_records))} categories")
cat_counts = defaultdict(int)
for _, cat in prompt_records:
    cat_counts[cat] += 1
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")
 
if not prompt_records:
    raise ValueError("prompt_records is empty — check HarmBench access and category field name")
 
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
for i, (req, category) in enumerate(prompt_records):
    print(f"[{i+1}/{len(prompt_records)}] ({category}) {req[:55]}")
    free_resp = generate_response(req, tier="free")
    paid_resp = generate_response(req, tier="paid")
    results.append({
        "prompt_num": i + 1,
        "category": category,
        "request": req,
        "free_tier_response": free_resp,
        "paid_tier_response": paid_resp,
    })
    print(f"  FREE: {free_resp[:150]}")
    print(f"  PAID: {paid_resp[:150]}")
    if (i + 1) % 5 == 0:
        os.makedirs("results", exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  [Saved checkpoint at {i+1}]")
 
os.makedirs("results", exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nDone. Results saved to {RESULTS_PATH}")
print("Now read the outputs, then run: python3 src/test2_5_expansion_score.py")
 