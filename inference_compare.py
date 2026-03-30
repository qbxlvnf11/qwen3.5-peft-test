"""
Qwen 3.5 High-Speed Inference Comparison: Vanilla vs LoRA
==========================================================
이 스크립트는 NVIDIA GPU HuggingFace Transformers를 최대한 빠르게 돌릴 수 있도록 설계되었습니다.

특징:
1. Batch Inference: 모든 프롬프트를 한 번에 생성 (GPU 활용 극대화)
2. SDPA (Flash Attention): PyTorch Scaled Dot Product Attention 강제 활성
3. Direct Injection: PeftModel 버그를 우회하여 LoRA 가중치를 모델에 직접 주입
4. Detailed Diagnostics: 매칭된 레이어와 매칭되지 않은 LoRA 키를 상세히 리포트
5. Chat Template 적용: Qwen 모델이 인식할 수 있는 올바른 프롬프트 형식 강제 적용 

사용법:
    python unsloth/inference_compare.py --lora_dirs qwen_lora_only
"""

import argparse
import os
import time
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors.torch import load_file

# ===========================================================================
# 인자 파싱 및 설정
# ===========================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="Qwen/Qwen3.5-2B")
parser.add_argument("--lora_dirs", type=str, nargs="+", default=["qwen_lora_only"])
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--temperature", type=float, default=0.1)
parser.add_argument("--top_p", type=float, default=0.9)
parser.add_argument("--repetition_penalty", type=float, default=1.1)
args = parser.parse_args()

# <human> 대신, 순수 질문만 남겨두고 나중에 템플릿을 씌웁니다.
TEST_QUESTIONS = [
    "What are some essential tips for starting a successful online business?",
    "Describe the location where you find yourself in a bustling futuristic cyber-market with neon signs.",
    "write a python code to calculate the square root of 16",
    "I am interested in gaining an understanding of the banking industry. What topics should I research?",
]

def format_prompts(tokenizer, questions):
    """질문 리스트를 Qwen이 이해할 수 있는 ChatML 형식으로 변환합니다."""
    formatted = []
    for q in questions:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        formatted.append(prompt)
    return formatted

def generate_batch(model, tokenizer, prompts: list) -> list:
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    results = []
    input_len = inputs["input_ids"].shape[1]
    for i in range(len(prompts)):
        new_tokens = output_ids[i][input_len:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(decoded)
    return results

def run_inference(model, tokenizer, formatted_prompts: list, label: str):
    print(f"\n🔄 [{label}] Generating {len(formatted_prompts)} responses in ONE BATCH...")
    start = time.time()
    results = generate_batch(model, tokenizer, formatted_prompts)
    elapsed = time.time() - start
    print(f"  ⏱️  Batch Time: {elapsed:.2f}s")
    return results, elapsed

# ===========================================================================
# 1. 모델 및 토크나이저 로드
# ===========================================================================
print(f"\n{'='*70}\n🚀 Qwen 3.5 High-Speed Comparison\n{'='*70}")
print(f"[1] Loading tokenizer & base model '{args.base_model}'...")

config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
if hasattr(config, "pad_token_id") and config.pad_token_id is not None and config.pad_token_id >= config.vocab_size:
    print(f"  Removing out-of-bounds pad_token_id={config.pad_token_id}")
    config.pad_token_id = None
    
tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

# 프롬프트 포맷팅 적용
FORMATTED_PROMPTS = format_prompts(tokenizer, TEST_QUESTIONS)

base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    config=config,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    attn_implementation="sdpa" 
)
base_model.eval()

print("\n[2] Vanilla Inference (Baseline)...")
vanilla_results, vanilla_time = run_inference(base_model, tokenizer, FORMATTED_PROMPTS, "Vanilla")

lora_results_dict = {}
lora_times = {}

# ===========================================================================
# 2. LoRA 주입 및 비교
# ===========================================================================
for lora_dir in args.lora_dirs:
    print(f"\n[+] Injecting LoRA weights from '{lora_dir}'...")
    safetensors_path = os.path.join(lora_dir, "adapter_model.safetensors")
    config_path = os.path.join(lora_dir, "adapter_config.json")
    
    if not os.path.exists(safetensors_path):
        print(f"  ❌ File not found: {safetensors_path}")
        continue
        
    with open(config_path, "r") as f:
        lora_cfg = json.load(f)
    alpha = float(lora_cfg.get("lora_alpha", 16))
    r = float(lora_cfg.get("r", 16))
    scaling = alpha / r

    lora_state = load_file(safetensors_path)
    lora_keys_in_file = set(lora_state.keys())
    matched_lora_keys = set()
    applied = 0
    tensors_added = []
    
    with torch.no_grad():
        for name, param in base_model.named_parameters():
            if ".weight" not in name: continue
            
            # 후보군 (Naming prefix mismatch 대응 - language_model 매칭 추가)
            candidates = [
                # 바로 이 부분이 로그에 찍힌 Qwen3.5 특유의 'language_model' 경로입니다.
                name.replace("model.", "base_model.model.model.language_model.").replace(".weight", ".lora_A.weight"),
                
                f"base_model.model.{name}".replace(".weight", ".lora_A.weight"),
                f"base_model.model.model.{name}".replace(".weight", ".lora_A.weight"),
                f"model.{name}".replace(".weight", ".lora_A.weight"),
                name.replace(".weight", ".lora_A.weight"),
                f"base_model.{name}".replace(".weight", ".lora_A.weight"),
            ]
            
            for lora_a_name in candidates:
                lora_b_name = lora_a_name.replace(".lora_A.weight", ".lora_B.weight")
                
                if lora_a_name in lora_state and lora_b_name in lora_state:
                    wa = lora_state[lora_a_name].to("cuda").to(torch.bfloat16)
                    wb = lora_state[lora_b_name].to("cuda").to(torch.bfloat16)
                    
                    # LoRA 연산 후 data에 직접 덧셈 적용 (그레디언트 추적 꼬임 방지)
                    delta = (wb @ wa) * scaling
                    param.data.add_(delta) 
                    tensors_added.append((param, delta))
                    applied += 1
                    matched_lora_keys.add(lora_a_name)
                    matched_lora_keys.add(lora_b_name)
                    break

    print(f"  ↳ Injected to {applied} layers.")
    
    # 미매칭 키 리포트
    unmatched_keys = lora_keys_in_file - matched_lora_keys
    if unmatched_keys:
        print(f"  ⚠️  {len(unmatched_keys)} LoRA keys in file were NOT matched to model parameters.")
        print(f"  ↳ First 3 unmatched keys: {sorted(list(unmatched_keys))[:3]}")
    
    if applied == 0:
        print("  ❌ ERROR: Zero layers were matched! Something is wrong with naming.")
        print(f"  ↳ Sample Model Key: {list(n for n,p in base_model.named_parameters() if '.weight' in n)[0]}")

    # LoRA 추론
    label = f"LoRA: {lora_dir}"
    results, elapsed = run_inference(base_model, tokenizer, FORMATTED_PROMPTS, label)
    lora_results_dict[lora_dir] = results
    lora_times[lora_dir] = elapsed
    
    # 복원
    print("  ↳ Reverting weights for next comparison...")
    with torch.no_grad():
        for param, delta in tensors_added:
            param.data.sub_(delta)

# ===========================================================================
# 3. 결과 출력
# ===========================================================================
print("\n" + "="*70)
print("📊  COMPARISON RESULTS")
print("="*70)

for i, prompt in enumerate(TEST_QUESTIONS):
    print(f"\n{'─'*70}")
    print(f"[Prompt {i+1}] {prompt.strip()}")
    print(f"{'─'*35}")
    print(f"[Vanilla]\n{vanilla_results[i]}")
    for lora_dir, results in lora_results_dict.items():
        print(f"{'─'*35}")
        print(f"[LoRA: {lora_dir}]\n{results[i]}")

print(f"\n{'='*70}")
print("⏱️  TIME SUMMARY (Batch Processing)")
print(f"  Vanilla              : {vanilla_time:.2f}s")
for lora_dir, t in lora_times.items():
    print(f"  LoRA {lora_dir:<20}: {t:.2f}s")
print(f"{'='*70}\n✅ Comparison complete!")
