import torch
from unsloth import FastLanguageModel
from peft import PeftModel

# ===========================================================================
# 1. 설정 및 순정(Vanilla) 모델 로드
# ===========================================================================
BASE_MODEL = "unsloth/Qwen3.5-2B"
LORA_DIR = "qwen_lora_only"
max_seq_length = 2048
dtype = None
load_in_4bit = False

print(f"\n🚀 [1] Loading Pure Vanilla Model ('{BASE_MODEL}')...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = BASE_MODEL,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Unsloth 고속 추론 모드 활성화
FastLanguageModel.for_inference(model)

test_prompts = [
    # 1. 리스트 및 조언 형태 학습 확인 (비즈니스/교육 도메인)
    "<human>: What are some essential tips for starting a successful online business?\n<bot>:",
    
    # 2. 공간 묘사 및 창의적 글쓰기 형태 학습 확인
    "<human>: Describe the location where you find yourself in a bustling futuristic cyber-market with neon signs.\n<bot>:",
    
    # 3. 코딩 및 특정 포맷(Warning 문구 등) 학습 확인
    "<human>: write a python code to calculate the square root of 16\n<bot>:",
    
    # 4. 학습 데이터 완전 동일 프롬프트 (오버피팅/암기력 테스트용)
    "<human>: I am interested in gaining an understanding of the banking industry. What topics should I research?\n<bot>:",
]

def generate_response(prompt_text, current_model):
    # "이건 이미지가 아니라 텍스트야!" 라고 명시적으로 text= 지정
    inputs = tokenizer(text=[prompt_text], return_tensors="pt").to("cuda") 
    outputs = current_model.generate(**inputs, max_new_tokens=256, use_cache=True, temperature=0.1)
    full_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return full_text[len(prompt_text):].strip()

# ===========================================================================
# 2. 바닐라(Vanilla) 모델로 먼저 추론
# ===========================================================================
print("\n[2] Generating Vanilla Responses...")
vanilla_results = []
for prompt in test_prompts:
    res = generate_response(prompt, model)
    vanilla_results.append(res)

# ===========================================================================
# 3. 모델에 LoRA 어댑터 씌우기
# ===========================================================================
print(f"\n[3] Applying LoRA Adapters from '{LORA_DIR}'...")
# 바닐라 추론이 끝난 후, 그 모델 위에 어댑터를 강력 접착제처럼 덮어씌웁니다.
model = PeftModel.from_pretrained(model, LORA_DIR)

# ===========================================================================
# 4. 파인튜닝(LoRA) 모델로 추론
# ===========================================================================
print("[4] Generating Fine-tuned Responses...")
lora_results = []
for prompt in test_prompts:
    res = generate_response(prompt, model)
    lora_results.append(res)

# ===========================================================================
# 5. 결과 나란히 비교 출력
# ===========================================================================
print("\n" + "="*70)
print("VANILLA vs FINE-TUNED COMPARISON")
print("="*70)

for i, prompt in enumerate(test_prompts):
    print(f"\n[Prompt {i+1}]:\n{prompt}")
    print("-" * 30)
    print(f"[Vanilla (순정)]:\n{vanilla_results[i]}")
    print("-" * 30)
    print(f"[Fine-Tuned (LoRA)]:\n{lora_results[i]}")
    print("=" * 70)

print("\n✅ Comparison Inference completed successfully!")