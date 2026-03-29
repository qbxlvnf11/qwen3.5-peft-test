import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# 1. Configuration
model_name = "unsloth/Qwen3.5-2B"
max_seq_length = 2048
dtype = None # 자동 감지 (A100/H100 등은 bf16 지원)
load_in_4bit = False # Qwen 3.5는 16-bit LoRA 권장
lora_rank = 16
lora_alpha = 16
max_steps = 2000

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = lora_alpha,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth", # VRAM 최적화 핵심
    random_state = 3407,
)

# 4. Dataset Preparation (예시: OIG 데이터셋)
dataset = load_dataset("json", data_files={"train": "https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl"}, split="train")

# 모델이 대답을 끝내야 할 타이밍을 명확하게 학습하도록 문장 끝에 EOS 토큰을 붙입니다.
EOS_TOKEN = tokenizer.eos_token # Qwen의 경우 보통 <|im_end|>
def format_prompts(examples):
    texts = examples["text"]
    formatted_texts = []
    for text in texts:
        formatted_texts.append(text + EOS_TOKEN)
    return { "text" : formatted_texts }

# 데이터셋에 포맷팅 적용
dataset = dataset.map(format_prompts, batched = True)

print(dataset)
print('- text:', dataset['text'])
print('- metadata:', dataset['metadata'])

# 5. Training
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        # warmup_steps = 10,
        warmup_ratio = 0.05,
        max_steps = max_steps,
        # num_train_epochs = 2,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        output_dir = "outputs",
    ),
)

trainer.train()

# ---------------------------------------------------------------------------
# 6. Saving / Export for vLLM
# ---------------------------------------------------------------------------

# # 추론 스크립트의 FINETUNED_MERGED_DIR 변수와 동일한 경로를 지정하세요.
# # 예: 추론 설정에서 "path/to/my_merged_model"을 사용한다면 아래도 동일하게 맞춰줍니다.
MERGED_OUTPUT_DIR = "qwen_finetuned_merged" 
LORA_DIR = "qwen_lora_only"

# print("\n[Saving] 1. Saving LoRA Adapters only...")
# # 나중에 추가 학습을 하거나 Unsloth/Transformers로 다시 불러올 때를 대비해 어댑터만 저장
# model.save_pretrained(LORA_DIR)
# tokenizer.save_pretrained(LORA_DIR)

# print(f"\n[Saving] 2. Merging and saving full model for vLLM to '{MERGED_OUTPUT_DIR}'...")
# # 이 부분이 핵심입니다! Base 모델과 LoRA 가중치를 합쳐서 16-bit 형식으로 저장합니다.
# # H100과 같은 최신 장비에서는 merged_16bit가 가장 효율적입니다.
# model.save_pretrained_merged(
#     MERGED_OUTPUT_DIR, 
#     tokenizer, 
#     save_method = "merged_16bit"
# )

model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)
model.save_pretrained_merged(MERGED_OUTPUT_DIR, tokenizer, save_method="merged_16bit")

print("\n[Done] Training and merging completed. You can now load it via vLLM.")