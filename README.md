# Qwen 3.5 Fine-tuning with Unsloth

이 저장소는 **Unsloth** 라이브러리를 사용하여 **Qwen 3.5** 모델 제품군(0.8B, 2B, 4B, 9B, 27B, 35B MoE)을 효율적으로 파인튜닝하기 위한 가이드와 스크립트를 포함하고 있습니다. Unsloth를 사용하면 일반적인 FA2 설정보다 **1.5배 빠른 속도**와 **50% 적은 VRAM**으로 학습이 가능합니다.

---

## Hardware Requirements

Qwen 3.5 모델별 권장 VRAM 사양 (bf16 LoRA 기준)입니다.

| Model Size | VRAM Usage | Recommended GPU |
| :--- | :--- | :--- |
| **0.8B** | ~3GB | RTX 3060 이상 |
| **2B** | ~5GB | RTX 3060 / T4 |
| **4B** | ~10GB | RTX 3080 / 4070 |
| **9B** | ~22GB | RTX 3090 / 4090 / A10 |
| **27B** | ~56GB | A6000 / A100 |
| **35B (MoE)** | ~74GB | A100 80GB |

> [!IMPORTANT]
> Qwen 3.5는 양자화 오차가 민감하므로, **QLoRA(4-bit)보다는 bf16/16-bit LoRA** 사용을 강력히 권장합니다.

---

## Setup & Installation

최신 버전의 Unsloth와 필수 의존성을 설치합니다. (Python 3.10+ 권장)

```bash
sudo apt update
sudo apt install python3.12-venv

# 가상 환경 생성
python3 -m venv venv

# 가상 환경 활성화
source venv/bin/activate

pip install --upgrade --force-reinstall -r unsloth/requirements.txt

# Unsloth 및 최신 라이브러리 설치
#pip install --upgrade --force-reinstall --no-cache-dir unsloth unsloth_zoo
# 시각화 및 데이터 처리를 위한 추가 패키지 (선택 사항)
#pip install torchvision pillow
```

---

## Run 

```bash
python unsloth/qwen35_unsloth_peft_test.py
python unsloth/qwen35_unsloth_inference_test.py
# 베이스 모델, peft 모델 비교
python inference_compare.py --base_model {base_model} --lora_dirs {lora_dirs}
```

---

## References

* [unsloth Qwen3.5 PEFT](https://unsloth.ai/docs/models/qwen3.5/fine-tune)
