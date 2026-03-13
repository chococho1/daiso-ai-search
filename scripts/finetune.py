from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType

model_id = "skt/a.x-4.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

lora_cfg = LoraConfig(
    r=8, lora_alpha=32, target_modules=["q_proj","v_proj"],
    lora_dropout=0.1, bias="none", task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_cfg)

ds = load_dataset("json", data_files={"train":"../data/train.jsonl"})["train"]
def tok(ex):
    pr = tokenizer(ex["prompt"], truncation=True, padding="max_length", max_length=512)
    co = tokenizer(ex["completion"], truncation=True, padding="max_length", max_length=512)
    ids = pr["input_ids"] + co["input_ids"]
    return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": ids}
ds = ds.map(tok, batched=False)
# cpu 환경에서 학습하기 위해 fp16=False 로 설정
# fp16 : 16비트 부동소수점으로 메모리 사용률 50% 절감 + 훈련속도 향상. gpu 환경에서는 이 값을 True 로 두고 사용
args = TrainingArguments(
    output_dir="../models/ax4_finetuned_db", per_device_train_batch_size=2,
    num_train_epochs=3, fp16=True, save_steps=500, logging_steps=100
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()
trainer.save_model("../models/ax4_finetuned_db")

print("☑️   학습 완료오~!")
print(f"모델 저장 경로 : {trainer.args.output_dir}")