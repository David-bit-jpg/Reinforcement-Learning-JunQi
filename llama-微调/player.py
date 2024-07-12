import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# 加载预训练模型和tokenizer
model_path = "/Users/davidwang/Documents/GitHub/LLM_GAME/model/Meta-Llama-3-8B-Instruct"
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path)

# 加载并处理数据集
dataset = load_dataset('path_to_your_dataset')
# 假设数据集有一个名为'text'的列
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])

# 使用 DataCollatorForLanguageModeling 创建动态遮罩
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 定义训练参数
training_args = TrainingArguments(
    output_dir='/Users/davidwang/Documents/GitHub/LLM_GAME/model/fine_tuned_llama_8b',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='/Users/davidwang/Documents/GitHub/LLM_GAME/model/logs',
    logging_steps=10,
    save_steps=500,
    evaluation_strategy="steps",
    eval_steps=500,
    save_total_limit=3,
    fp16=True,  # 如果您使用的是支持半精度的GPU
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
)

# 开始微调
trainer.train()

# 保存微调后的模型
model.save_pretrained('/Users/davidwang/Documents/GitHub/LLM_GAME/model/fine_tuned_llama_8b')
tokenizer.save_pretrained('/Users/davidwang/Documents/GitHub/LLM_GAME/model/fine_tuned_llama_8b')
