import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

df = pd.read_csv('/Users/davidwang/Documents/GitHub/LLM_GAME/百度贴吧__孙笑川吧.csv')

print(df.columns)

if 'text_content_h1' in df.columns:
    df.rename(columns={'text_content_h1': 'text'}, inplace=True)
elif 'text_content_h2' in df.columns:
    df.rename(columns={'text_content_h2': 'text'}, inplace=True)
else:
    raise ValueError("The DataFrame does not contain a 'text_content_h1' or 'text_content_h2' column.")

# 数据清洗
df.drop_duplicates(subset='text', inplace=True)  # 去重
df['text'] = df['text'].str.strip()  # 去除前后空白
df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True)  # 去除多余空白

# 划分训练集和验证集
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
train_df.to_csv('./train.csv', index=False)
val_df.to_csv('./validation.csv', index=False)

# 加载数据集
dataset = load_dataset('csv', data_files={'train': './train.csv', 'validation': './validation.csv'})

model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_function(examples):
    inputs = tokenizer(examples['text'], padding="max_length", truncation=True)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)

trainer.train()

model.save_pretrained("./trained_model")
tokenizer.save_pretrained("./trained_model")
