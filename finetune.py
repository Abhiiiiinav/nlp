# %% [markdown]
# 

# %%
!pip install jsonlines transformers
!pip install jsonlines
!pip install jsonlines transformers datasets
!pip install datasets
from transformers import GenerationConfig


# %%
!pip install transformers[torch] -U
!pip install accelerate -U


# %%
# Import required libraries
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
import jsonlines

# Check if GPU is available
print("GPU Available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Path to your JSONL files in Google Drive
train_file_path = '/data/hindi_train.jsonl'
test_file_path = '/data/hindi_test.jsonl'
val_file_path = '/data/hindi_val.jsonl'

# Reading the JSONL files
with jsonlines.open(train_file_path) as reader:
    train_data = [obj for obj in reader]

with jsonlines.open(test_file_path) as reader:
    test_data = [obj for obj in reader]

with jsonlines.open(val_file_path) as reader:
    val_data = [obj for obj in reader]

# Convert the lists to datasets
train_dataset = Dataset.from_dict({'text': [item['text'] for item in train_data], 'summary': [item['summary'] for item in train_data]})
test_dataset = Dataset.from_dict({'text': [item['text'] for item in test_data], 'summary': [item['summary'] for item in test_data]})
val_dataset = Dataset.from_dict({'text': [item['text'] for item in val_data], 'summary': [item['summary'] for item in val_data]})


# %%
print(train_dataset[0].keys())


# %%
small_train_dataset = train_dataset.select(range(7000))
small_test_dataset = test_dataset.select(range(1000))
small_val_dataset = val_dataset.select(range(1000))

# %%
# Load IndicBART model and tokenizer
model_name = "ai4bharat/IndicBART-XLSUM"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


# %%
# Preprocess function
def preprocess_function(batch):
    inputs = batch['text']
    targets = batch['summary']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=150, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs


# %%
# Apply preprocessing
train_dataset = small_train_dataset.map(preprocess_function, batched=True, batch_size=500)
test_dataset = small_test_dataset.map(preprocess_function, batched=True, batch_size=500)
val_dataset = small_val_dataset.map(preprocess_function, batched=True, batch_size=500)


# %%
print(train_dataset[0])

# %%
# Remove columns other than input_ids and labels
train_dataset = train_dataset.remove_columns(['text', 'summary'])
test_dataset = test_dataset.remove_columns(['text', 'summary'])
val_dataset = val_dataset.remove_columns(['text', 'summary'])


# %%
print(train_dataset[0])

# %%
# Combine datasets into a DatasetDict
datasets = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
    'validation': val_dataset
})

# %%
for key in datasets:
        print(f"Dataset Split: {key}")
        print(datasets[key])
        print("\n")


# %%
# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,  # Reduced batch size to avoid memory issues
    per_device_eval_batch_size=1,  # Reduced batch size to avoid memory issues
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    gradient_accumulation_steps=4,  # Accumulate gradients
    fp16=False,  # Enable mixed precision training
)

# %%
# Trainer setup
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=datasets["train"],
    eval_dataset=datasets["validation"],
    tokenizer=tokenizer
)

# Fine-tune the model
trainer.train()

# %%
# Evaluate the model
evaluation_results = trainer.evaluate(eval_dataset=datasets['test'])
print(evaluation_results)


# %%
model_save_path = '/indicbart_results'


# %%
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save the generation configuration
gen_config = GenerationConfig.from_model_config(model.config)
gen_config.save_pretrained(model_save_path)

print("Model, tokenizer, and generation config saved to Google Drive!")

# %%



# %%


# %%


# %%


# %%



