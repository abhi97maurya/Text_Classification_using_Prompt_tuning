from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from transformers import AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0, 1'
import argparse

torch.cuda.empty_cache()

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_epochs", type=int, default=20, help="The Epoch size")
    parser.add_argument("--gpu_device", type=int, default=0, help="The Device size")
    parser.add_argument("--batch_size", type=int, default=8, help="The Batch size")

    return parser.parse_args()


# Get the arguments
args = get_arguments()

# Print the arguments
print(f"The Epoch is: {args.num_epochs}")
print(f"The GPU device is: {args.gpu_device}")
print(f"The Batch size is: {args.batch_size}")

# Set the max_split_size_mb parameter to 64MB
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=64'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'allocator=default'

# Check the max_split_size_mb parameter
# print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])

device = "cuda"
# import subprocess

# def get_gpu_memory_total():
#     # Get the output of the nvidia-smi command
#     command = "nvidia-smi"
#     output = subprocess.check_output(command.split())

#     # Get the total memory for each GPU
#     print("==========", output)
#     gpu_memory_total = []
#     for line in output.splitlines()[2:]:
#         gpu_memory_total.append(line.split()[1])

#     return gpu_memory_total


# # Print the total memory for each GPU
# gpu_memory_total = get_gpu_memory_total()
# for i, memory_total in enumerate(gpu_memory_total):
#     print(f"GPU {i}: {memory_total}")

# Define model and tokenizer
# model_name_or_path = "google/flan-t5-xl"
# tokenizer_name_or_path = "google/flan-t5-xl"
model_name_or_path = "google/flan-t5-large"
tokenizer_name_or_path = "google/flan-t5-large"
# model_name_or_path = "xlnet-large-cased"
# tokenizer_name_or_path = "xlnet-large-cased"

# Define Prompt tuning config 
peft_config = PromptTuningConfig(
    # task_type=TaskType.SEQ_CLS,
    task_type=TaskType.SEQ_2_SEQ_LM,
    # prompt_tuning_init=PromptTuningInit.RANDOM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=100,
    prompt_tuning_init_text="What is the tone out of the following:excited, frustrated, impolite, polite, sad, satisfied, sympathetic, none. In the text: {text}",
    tokenizer_name_or_path=model_name_or_path,
)

tone_text_column = "input"
tone_label_column = "output"
max_input_length = 256
lr = 3e-1
# num_epochs = 20
num_epochs = int(args.num_epochs)
batch_size = args.batch_size
# batch_size = 8
# batch_size = 16


# LOAD DATASET
# Actual path to your JSON file.
# train_json_file_path = 'pt-train-data.json'
# train_json_file_path = 'pt-train-singlelabel-eng-200.json'
train_json_file_path = 'pt-train-singlelabel-eng-400.json'
# train_json_file_path = 'test.json'
# validation_json_file_path = 'pt-validation-data.json'
# validation_json_file_path = 'pt-valid-singlelabel-eng-200.json'
validation_json_file_path = 'pt-valid-singlelabel-eng-400.json'

# Assuming your JSON file follows the format shown above, you can load it like this:
tone_dataset = load_dataset('json', data_files={"train": train_json_file_path, "test": validation_json_file_path})

# Get model and tokenized
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id


# Tone Preprocess function
def tone_preprocess_function(examples):
    batch_size = len(examples[tone_text_column])
    inputs = [f"{tone_text_column} : Text: {x}, output : " for x in examples[tone_text_column]]
    targets = [str(x) for x in examples[tone_label_column]]
    model_inputs = tokenizer(inputs)
    labels = tokenizer(targets)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i] + [tokenizer.pad_token_id]
        # print(i, sample_input_ids, label_input_ids)
        model_inputs["input_ids"][i] = sample_input_ids + label_input_ids
        labels["input_ids"][i] = [-100] * len(sample_input_ids) + label_input_ids
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
    # print(model_inputs)
    for i in range(batch_size):
        sample_input_ids = model_inputs["input_ids"][i]
        label_input_ids = labels["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_input_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_input_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        labels["input_ids"][i] = [-100] * (max_input_length - len(sample_input_ids)) + label_input_ids
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_input_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_input_length])
        labels["input_ids"][i] = torch.tensor(labels["input_ids"][i][:max_input_length])
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Apply preprocess to entire dataset
processed_tone_datasets = tone_dataset.map(
    tone_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=tone_dataset["train"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

# Prepare dataset
tone_train_dataset = processed_tone_datasets["train"]
tone_eval_dataset = processed_tone_datasets["test"]


train_dataloader = DataLoader(
    tone_train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)
eval_dataloader = DataLoader(tone_eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True)

# Train model
model = get_peft_model(model, peft_config) # Get PEFT model object
print(model.print_trainable_parameters())
"trainable params: 8192 || all params: 559222784 || trainable%: 0.0014648902430985358"

# Setup an optimizer and learning rate scheduler:
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=(len(train_dataloader) * num_epochs),
)

# Move the model to the GPU, then write a training loop to start training!
# device = "cpu"
# model = model.to(device)
# model = model.to("cuda")
# import torch
torch.cuda.set_device(args.gpu_device)

model = model.cuda()
# # Get the list of available GPUs
# available_gpus = torch.cuda.device_count()

# # Get the current device that the model is using
current_device = model.device

# print("-------------", available_gpus)
print("-------------current_device-----------", current_device)
print("-------------current_device-----------", torch.cuda.current_device())

# Tone
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(tqdm(train_dataloader)):
        batch = {k: v.to(device) for k, v in batch.items()}
        # batch = {k: v.to("cuda") for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    model.eval()
    eval_loss = 0
    eval_preds = []
    for step, batch in enumerate(tqdm(eval_dataloader)):
        # batch = {k: v.to("cuda") for k, v in batch.items()}
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.detach().float()
        eval_preds.extend(
            tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
        )

    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


save_model_path = "model_eng_sl/"

# Save the model
model.save_pretrained(save_model_path)

# Optionally, also save the tokenizer if needed
tokenizer.save_pretrained(save_model_path)

# Save the prompt tuning configuration separately (optional)
# This will allow you to reconstruct the peft_config when loading the model later
# If you prefer, you can also store this information as metadata in the model's config.
peft_config.save_pretrained(save_model_path)

print("Model and tokenizer saved successfully.")



