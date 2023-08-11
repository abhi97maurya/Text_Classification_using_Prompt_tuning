from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_config, get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType, PeftType
import torch
from datasets import load_dataset
import json, random
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate
import argparse

# metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
# metrics1 = evaluate.combine([
#     evaluate.load("recall", average="micro"),
#     evaluate.load("f1", average="micro"),
#     evaluate.load("precision", average="micro")
# ])

def get_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_device", type=int, default=0, help="The Device size")
    parser.add_argument("--batch_size", type=int, default=8, help="The Batch size")

    return parser.parse_args()


# Get the arguments
args = get_arguments()
# Print the arguments
print(f"The GPU device is: {args.gpu_device}")
print(f"The Batch size is: {args.batch_size}")

batch_size = args.batch_size

tone_text_column = "input"
tone_label_column = "output"
max_input_length = 256
# batch_size = 8
pred_set = []
pred_set_num = []
gold_tone_num = []

output_dict = {
    "excited" : 1,
    "sad" : 2,
    "polite": 3,
    "impolite" : 4,
    "satisfied" : 5,
    "frustrated" : 6,
    "sympathetic" : 7,
    "none" : 8

}
device = "cuda"
# device = "cpu"
text_column = "input"

# Define model and tokenizer
model_name_or_path = "google/flan-t5-large"
tokenizer_name_or_path = "google/flan-t5-large"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

# Get PEFT MODEL FROM TRAINED MODEL
peft_model_id = "model_eng_sl"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, peft_model_id)

torch.cuda.set_device(args.gpu_device)

model = model.cuda()
current_device = model.device
print("-------------current_device-----------", current_device)
print("-------------current_device-----------", torch.cuda.current_device())

test_dataset = None
# Calculate accuracy
test_dataset_file = "pt-test-singlelabel-eng-all.json"
# test_dataset_file = "pt-test-singlelabel-eng-8.json"
with open(test_dataset_file, 'r') as file:
    test_dataset = json.loads(file.read())  

# Create dataloader for test
tone_dataset = load_dataset('json', data_files={"test": test_dataset_file})

# #==============================
# test_dataset = test_dataset[:16]
# #==============================


def tone_test_preprocess_function(examples):
    batch_size = len(examples[tone_text_column])
    inputs = [f"{tone_text_column} : Text: {x}, output : " for x in examples[tone_text_column]]
    model_inputs = tokenizer(inputs)
    
    # Add attention mask and padding
    for i in range(batch_size):
        model_inputs["attention_mask"][i] = [1] * len(model_inputs["input_ids"][i])
        sample_input_ids = model_inputs["input_ids"][i]
        model_inputs["input_ids"][i] = [tokenizer.pad_token_id] * (
            max_input_length - len(sample_input_ids)
        ) + sample_input_ids
        model_inputs["attention_mask"][i] = [0] * (max_input_length - len(sample_input_ids)) + model_inputs[
            "attention_mask"
        ][i]
        model_inputs["input_ids"][i] = torch.tensor(model_inputs["input_ids"][i][:max_input_length])
        model_inputs["attention_mask"][i] = torch.tensor(model_inputs["attention_mask"][i][:max_input_length])
    return model_inputs

# Apply preprocess to entire dataset
processed_tone_datasets = tone_dataset.map(
    tone_test_preprocess_function,
    batched=True,
    num_proc=1,
    remove_columns=tone_dataset["test"].column_names,
    load_from_cache_file=False,
    desc="Running tokenizer on dataset",
)

test_dataloader = DataLoader(
    processed_tone_datasets["test"], shuffle=False, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
)

for step, batch in enumerate(tqdm(test_dataloader)):
    print(f"Running Batch {step}...")
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model.generate(**batch, max_new_tokens=128, eos_token_id=3)

    # List of 7 outputs
    output_text_list = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)
    for output in output_text_list:
        predicted_tone = output.split(" ")[0]
        pred_set.append(predicted_tone)
        predicted_tone_num = random.randint(100, 1000)
        if predicted_tone in output_dict:
            predicted_tone_num =  output_dict[predicted_tone] 
        pred_set_num.append(predicted_tone_num)

# Get gold set
def extract_gold_set(dataset):
    """Obtains the gold set predictions for evaluation.
    Args:
        dataset:  list(dict)
            Each item in the list is a dictionary where the key "labels" holds the classes
            and the key "text" holds the corresponding text
    Returns:
        list(list(tuple))
            A list of lists of "gold-set" classes which are in the form of
            tuples that look like: (text, class)
            E.g.: [[('Hello', 'Greeting')], ...]
    """
    
    gold_data = []
    for example in dataset:
        gold_data.append(example["output"])
        gold_tone_num.append(output_dict[example["output"]])

    return gold_data

gold_set = extract_gold_set(dataset=test_dataset) 

# print(gold_set, pred_set)
# metrics1.add_batch(references=gold_tone_num, predictions=pred_set_num)
# print(metrics1.compute(average='micro')) 
# print(metrics.compute(average='micro')) 

f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")

results = {}
results.update(f1_metric.compute(predictions=pred_set_num, references = gold_tone_num, average="micro"))
results.update(recall_metric.compute(predictions=pred_set_num, references = gold_tone_num, average="micro"))
results.update(precision_metric.compute(predictions=pred_set_num, references = gold_tone_num, average="micro"))
print(results)
with open("result.txt", "w+") as file:
    file.write(json.dumps(results))
    file.write("\n")
    file.write(json.dumps(gold_set))
    file.write("\n")
    file.write(json.dumps(pred_set))