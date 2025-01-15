import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import os
from datasets import load_dataset
import evaluate
import random

# Print transformers version
print(transformers.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print whether GPU is being used or not
if torch.cuda.is_available():
    print("Using GPU:", torch.cuda.get_device_name(0))  
else:
    print("Using CPU")

# Load dataset
dataset_dir = "./dataset"
os.makedirs(dataset_dir, exist_ok=True)

try:
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir=dataset_dir)
    print('Dataset download successful')
   # print(ds)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Load model
weights_dir = "./weights"
os.makedirs(weights_dir, exist_ok=True)

try:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir=weights_dir)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir=weights_dir).to(device)
     # If multiple GPUs are available, wrap the model in DataParallel
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print('utilizing 2 gpu')
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()


#printing the 1st dataset
#print(ds["train"][0])
#print()

# defining every parts of data
train_data = ds['train']
validation_data = ds['validation']
test_data = ds['test']

articles = train_data['article']
highlights = train_data['highlights']
ids = train_data['id']

# Display a few samples to check
print("First article:", articles[0])
print()
print("First highlight:", highlights[0])
print()
print("First id:", ids[0])
print()


# function for summary generation
def generate_summary(train_input, test_input):

    # Combine the few-shot training examples with the test input
    few_shot_prompt = f"""
    The following examples show how to summarize a text into a brief and informative summary:
    {train_input}
    Based on these examples, now summarize the following text into a concise and informative summary, using the same words and avoiding paraphrasing:
    Text:
    {test_input}

    """

    #print('prompt is: ', few_shot_prompt)

    # Tokenize input text
    inputs = tokenizer(few_shot_prompt, return_tensors="pt").to(device)
    pad_token_id = tokenizer.eos_token_id

    ###############################################################################
    # Generate the summary with controlled parameters
    summary_ids = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        num_beams=4,
        num_return_sequences=1,
        early_stopping=True,
        pad_token_id=pad_token_id
    )

    full_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if 'Summary:' in full_output:
        summary = full_output.split('Summary:')[-1].strip()
    else:
        summary = full_output

    return summary


# rouge score computation
rouge = evaluate.load('rouge')
rouge_1_scores = []
rouge_2_scores = []
rouge_L_scores = []
def compute_rouge_updated(test_input, summary_model):
    # Compute ROUGE scores
    results = rouge.compute(predictions=[summary_model], references=[test_input])

    # Extract ROUGE scores
    rouge_1 = results["rouge1"]
    rouge_2 = results["rouge2"]
    rouge_L = results["rougeL"]

    # Append the scores to the lists for later plotting
    rouge_1_scores.append(rouge_1)
    rouge_2_scores.append(rouge_2)
    rouge_L_scores.append(rouge_L)

    return {
        "ROUGE-1": rouge_1,
        "ROUGE-2": rouge_2,
        "ROUGE-L": rouge_L,
    }

############################################################################
#### main calling part ######
epochs = 10
for epoch in range(1, epochs + 1):
    # Create a list to store randomly selected examples

    print('Epoch:', epoch)
    print('--------------------')


    # Randomly select `epoch` number of examples for this epoch
    random_indices = random.sample(range(len(articles)), epoch)

    print('randomly selected indexes are', random_indices)
    train_input = ""
    example_number = 1
    # Construct the train_input string with the selected examples
    for i in random_indices:
        train_input += f"Example {example_number}:\nText: \"{articles[i]}\"\nSummary: \"{highlights[i]}\"\n\n"
        example_number += 1

    # Extract the first article from the test data for summarization
    test_input = articles[4]

    print('---------Summary is generating----------')
    # Call the summarization function with the selected train data and test input
    summary = generate_summary(train_input, test_input)
    print(summary)


    print('---------calculating rouge----------')
    # Compute ROUGE scores for this epoch and store them in the lists
    rouge_s = compute_rouge_updated(test_input, summary)
    print('rouge score is', rouge_s)






