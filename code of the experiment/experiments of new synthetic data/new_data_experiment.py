import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
from rouge_score import rouge_scorer
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# salloc -A demelo -p sorcery --gpus=1 --mem=80G 
#salloc -A demelo -p sorcery --gres=gpu:a100:1 --mem=80G (gx01)
# du -sh . how much memory do u utilize
#cp -r /hpi/fs00/home/afsana.mimi/llama_project/memory_eff.py /hpi/fs00/scratch/afsana.mimi/
#rm -rf weights
# scontrol show job shows gpu details
#scancel -u afsana.mimi camcel all jobs
#env actiavte source /hpi/fs00/home/afsana.mimi/llama_project/icl/bin/activate
# nohup python your_script.py &   for running the script in the backgroud
#nohup python more_test_example.py | tee output.log &
#python llama_project/more_test_example.py 

 



# Set the memory management configuration for PyTorch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/hpi/fs00/home/afsana.mimi/llama_project/weights'
#export TRANSFORMERS_CACHE="/hpi/fs00/home/afsana.mimi/llama_project/weights"



# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir="/hpi/fs00/home/afsana.mimi/llama_project/weights", local_files_only=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct", cache_dir="/hpi/fs00/home/afsana.mimi/llama_project/weights", local_files_only=True).to(device)


# reading from cache
#tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
#model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct").to(device)
print(' -------------------')



# Function to generate summaries
def generate_summary(train_input, test_input):
    prompt = f"""The following examples show how to summarize a text into a summary:
    {train_input}
    
    
    Based on these examples, now summarize only the following text into a summary, use the same words and avoid paraphrasing:
    ---
    <text to summarize>
    Text: {test_input}
    ---
    Summary: 
    """

    print(f"Prompt for text summarization:\n{prompt}")
    
    # Tokenize input
    #inputs = tokenizer(prompt, return_tensors="pt",  truncation=True, padding= False,  max_length=5500).to(device)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    attention_mask = inputs["attention_mask"]
    # Track token length
    input_length = inputs['input_ids'].shape[1]  # Number of tokens in the input
    print(f"Token Length: {input_length}")

    pad_token_id = tokenizer.eos_token_id
    
    # Use mixed precision to reduce memory usage
    with torch.no_grad():
        with autocast():
            summary_ids = model.generate(
                inputs["input_ids"],
                attention_mask=attention_mask,
                max_new_tokens=400,  # Reduced from 200 (chnage)
                num_beams=4,     # (chnage)
                temperature=0.7,
                top_p=0.7, 
                num_return_sequences=1,
                early_stopping=True,
                pad_token_id=pad_token_id
            )
    # Decode summary
    full_output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Decode summary
    if 'Summary:' in full_output:
        summary = full_output.split('Summary:')[-1].strip()
    else:
        summary = full_output

    return summary 

# Load dataset
ds_train =  pd.read_csv("/hpi/fs00/home/afsana.mimi/llama_project/synthetic_data/data1.csv")
articles_train = ds_train['article']
highlights_train = ds_train['highlight'] # in the demo data its writeen as highlight

ds = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="/hpi/fs00/home/afsana.mimi/llama_project/dataset")
articles = ds['test']['article']
highlights = ds['test']['highlights']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def manage_memory():
    print(f"Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    torch.cuda.empty_cache()
    print(f"Memory after clearing cache: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")


# Main loop
# Create the folder for saving plots if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))

plots_dir = os.path.join(script_dir, 'experiment1_plots')
os.makedirs(plots_dir, exist_ok=True)

# Initialize lists to store precision scores for each epoch
rouge1_precision_scores = []
rouge2_precision_scores = []
rougeL_precision_scores = []

# Initialize lists to store average precision scores for each epoch
avg_rouge1_precision = []
avg_rouge2_precision = []
avg_rougeL_precision = []
# -----------------------------------------
epochs = 10   ###chnaged from 10
for epoch in range(1, epochs + 1):
    print()
    print(f"Epoch {epoch}")
    print('----------------------------')
    
    # Increase in-context examples with each epoch
    random_indices = random.sample(range(len(articles_train)), epoch)
    print('randomly selected indexes are', random_indices)
    print()
    train_input = ""
    example_number = 1
    for idx, i in enumerate(random_indices):
        train_input += f"Example {example_number}:\nText: \"{articles_train[i]}\"\nSummary: \"{highlights_train[i]}\"\n\n"
        example_number += 1
    

    temp_rouge1_precision = []
    temp_rouge2_precision = []
    temp_rougeL_precision = []
     # Generate summaries for the first 50 test inputs
    for i in range(300):  # Iterate through the first 50 examples (chnaged from 50)
        test_input = articles[i]
        reference_summary = highlights[i]
        
        print(f"------------Summary {i+1} is generating by model-------------")
        summary = generate_summary(train_input, test_input)
        print(f"Generated Summary {i+1}:", summary)
        
        # Calculate ROUGE score for the summary
        print(' -------------------')
        scores = scorer.score(reference_summary, summary)
        print(f"ROUGE scores for Summary {i+1}: {scores}")

        # Extract precision scores
        temp_rouge1_precision.append(scores['rouge1'].precision)
        temp_rouge2_precision.append(scores['rouge2'].precision)
        temp_rougeL_precision.append(scores['rougeL'].precision)

    # Compute average precision scores for this epoch
    avg_rouge1_precision.append(sum(temp_rouge1_precision) / len(temp_rouge1_precision))
    avg_rouge2_precision.append(sum(temp_rouge2_precision) / len(temp_rouge2_precision))
    avg_rougeL_precision.append(sum(temp_rougeL_precision) / len(temp_rougeL_precision))


    # Store the precision scores for this epoch
    rouge1_precision_scores.append(temp_rouge1_precision)
    rouge2_precision_scores.append(temp_rouge2_precision)
    rougeL_precision_scores.append(temp_rougeL_precision)

    

    # Create a range starting from 1 for the x-axis
    x_values = np.arange(1, len(temp_rouge1_precision) + 1) 

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, temp_rouge1_precision, label='ROUGE-1 Precision')
    plt.plot(x_values, temp_rouge2_precision, label='ROUGE-2 Precision')
    plt.plot(x_values, temp_rougeL_precision, label='ROUGE-L Precision')
    plt.xlabel('Test Example')
    plt.ylabel('Rourge_Precision')
    plt.title(f'Precision Scores for 300 Test Examples with {epoch} In-Context Example(s)')
    plt.legend()
    

    # Set x-ticks to show every 2 test examples (from 1 to 50)
    plt.xticks(np.arange(1, len(temp_rouge1_precision) + 1, 29)) 
    # Save the plot
    plt.savefig(os.path.join(plots_dir, f'epoch_{epoch}_precision_scores.png'))
    plt.close()


    # Print GPU memory usage after each epoch
    print(f"Epoch {epoch} GPU Memory Usage:")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    
    # Manage GPU memory
    manage_memory()

# Plot the average precision scores after all epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), avg_rouge1_precision, label='Avg ROUGE-1 Precision', marker='o')
plt.plot(range(1, epochs + 1), avg_rouge2_precision, label='Avg ROUGE-2 Precision', marker='o')
plt.plot(range(1, epochs + 1), avg_rougeL_precision, label='Avg ROUGE-L Precision', marker='o')
plt.xlabel('No of In Context example')
plt.ylabel('Average Rouge Score')
plt.title('Average ROUGE Precision Scores Based on Number of In-Context Examples')
plt.legend()
# set xticks here   --------!!!!!!!!!!!!!!!!!!
plt.xticks(np.arange(1, epochs + 1, 1)) 


# Save the final plot
plt.savefig(os.path.join(plots_dir, 'average rouge_scores vs the number of examples.png'))

plt.close()
