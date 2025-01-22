
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import random
from rouge_score import rouge_scorer
from torch.cuda.amp import autocast

# salloc -A demelo -p sorcery --gpus=1 --mem=80G 
#salloc -A demelo -p sorcery --gres=gpu:a100:1 --mem=80G (gx01)
# du -sh . how much memory do u utilize
#cp -r /hpi/fs00/home/afsana.mimi/llama_project/memory_eff.py /hpi/fs00/scratch/afsana.mimi/
#rm -rf weights
# scontrol show job shows gpu details
#scancel -u afsana.mimi camcel all jobs
#env actiavte source /hpi/fs00/home/afsana.mimi/llama_project/icl/bin/activate

 



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
print('checking -------------------')



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
                num_beams=5,     # (chnage)
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
ds = load_dataset("abisee/cnn_dailymail", "3.0.0", cache_dir="/hpi/fs00/home/afsana.mimi/llama_project/dataset")
articles = ds['train']['article']
highlights = ds['train']['highlights']

# ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def manage_memory():
    print(f"Memory before operation: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    torch.cuda.empty_cache()
    print(f"Memory after clearing cache: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")


# Main loop
epochs = 10
for epoch in range(1, epochs + 1):
    print()
    print(f"Epoch {epoch}")
    print('----------------------------')
    
    # Increase in-context examples with each epoch
    random_indices = random.sample(range(len(articles)), epoch)
    print('randomly selected indexes are', random_indices)
    print()
    train_input = ""
    example_number = 1
    for idx, i in enumerate(random_indices):
        train_input += f"Example {example_number}:\nText: \"{articles[i]}\"\nSummary: \"{highlights[i]}\"\n\n"
        example_number += 1
    # Use a fixed test input
    test_input = articles[0]
    reference_summary = highlights[0]
    
    # Generate summary
    summary = generate_summary(train_input, test_input)
    print("------------summary is generating by model-------------")
    print("Generated Summary:", summary)
    
    
    # Calculate ROUGE score
    scores = scorer.score(reference_summary, summary)
    print(f"ROUGE scores: {scores}")

    # Print GPU memory usage after each epoch
    print(f"Epoch {epoch} GPU Memory Usage:")
    print(f"Allocated Memory: {torch.cuda.memory_allocated(0) / (1024 ** 3):.2f} GB")
    print(f"Reserved Memory: {torch.cuda.memory_reserved(0) / (1024 ** 3):.2f} GB")
    

     # Clear GPU cache
    #torch.cuda.empty_cache()

     # Manage GPU memory
    manage_memory()

