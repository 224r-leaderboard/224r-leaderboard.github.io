import datasets
import random
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import time
import os

def construct_fewshot_prompt(all_prompts):
    """Create a few-shot prompt with 5 random examples."""
    if len(all_prompts) < 5:
        raise ValueError("At least 5 example prompts are required.")
    
    sampled_prompts = random.sample(all_prompts, 5)
    
    user_prompt = f"""
    # Examples
    ## Example 1
    {sampled_prompts[0]}

    ## Example 2
    {sampled_prompts[1]}

    ## Example 3
    {sampled_prompts[2]}

    ## Example 4
    {sampled_prompts[3]}

    ## Example 5
    {sampled_prompts[4]}
    
    # Task
    Generate a new question that is similar in **topic, structure, and complexity** to the following example questions. Do not include any additional context or information. The new question should be a standalone question that could be used in a similar context as the examples provided. Just provide the new question without any additional text or explanation.

    # New Question:
    """

    return [
        {'role': 'system', 'content': 'You are an expert question generator. Your goal is to create a new question that mirrors the provided examples in terms of its subject matter, grammatical structure, and difficulty level. Ensure the generated question is clear, concise, and directly related to the patterns observed in the examples.'},
        {'role': 'user', 'content': user_prompt}
    ]

def generate_question(args):
    """Generate a single question with error handling and retries."""
    idx, all_prompts, model, client = args
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=construct_fewshot_prompt(all_prompts),
                temperature=0.7,
                max_tokens=256
            )
            return idx, completion.choices[0].message.content
        except Exception as e:
            wait_time = backoff_factor ** attempt
            print(f"Error on attempt {attempt+1}/{max_retries} for idx {idx}: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Failed to generate question after {max_retries} attempts.")
                return idx, None

def main():
    # Configuration parameters
    num_prompts = 200
    model = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'
    output_path = '/Users/anikaitsingh/Desktop/leaderboard/data/ultrafeedback_heldout_prompt.json'
    max_workers = min(32, os.cpu_count() + 4)  # Recommended ThreadPoolExecutor formula
    
    # Load dataset
    print("Loading dataset...")
    ds = datasets.load_dataset('HuggingFaceH4/ultrafeedback_binarized', split='test_prefs')
    all_prompts = ds['prompt']
    
    # Initialize API client
    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key="956db579df40e0fbf52986dfa2b5ff6e9b7873eeabf62b7ce9866a81cacb3bf9"
    )
    
    # Prepare arguments for parallel processing
    args_list = [(i, all_prompts, model, client) for i in range(num_prompts)]
    
    # Generate questions in parallel
    new_questions = [None] * num_prompts
    
    print(f"Generating {num_prompts} questions using {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_question, args) for args in args_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_prompts, desc="Generating questions"):
            idx, content = future.result()
            if content:
                new_questions[idx] = content
    
    # Remove any failed generations
    new_questions = [q for q in new_questions if q is not None]
    
    # Create DataFrame and save results
    result_df = pd.DataFrame({
        'prompt': new_questions,
        'id': [str(i) for i in range(len(new_questions))],
        'model': model,
        'dataset': 'ultrafeedback',
        'split': 'test'
    })
    
    result_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"Successfully generated {len(new_questions)}/{num_prompts} questions and saved to {output_path}")

if __name__ == "__main__":
    main()