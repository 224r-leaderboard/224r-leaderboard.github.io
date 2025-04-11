import datasets
import random
from openai import OpenAI
import pandas as pd
from tqdm import tqdm
import concurrent.futures
import time
import os
import pandas as pd

def generate_responses(args):
    """Generate a single question with error handling and retries."""
    idx, prompt, model, client = args
    max_retries = 3
    backoff_factor = 2
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
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
    model = 'Qwen/Qwen2.5-7B-Instruct-Turbo'
    input_path = '/Users/anikaitsingh/Desktop/leaderboard/data/ultrafeedback_heldout_prompts.json'
    output_path = '/Users/anikaitsingh/Desktop/leaderboard/data/ultrafeedback_heldout_prompt_responses.json'
    max_workers = min(32, os.cpu_count() + 4)  # Recommended ThreadPoolExecutor formula
    
    # Load dataset
    print("Loading dataset...")
    ds = pd.read_json(input_path, orient='records', lines=True)
    all_prompts = ds['prompt']
    
    # Initialize API client
    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key="956db579df40e0fbf52986dfa2b5ff6e9b7873eeabf62b7ce9866a81cacb3bf9"
    )
    
    # Prepare arguments for parallel processing
    args_list = [(i, all_prompts[i], model, client) for i in range(len(all_prompts))]
    
    # Generate responses in parallel
    responses = [None] * len(all_prompts)
    print(f"Generating {len(all_prompts)} responses using {max_workers} workers...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_responses, args) for args in args_list]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(all_prompts), desc="Generating questions"):
            idx, content = future.result()
            if content:
                responses[idx] = content
    
    # Create DataFrame and save results
    result_df = pd.DataFrame({
        'prompt': all_prompts,
        'response': responses,
        'id': [str(i) for i in range(len(all_prompts))],
        'model': model,
        'dataset': 'ultrafeedback',
        'split': 'test'
    })
    
    result_df.to_json(output_path, orient='records', lines=True, force_ascii=False)
    print(f"Successfully generated {len(all_prompts)} responses and saved to {output_path}")

if __name__ == "__main__":
    main()