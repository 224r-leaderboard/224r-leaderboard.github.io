import os
import json
import time
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Initialize FastAPI app
app = FastAPI(title="Model Evaluation Leaderboard")
security = HTTPBasic()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
LEADERBOARD_FILE = os.path.join(BASE_DIR, "leaderboard.json")
EVAL_SETS = {
    "instruction_following": os.path.join(BASE_DIR, "data/ultrafeedback_hidden_eval.json"),
    "math_reasoning": os.path.join(BASE_DIR, "data/countdown_hidden_eval.json")
}

# Create necessary directories
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

# Initialize leaderboard if it doesn't exist
if not os.path.exists(LEADERBOARD_FILE):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump({
            "instruction_following": [],
            "math_reasoning": []
        }, f)

# Models for validation
class SubmissionRequest(BaseModel):
    group_name: str
    task_type: str  # "instruction_following" or "math_reasoning"
    model_huggingface_path: str

class LeaderboardEntry(BaseModel):
    rank: int
    group_name: str
    score: float
    submission_time: str
    model_huggingface_path: str

# Authentication function
def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    # In a real system, check against a database of authorized users
    # This is a simplified example
    correct_username = "admin"
    correct_password = "admin123"
    
    is_username_correct = credentials.username == correct_username
    is_password_correct = credentials.password == correct_password
    
    if not (is_username_correct and is_password_correct):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    
    return credentials.username

# Function to evaluate a submission
def evaluate_submission(model_path, task_type):
    # Load the evaluation dataset
    eval_data = load_dataset("json", data_files=EVAL_SETS[task_type], split="train")
    
    # Load the model from Hugging Face using vllm
    try:
        # Configure vllm with appropriate settings
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,  # Set based on available GPUs
            gpu_memory_utilization=0.8,
            trust_remote_code=True,
            dtype="half"  # Using half precision for efficiency
        )
        
        # Define sampling parameters
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512
        )
        
        # Process in batches for better efficiency
        batch_size = 8  # Adjust based on available memory
        scores = []
        
        for i in range(0, len(eval_data), batch_size):
            batch = eval_data[i:i+batch_size]
            
            # Prepare batch inputs
            if task_type == "instruction_following":
                batch_inputs = [example["instruction"] for example in batch]
                batch_references = [example["reference"] for example in batch]
            else:  # math_reasoning
                batch_inputs = [example["problem"] for example in batch]
                batch_references = [example["solution"] for example in batch]
            
            # Generate outputs for the batch
            batch_outputs = llm.generate(batch_inputs, sampling_params)
            batch_predictions = [output.outputs[0].text for output in batch_outputs]
            
            # Calculate scores for the batch
            batch_scores = []
            for j, (prediction, reference) in enumerate(zip(batch_predictions, batch_references)):
                if task_type == "instruction_following":
                    score = calculate_instruction_score(prediction, reference)
                else:  # math_reasoning
                    score = calculate_math_score(prediction, reference)
                batch_scores.append(score)
            
            scores.extend(batch_scores)
            
        # Calculate final score
        if not scores:
            return {"error": "No valid scores were calculated"}
        
        final_score = sum(scores) / len(scores)
        return {"score": final_score}
    
    except Exception as e:
        return {"error": f"Error executing submission: {str(e)}"}

# Simplified scoring functions
def calculate_instruction_score(prediction, reference):
    # This would be replaced with a more sophisticated evaluation
    # For example, using GPT-4 to rate responses or comparing with reference responses
    # For simplicity, let's use a dummy score based on length match
    return max(0, 1 - abs(len(prediction) - len(reference)) / max(len(reference), 1))

def calculate_math_score(prediction, reference):
    # This would be replaced with actual math problem evaluation
    # For simplicity, let's use a dummy exact match score
    return 1.0 if prediction.strip().lower() == reference.strip().lower() else 0.0

# Function to update the leaderboard
def update_leaderboard(group_name, task_type, score, model_huggingface_path):
    with open(LEADERBOARD_FILE, "r") as f:
        leaderboard = json.load(f)
    
    # Add new submission
    submission = {
        "group_name": group_name,
        "score": score,
        "submission_time": datetime.now().isoformat(),
        "model_huggingface_path": model_huggingface_path
    }
    
    leaderboard[task_type].append(submission)
    
    # Sort by score (descending)
    leaderboard[task_type].sort(key=lambda x: x["score"], reverse=True)
    
    # Update ranks
    for i, entry in enumerate(leaderboard[task_type]):
        entry["rank"] = i + 1
    
    # Save updated leaderboard
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(leaderboard, f, indent=2)
    
    return leaderboard[task_type]

# API endpoints
@app.post("/submit")
async def submit_model(
    background_tasks: BackgroundTasks,
    submission: SubmissionRequest
):
    # Validate task type
    if submission.task_type not in ["instruction_following", "math_reasoning"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task type. Must be 'instruction_following' or 'math_reasoning'"
        )
    
    # Generate a unique ID for this submission
    timestamp = int(time.time())
    submission_id = f"{submission.group_name.replace(' ', '_')}_{submission.task_type}_{timestamp}"
    
    # Start evaluation in background
    background_tasks.add_task(
        process_submission,
        submission_id,
        submission.task_type,
        submission.group_name,
        submission.model_huggingface_path
    )
    
    return {
        "status": "success",
        "message": "Submission received and being processed",
        "submission_id": submission_id
    }

# Function to process submission in background
def process_submission(submission_id, task_type, group_name, model_huggingface_path):
    # Evaluate the submission
    result = evaluate_submission(model_huggingface_path, task_type)
    
    if "error" in result:
        # Log the error
        with open(f"{SUBMISSIONS_DIR}/{submission_id}.error", "w") as f:
            f.write(result["error"])
        return
    
    # Update the leaderboard
    update_leaderboard(group_name, task_type, result["score"], model_huggingface_path)
    
    # Save metadata
    with open(f"{SUBMISSIONS_DIR}/{submission_id}.meta", "w") as f:
        json.dump({
            "group_name": group_name,
            "task_type": task_type,
            "score": result["score"],
            "submission_time": datetime.now().isoformat(),
            "model_huggingface_path": model_huggingface_path
        }, f, indent=2)

@app.get("/leaderboard/{task_type}")
async def get_leaderboard(task_type: str):
    if task_type not in ["instruction_following", "math_reasoning"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task type. Must be 'instruction_following' or 'math_reasoning'"
        )
    
    with open(LEADERBOARD_FILE, "r") as f:
        leaderboard = json.load(f)
    
    return leaderboard[task_type]

@app.get("/submission_status/{submission_id}")
async def get_submission_status(submission_id: str):
    submission_path = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.py")
    meta_path = f"{submission_path}.meta"
    error_path = f"{submission_path}.error"
    
    if not os.path.exists(submission_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found"
        )
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
        return {
            "status": "completed",
            "submission_id": submission_id,
            "score": meta["score"],
            "submission_time": meta["submission_time"]
        }
    elif os.path.exists(error_path):
        with open(error_path, "r") as f:
            error = f.read()
        return {
            "status": "error",
            "submission_id": submission_id,
            "error": error
        }
    else:
        return {
            "status": "processing",
            "submission_id": submission_id
        }

@app.get("/thresholds")
async def get_thresholds():
    # Return the current minimum thresholds for basic and extension implementations
    return {
        "instruction_following": {
            "basic_implementation": 0.65,
            "extension": 0.75
        },
        "math_reasoning": {
            "basic_implementation": 0.60,
            "extension": 0.70
        }
    }

@app.get("/guidelines")
async def get_guidelines():
    # Return submission guidelines
    if False:
        return HTMLResponse("""
    <h2>Model Requirements</h2>
    <ul>
        <li>All submissions must use the Qwen 2.5 0.5B (non-instruct) model as the base model</li>
    </ul>
    <h2>Evaluation Criteria</h2>
    <h3>Instruction Following</h3>
    <p>Based on the Ultrafeedback benchmark, evaluating:</p>
    <ul>
        <li>Helpfulness: How well the model follows the given instruction</li>
        <li>Harmlessness: Ensuring the model doesn't produce harmful content</li>
    </ul>
    <h3>Math Reasoning</h3>
    <p>Based on the Countdown benchmark, evaluating:</p>
    <ul>
        <li>Correctness: Whether the final answer matches the reference</li>
    </ul>
    <h2>Submission Limits</h2>
    <p>Each student may submit up to 5 times in total for each task type.</p>
    """)
    else:
        return HTMLResponse("""
<h3>Submission Guidelines</h3>

<h4>Stylistic Personalization</h4>
<p>Models should adapt to individual writing styles and communication preferences.</p>
<ul>
    <li>Focus on tone, formality, and linguistic patterns</li>
    <li>Should maintain consistency with user's preferred style</li>
    <li>Example: Adapting to formal vs. casual communication</li>
</ul>

<h4>Recommendation Personalization</h4>
<p>Models should provide personalized recommendations based on user preferences.</p>
<ul>
    <li>Focus on understanding user preferences based on their history of past interactions</li>
    <li>Should provide relevant and diverse recommendations</li>
    <li>Evaluation based on relevance and personalization</li>
    <li>Example: What should I do this weekend in San Francisco? Go to Cherry Blossom Festival, take a walk in Golden Gate Park, or bar-hopping in the Mission District.</li>
</ul>

<h4>Value Personalization</h4>
<p>Models should align with user's values and ethical preferences.</p>
<ul>
    <li>Focus on ethical alignment and value consistency</li>
    <li>Should respect user's moral and ethical boundaries</li>
    <li>Example: What is your opinion on the death penalty?</li>
</ul>

<h4>Concierge Personalization</h4>
<p>Models should provide personalized assistance and service.</p>
<ul>
    <li>Asking clarifying questions to understand the user's needs and preferences</li>
    <li>Necessary in underspecified tasks, providing helpful and relevant information</li>
    <li>Evaluation based on overall helpfulness and accuracy of the response as well as whether there are too many or too few questions asked</li>
</ul>

<h4>Submission Process</h4>
<ol>
    <li>Choose the appropriate personalization category</li>
    <li>Provide your group name and model path</li>
    <li>Submit the model for evaluation</li>
    <li>Receive a submission ID for tracking</li>
    <li>Check status using the submission ID</li>
</ol>
    """)

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)