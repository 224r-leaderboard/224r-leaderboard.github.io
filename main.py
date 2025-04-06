import os
import json
import time
import hashlib
import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Form, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Union
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

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

# Load base model (Qwen 2.5 0.5B non-instruct)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

# Models for validation
class SubmissionRequest(BaseModel):
    student_id: str
    task_type: str  # "instruction_following" or "math_reasoning"
    submission_code: str
    description: str
    hyperparameters: Dict[str, Union[str, int, float, bool, None]]

class LeaderboardEntry(BaseModel):
    rank: int
    student_id: str
    score: float
    submission_time: str
    description: str

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
def evaluate_submission(submission_path, task_type):
    # Load the submission code
    with open(submission_path, "r") as f:
        submission_code = f.read()
    
    # Load the evaluation dataset
    eval_data = load_dataset("json", data_files=EVAL_SETS[task_type], split="train")
    
    # Load the base model
    tokenizer, base_model = load_base_model()
    
    # Execute the submission code in a safe environment
    # This is a simplified example - in practice, you would need more robust sandboxing
    try:
        # Create a namespace for execution
        namespace = {
            "base_model": base_model,
            "tokenizer": tokenizer,
            "torch": torch,
            "np": np
        }
        
        # Execute the submission code
        exec(submission_code, namespace)
        
        # The code should define a 'run_model' function that takes an input and returns a prediction
        if "run_model" not in namespace:
            return {"error": "Submission does not define a 'run_model' function"}
        
        run_model = namespace["run_model"]
        
        # Evaluate on each example
        scores = []
        for example in eval_data:
            try:
                if task_type == "instruction_following":
                    # For instruction following, evaluate based on Ultrafeedback metrics
                    input_text = example["instruction"]
                    prediction = run_model(input_text)
                    reference = example["reference"]
                    
                    # Calculate scores based on helpfulness, honesty, harmlessness criteria
                    # This is simplified - in practice, you would use a more sophisticated evaluation
                    score = calculate_instruction_score(prediction, reference)
                    
                elif task_type == "math_reasoning":
                    # For math reasoning, evaluate based on Countdown metrics
                    input_text = example["problem"]
                    prediction = run_model(input_text)
                    reference = example["solution"]
                    
                    # Calculate scores based on correctness and reasoning
                    score = calculate_math_score(prediction, reference)
                
                scores.append(score)
            except Exception as e:
                return {"error": f"Error evaluating example: {str(e)}"}
        
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
def update_leaderboard(student_id, task_type, score, description):
    with open(LEADERBOARD_FILE, "r") as f:
        leaderboard = json.load(f)
    
    # Add new submission
    submission = {
        "student_id": student_id,
        "score": score,
        "submission_time": datetime.now().isoformat(),
        "description": description
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
    submission_id = f"{submission.student_id}_{submission.task_type}_{timestamp}"
    
    # Save submission to disk
    submission_path = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.py")
    with open(submission_path, "w") as f:
        f.write(submission.submission_code)
    
    # Start evaluation in background
    background_tasks.add_task(
        process_submission,
        submission_path,
        submission.task_type,
        submission.student_id,
        submission.description,
        submission.hyperparameters
    )
    
    return {
        "status": "success",
        "message": "Submission received and being processed",
        "submission_id": submission_id
    }

# Function to process submission in background
def process_submission(submission_path, task_type, student_id, description, hyperparameters):
    # Evaluate the submission
    result = evaluate_submission(submission_path, task_type)
    
    if "error" in result:
        # Log the error
        with open(f"{submission_path}.error", "w") as f:
            f.write(result["error"])
        return
    
    # Update the leaderboard
    update_leaderboard(student_id, task_type, result["score"], description)
    
    # Save metadata
    with open(f"{submission_path}.meta", "w") as f:
        json.dump({
            "student_id": student_id,
            "task_type": task_type,
            "score": result["score"],
            "submission_time": datetime.now().isoformat(),
            "description": description,
            "hyperparameters": hyperparameters
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
    return HTMLResponse("""
    <h1>Leaderboard Submission Guidelines</h1>
    <h2>Model Requirements</h2>
    <ul>
        <li>All submissions must use the Qwen 2.5 0.5B (non-instruct) model as the base model</li>
        <li>Your code must define a function named 'run_model' that takes an input string and returns the model's prediction</li>
        <li>You may modify this base model through extensions, prompt engineering, or other techniques</li>
    </ul>
    <h2>Evaluation Criteria</h2>
    <h3>Instruction Following</h3>
    <p>Based on the Ultrafeedback benchmark, evaluating:</p>
    <ul>
        <li>Helpfulness: How well the model follows the given instruction</li>
        <li>Honesty: The factual accuracy of the model's response</li>
        <li>Harmlessness: Ensuring the model doesn't produce harmful content</li>
    </ul>
    <h3>Math Reasoning</h3>
    <p>Based on the Countdown benchmark, evaluating:</p>
    <ul>
        <li>Correctness: Whether the final answer matches the reference</li>
        <li>Reasoning: The quality of the step-by-step reasoning</li>
    </ul>
    <h2>Submission Limits</h2>
    <p>Each student may submit up to 5 times per day and 20 times in total for each task type.</p>
    """)

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)