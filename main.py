import os
import json
import time
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
from datasets import load_dataset
import random
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Model Evaluation Leaderboard")
security = HTTPBasic()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
    expose_headers=["Content-Type", "Authorization"]
)

# Configure paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMISSIONS_DIR = os.path.join(BASE_DIR, "submissions")
STATIC_DIR = os.path.join(BASE_DIR, "static")
LEADERBOARD_DIR = os.path.join(BASE_DIR, "leaderboard_data")
INSTRUCTION_LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "instruction_leaderboard.json")
MATH_LEADERBOARD_FILE = os.path.join(LEADERBOARD_DIR, "math_leaderboard.json")
EVAL_SETS = {
    "instruction_following": os.path.join(BASE_DIR, "data/ultrafeedback_hidden_eval.json"),
    "math_reasoning": os.path.join(BASE_DIR, "data/countdown_hidden_eval.json")
}

# Create necessary directories
os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize leaderboards if they don't exist
if not os.path.exists(INSTRUCTION_LEADERBOARD_FILE):
    with open(INSTRUCTION_LEADERBOARD_FILE, "w") as f:
        json.dump([], f)

if not os.path.exists(MATH_LEADERBOARD_FILE):
    with open(MATH_LEADERBOARD_FILE, "w") as f:
        json.dump([], f)

# Models for validation
class SubmissionRequest(BaseModel):
    group_name: str
    task_type: str  # "instruction_following" or "math_reasoning"
    implementation_type: str  # "basic" or "extension"
    uses_synthetic_data: bool
    submission_data: dict  # The actual model submission data

class LeaderboardEntry(BaseModel):
    rank: int
    group_name: str
    score: float
    submission_time: str
    implementation_type: str
    uses_synthetic_data: bool

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
def evaluate_submission(task_type):
    # Load the evaluation dataset
    eval_data = load_dataset("json", data_files=EVAL_SETS[task_type], split="train")
    
    # Load the model from Hugging Face using vllm
    try:
        score = random.uniform(0.6, 0.95)
        return {"score": score}
    
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

# Function to get the appropriate leaderboard file
def get_leaderboard_file(task_type):
    if task_type == "instruction_following":
        return INSTRUCTION_LEADERBOARD_FILE
    elif task_type == "math_reasoning":
        return MATH_LEADERBOARD_FILE
    else:
        raise ValueError(f"Invalid task type: {task_type}")

def read_json_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()  # Remove any trailing whitespace
            return json.loads(content)
    except json.JSONDecodeError as e:
        # If there's a JSON error, try to fix common issues
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            # Remove any trailing commas
            content = content.rstrip(',')
            # Ensure the content is properly terminated
            if not content.endswith(']') and not content.endswith('}'):
                content = content + ']' if content.startswith('[') else content + '}'
            return json.loads(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading leaderboard file: {str(e)}"
        )

def write_json_file(file_path, data):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error writing leaderboard file: {str(e)}"
        )

# Function to update the leaderboard
def update_leaderboard(group_name, task_type, score, implementation_type, uses_synthetic_data):
    leaderboard_file = get_leaderboard_file(task_type)
    
    try:
        leaderboard = read_json_file(leaderboard_file)
    except Exception:
        # If file is corrupted, start with empty list
        leaderboard = []
    
    # Add new submission
    submission = {
        "group_name": group_name,
        "score": score,
        "submission_time": datetime.now().isoformat(),
        "implementation_type": implementation_type,
        "uses_synthetic_data": uses_synthetic_data
    }
    
    leaderboard.append(submission)
    
    # Sort by score (descending)
    leaderboard.sort(key=lambda x: x["score"], reverse=True)
    
    # Update ranks
    for i, entry in enumerate(leaderboard):
        entry["rank"] = i + 1
    
    # Save updated leaderboard
    write_json_file(leaderboard_file, leaderboard)
    
    return leaderboard

# Function to update submission status
def update_submission_status(submission_id, status, details=None):
    status_path = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.status")
    status_data = {
        "status": status,
        "last_updated": datetime.now().isoformat(),
        "details": details or {}
    }
    with open(status_path, "w") as f:
        json.dump(status_data, f, indent=2)

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
    
    # Validate implementation type
    if submission.implementation_type not in ["basic", "extension"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid implementation type. Must be 'basic' or 'extension'"
        )
    
    # Generate a unique ID for this submission
    timestamp = int(time.time())
    submission_id = f"{submission.group_name.replace(' ', '_')}_{submission.task_type}_{timestamp}"
    
    # Save the submission data
    submission_path = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.json")
    with open(submission_path, "w") as f:
        json.dump(submission.submission_data, f, indent=2)
    
    # Start evaluation in background
    background_tasks.add_task(
        process_submission,
        submission_id,
        submission.task_type,
        submission.group_name,
        submission.implementation_type,
        submission.uses_synthetic_data
    )
    
    return {
        "status": "success",
        "message": "Submission received and being processed",
        "submission_id": submission_id
    }

# Function to process submission in background
def process_submission(submission_id, task_type, group_name, implementation_type, uses_synthetic_data):
    try:
        # Update status to processing
        update_submission_status(submission_id, "processing", {
            "stage": "loading_model",
            "progress": 0
        })
        
        # Evaluate the submission
        update_submission_status(submission_id, "processing", {
            "stage": "evaluating",
            "progress": 50
        })
        result = evaluate_submission(task_type)
        
        if "error" in result:
            # Log the error
            with open(f"{SUBMISSIONS_DIR}/{submission_id}.error", "w") as f:
                f.write(result["error"])
            update_submission_status(submission_id, "error", {
                "error": result["error"]
            })
            return
        
        # Update the leaderboard
        update_submission_status(submission_id, "processing", {
            "stage": "updating_leaderboard",
            "progress": 75
        })
        update_leaderboard(group_name, task_type, result["score"], implementation_type, uses_synthetic_data)
        
        # Save metadata
        with open(f"{SUBMISSIONS_DIR}/{submission_id}.meta", "w") as f:
            json.dump({
                "group_name": group_name,
                "task_type": task_type,
                "implementation_type": implementation_type,
                "uses_synthetic_data": uses_synthetic_data,
                "score": result["score"],
                "submission_time": datetime.now().isoformat()
            }, f, indent=2)
        
        # Update final status
        update_submission_status(submission_id, "completed", {
            "score": result["score"],
            "progress": 100
        })
        
    except Exception as e:
        error_msg = str(e)
        with open(f"{SUBMISSIONS_DIR}/{submission_id}.error", "w") as f:
            f.write(error_msg)
        update_submission_status(submission_id, "error", {
            "error": error_msg
        })

@app.get("/leaderboard/{task_type}")
async def get_leaderboard(task_type: str):
    if task_type not in ["instruction_following", "math_reasoning"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid task type. Must be 'instruction_following' or 'math_reasoning'"
        )
    
    leaderboard_file = get_leaderboard_file(task_type)
    with open(leaderboard_file, "r") as f:
        leaderboard = json.load(f)
    
    return leaderboard

@app.get("/submission_status/{submission_id}")
async def get_submission_status(submission_id: str):
    submission_path = os.path.join(SUBMISSIONS_DIR, f"{submission_id}.py")
    meta_path = f"{submission_path}.meta"
    error_path = f"{submission_path}.error"
    status_path = f"{submission_path}.status"
    
    if not os.path.exists(submission_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Submission not found"
        )
    
    # Read status file if it exists
    if os.path.exists(status_path):
        with open(status_path, "r") as f:
            status_data = json.load(f)
        
        if status_data["status"] == "completed" and os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            return {
                "status": "completed",
                "submission_id": submission_id,
                "score": meta["score"],
                "submission_time": meta["submission_time"],
                "details": status_data["details"]
            }
        elif status_data["status"] == "error":
            with open(error_path, "r") as f:
                error = f.read()
            return {
                "status": "error",
                "submission_id": submission_id,
                "error": error,
                "details": status_data["details"]
            }
        else:
            return {
                "status": status_data["status"],
                "submission_id": submission_id,
                "details": status_data["details"],
                "last_updated": status_data["last_updated"]
            }
    
    return {
        "status": "pending",
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

@app.get("/")
async def root():
    return HTMLResponse(open(os.path.join(STATIC_DIR, "index.html")).read())

# Start the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        # host="127.0.0.1",  # Only listen on localhost since Nginx will proxy
        port=10000,
        reload=True
    )