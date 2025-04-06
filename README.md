# ML Model Leaderboard System Documentation

## Overview

This leaderboard system allows students to submit and compare their model implementations based on the Qwen 2.5 0.5B (non-instruct) model. The system evaluates submissions on two tasks:

1. **Instruction Following (Ultrafeedback)** - Evaluating how well the model follows instructions
2. **Math Reasoning (Countdown)** - Evaluating mathematical problem-solving capabilities

## Requirements

- **Base Model**: All submissions must use the Qwen 2.5 0.5B (non-instruct) model
- **Evaluation**: Submissions are scored using a hidden evaluation set
- **Fairness**: All submissions use the same base model for fair comparison

## System Components

The leaderboard system consists of:

1. **Backend API** - Handles submission processing, evaluation, and leaderboard data
2. **Frontend Interface** - Provides a user-friendly way to interact with the leaderboard
3. **Evaluation Pipeline** - Processes submitted models against hidden test sets

## Installation and Setup

### Prerequisites

- Python 3.8+
- Node.js and npm (for frontend development)
- Required Python packages:
  - fastapi
  - uvicorn
  - transformers
  - torch
  - datasets
  - pydantic

### Backend Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/your-organization/model-leaderboard.git
   cd model-leaderboard
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create necessary directories:
   ```bash
   mkdir -p data submissions
   ```

4. Place evaluation datasets in the data directory:
   - `data/ultrafeedback_hidden_eval.json`
   - `data/countdown_hidden_eval.json`

5. Start the backend server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Frontend Setup

1. Open the HTML file in a web browser:
   ```bash
   open frontend/index.html
   ```

## Using the Leaderboard

### Viewing the Leaderboard

1. Navigate to the leaderboard page at `http://localhost:8000` in your web browser
2. View current standings for both Instruction Following and Math Reasoning tasks
3. See current thresholds for basic implementation and extensions

### Submitting a Model

1. Navigate to the "Submit Model" section
2. Complete the submission form:
   - Student ID: Your unique identifier
   - Task Type: Select either Instruction Following or Math Reasoning
   - Description: Brief explanation of your approach
   - Hyperparameters: JSON object with your model's hyperparameters
   - Model Code: Python code containing your model implementation

3. Requirements for model code:
   - Must define a `run_model(input_text)` function
   - Must use the provided base model (`base_model` and `tokenizer` are injected)
   - Should return a string prediction

Example submission code:

```python
def run_model(input_text):
    # Preprocessing
    input_text = input_text.strip()
    
    # Tokenize
    input_tokens = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate output
    output_tokens = base_model.generate(
        input_tokens,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        num_beams=4
    )
    
    # Decode and return
    return tokenizer.decode(output_tokens[0], skip_special_tokens=True)
```

### Checking Submission Status

1. Navigate to the "Submission Status" section
2. Enter your submission ID
3. View the current status:
   - Processing: Your submission is being evaluated
   - Completed: Evaluation finished, score available
   - Error: Something went wrong during evaluation

## Evaluation Criteria

### Instruction Following (Ultrafeedback)

The Ultrafeedback evaluation assesses:
- Helpfulness: How well the model follows the given instruction
- Honesty: The factual accuracy of the model's response
- Harmlessness: Ensuring the model doesn't produce harmful content

The score is a weighted combination of these aspects.

### Math Reasoning (Countdown)

The Countdown evaluation assesses:
- Correctness: Whether the final answer matches the reference
- Reasoning: The quality of the step-by-step reasoning
- Efficiency: The conciseness and clarity of the solution

## Submission Guidelines

1. **Submission Limits**:
   - Maximum 5 submissions per student per day
   - Maximum 20 submissions total per task type

2. **Extension Ideas**:
   - Improving prompt engineering techniques
   - Implementing context distillation
   - Designing specialized instruction templates
   - Synthetic data generation for domain-specific knowledge
   - Implementing parameter-efficient fine-tuning

3. **Minimum Thresholds**:
   - Instruction Following:
     - Basic Implementation: 0.65
     - Extension: 0.75
   - Math Reasoning:
     - Basic Implementation: 0.60
     - Extension: 0.70

## API Reference

### Endpoints

- `POST /submit` - Submit a model
- `GET /leaderboard/{task_type}` - Get leaderboard data
- `GET /submission_status/{submission_id}` - Check submission status
- `GET /thresholds` - Get current threshold values
- `GET /guidelines` - Get submission guidelines
