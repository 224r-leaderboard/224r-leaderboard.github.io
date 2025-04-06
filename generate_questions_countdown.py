import random
import math
import itertools
import operator
import pandas as pd
from countdown import evaluate_equation, compute_score

min_number = 0
max_number = 99
number_range = range(3, 5)  # Will generate 3-4 numbers
operations = ['+', '-', '*', '/']
num_problems = 200

def generate_numbers():
    """Generate a list of random numbers."""
    return [random.randint(min_number, max_number) for _ in range(random.choice(number_range))]

def generate_operations(num_operations):
    """Generate a list of random operations."""
    return [random.choice(operations) for _ in range(num_operations)]

def generate_expression(numbers, operations):
    """Generate a mathematical expression from numbers and operations with parentheses."""
    # Basic expression without parentheses
    expression_parts = []
    for i in range(len(numbers)):
        if i > 0:
            expression_parts.append(operations[i-1])
        expression_parts.append(str(numbers[i]))
    
    # Randomly add parentheses if we have at least 3 numbers
    if len(numbers) >= 3:
        # Decide whether to add parentheses (50% chance)
        if random.random() > 0.5:
            # Choose a random position to add parentheses (around 2 numbers and 1 operation)
            start_pos = random.randint(0, len(numbers) - 2)
            
            # Add opening parenthesis before the number
            expression_parts[start_pos * 2] = "(" + expression_parts[start_pos * 2]
            
            # Add closing parenthesis after the operation and next number
            if start_pos < len(numbers) - 2:
                expression_parts[start_pos * 2 + 3] = expression_parts[start_pos * 2 + 3] + ")"
            else:
                expression_parts[-1] = expression_parts[-1] + ")"
    
    # Join all parts to form the expression
    expression = " ".join(expression_parts)
    return expression

def generate_problem():
    """Generate a random problem."""
    numbers = generate_numbers()
    num_operations = len(numbers) - 1
    operations = generate_operations(num_operations)
    expression = generate_expression(numbers, operations)
    result = evaluate_equation(expression)
    
    # Check if result is None (error occurred) or not an integer
    # or out of the specified range
    if result is None or result != int(result) or not (min_number <= result <= max_number):
        return generate_problem()
    
    return numbers, expression, int(result)

def generate_problems(num_problems):
    """Generate a list of random problems."""
    problems = []
    for _ in range(num_problems):
        numbers, expression, result = generate_problem()
        problems.append((numbers, expression, result))
    return problems

# Generate problems and save to JSON
generated_problems = generate_problems(num_problems)

# now validate
valid_problems = []
num_invalid = 0
for numbers, expression, result in generated_problems:
    # Check if the equation is valid
    ground_truth = {'target': result, 'numbers': numbers}
    if compute_score("Assistant: <answer>" + expression + '</answer>', ground_truth) == 1:
        valid_problems.append((numbers, expression, result))
    else:
        print(f"Invalid problem: {numbers}, {expression}, {result}")
        num_invalid += 1
        continue
print(f"Invalid problems: {num_invalid}")

df = pd.DataFrame(generated_problems, columns=['num', 'expression', 'target'])
df.to_json('/Users/anikaitsingh/Desktop/leaderboard/data/countdown_heldout_prompts.json', orient='records', lines=True)