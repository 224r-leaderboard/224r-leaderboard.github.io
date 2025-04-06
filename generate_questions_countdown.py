import random
import math
import itertools
import operator
import pandas as pd

min_number = 0
max_number = 99
number_range = range(3,5)
operations = ['+', '-', '*', '/']
num_problems = 200

def generate_numbers():
    """Generate a list of random numbers."""
    return [random.randint(min_number, max_number) for _ in range(random.choice(number_range))]

def generate_operations(num_operations):
    """Generate a list of random operations."""
    return [random.choice(operations) for _ in range(num_operations)]

def generate_expression(numbers, operations):
    """Generate a mathematical expression from numbers and operations."""
    expression = str(numbers[0])
    for i in range(len(operations)):
        expression += f" {operations[i]} {numbers[i + 1]}"
    return expression

def evaluate_expression(expression):
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression)
        return result
    except:
        # Handle potential errors (division by zero, etc.)
        return None

def generate_problem():
    """Generate a random problem."""
    numbers = generate_numbers()
    num_operations = len(numbers) - 1
    operations = generate_operations(num_operations)
    expression = generate_expression(numbers, operations)
    result = evaluate_expression(expression)
    
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


generated_problems = generate_problems(num_problems)
df = pd.DataFrame(generated_problems, columns=['num', 'expression', 'target'])
df.to_json('/Users/anikaitsingh/Desktop/leaderboard/data/countdown_heldout_prompts.json', orient='records', lines=True)