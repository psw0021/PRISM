import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
# Add the root directory to Python's module search path
sys.path.append(root_dir)

def update_cost_in_file(prompt_tokens, completion_tokens, model, file_path="openai_cost.txt"):
    """
    Computes the cost of an OpenAI API call and updates it in a file.

    Args:
        prompt_tokens (int): The number of tokens in the input prompt.
        completion_tokens (int): The number of tokens in the generated completion.
        model (str): The model used (e.g., "gpt-4", "gpt-3.5-turbo").
        file_path (str): Path to the file where costs are logged.

    Returns:
        None
    """
    try:
        # Pricing per 1,000 tokens (as of 2024)
        pricing = {
            "gpt-4o-2024-08-06": {"prompt": 0.00250 , "completion": 0.01000}, 
            "gpt-4o-mini-2024-07-18": {"prompt": 0.000150, "completion": 0.000600}
        }

        if model not in pricing:
            raise ValueError(f"Pricing for model '{model}' is not defined.")
    
        # Calculate costs
        prompt_cost_per_token = pricing[model]["prompt"] / 1000
        completion_cost_per_token = pricing[model]["completion"] / 1000
    
        current_total_cost = (prompt_cost_per_token * prompt_tokens) + (completion_cost_per_token * completion_tokens)
    
    
        # update the cost
        with open(file_path, 'r') as file:
            previous_cost = file.read()
            
        updated_cost = float(previous_cost) + current_total_cost
        
        with open(file_path, 'w') as f:
            f.write(str(updated_cost))
                    
    except:
        pass