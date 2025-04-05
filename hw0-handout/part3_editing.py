""" Implement this script so that
    ```
    uv run part3_editing.py
    ```

    will edit the last layer of your trained DNN from Part 1, so that the edited
    DNN correctly classifies both the original and adversarial images you found
    in part 2.

    You could use the following code to load the adversarial example:
    ```
    from helpers import AdversarialExamples
    example = AdversarialExamples.load('artifacts/adversarial_examples.pt')
    ```

    And use the following code to save the edited last layer:
    ```
    model.save_layer(-1, 'artifacts/edited_last_layer.pt')
    ```

    See README.md for the requirements.
"""
import gurobipy as gp
from gurobipy import GRB
import torch
import numpy as np
from utils import test_performance



from helpers import Model
model = Model().load('artifacts/model.pt')

from helpers import AdversarialExamples
example = AdversarialExamples.load('artifacts/adversarial_examples.pt')

with torch.no_grad():
    adv_intermediate = model.layers[:-1](example.adv_images)
    adv_intermediate_np = adv_intermediate.detach().cpu().numpy()[0]  # Take the first example
    print("adv_intermediate_np shape:", adv_intermediate_np.shape)
    
    orig_intermediate = model.layers[:-1](example.images)
    orig_intermediate_np = orig_intermediate.detach().cpu().numpy()[0]  # Take the first example
    print("orig_intermediate_np shape:", orig_intermediate_np.shape)
    
correct_label = example.targets.item()
print("Correct Label: ", correct_label)

last_layer = model.layers[-1]
# print("last_layer: ", last_layer)
# print("Last layer weights: ", last_layer.weight)
# print("Last layer biases: ", last_layer.bias)
weights = last_layer.weight.detach().cpu().numpy() 
biases = last_layer.bias.detach().cpu().numpy() 
print("Weights shape:", weights.shape)
print("Biases shape:", biases.shape)

def validate_inputs(v1, v2, W_old, b_old, l1, l2):
    # Check if v1 and v2 are 1D NumPy arrays of size 100
    if not isinstance(v1, np.ndarray) or v1.ndim != 1 or v1.shape[0] != 100:
        raise ValueError("v1 must be a 1D NumPy array with shape (100,)")
    if not isinstance(v2, np.ndarray) or v2.ndim != 1 or v2.shape[0] != 100:
        raise ValueError("v2 must be a 1D NumPy array with shape (100,)")

    # Check if W_old is a 2D NumPy array of shape (10, 100)
    if not isinstance(W_old, np.ndarray) or W_old.ndim != 2 or W_old.shape != (10, 100):
        raise ValueError("W_old must be a 2D NumPy array with shape (10, 100)")

    # Check if b_old is a 1D NumPy array of size 10
    if not isinstance(b_old, np.ndarray) or b_old.ndim != 1 or b_old.shape[0] != 10:
        raise ValueError("b_old must be a 1D NumPy array with shape (10,)")
    
    if not isinstance(l1, int) or not (0 <= l1 <= 9):
        raise ValueError("l1 must be an integer between 0 and 9")
    if not isinstance(l2, int) or not (0 <= l2 <= 9):
        raise ValueError("l2 must be an integer between 0 and 9")

def find_last_layer(v1, l1, v2, l2, W_old, b_old, epsilon=1e-3):
    
    validate_inputs(v1, v2, W_old, b_old, l1, l2)

    gp_model = gp.Model("gp_model")
    gp_model.setParam('OutputFlag', 0) 

    W = gp_model.addVars(10, 100, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="W")
    b = gp_model.addVars(10, lb=-GRB.INFINITY, ub=GRB.INFINITY, name="b")

    y1 = {}
    for i in range(10):
        y1[i] = gp.LinExpr()
        y1[i] += b[i]
        for j in range(100):
            y1[i] += W[i, j] * v1[j]

    y2 = {}
    for i in range(10):
        y2[i] = gp.LinExpr()
        y2[i] += b[i]
        for j in range(100):
            y2[i] += W[i, j] * v2[j]

    for i in range(10):
        if i != l1:
            gp_model.addConstr(y1[l1] >= y1[i] + epsilon)
        if i != l2:
            gp_model.addConstr(y2[l2] >= y2[i] + epsilon)  
            
    obj = gp.QuadExpr()
    for i in range(10):
        for j in range(100):
            obj += (W[i, j] - W_old[i, j])**2  # Minimize change in W
    for i in range(10):
        obj += (b[i] - b_old[i])**2  # Minimize change in b
        
    gp_model.setObjective(obj, GRB.MINIMIZE)
    
    gp_model.optimize()
    
    W_opt = np.array([[W[i, j].X for j in range(100)] for i in range(10)])
    b_opt = np.array([b[i].X for i in range(10)])
    
    return W_opt, b_opt


original_performance = test_performance(model)

new_weights, new_biases = find_last_layer(orig_intermediate_np, correct_label, adv_intermediate_np, correct_label, weights, biases)
with torch.no_grad():
    model.layers[-1].weight.copy_(torch.tensor(new_weights))
    model.layers[-1].bias.copy_(torch.tensor(new_biases))

new_performance = test_performance(model)

drawdown = (original_performance - new_performance) / original_performance * 100
print(f"Performance drawdown: {drawdown:.2f}%")








model.save_layer(-1, 'artifacts/edited_last_layer.pt')