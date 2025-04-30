import torch
from torch import Tensor
import sytorch as st
from helpers import load_testset, load_edit_dataset_part_1, load_model, save_model, test
from copy import deepcopy
import operator
import os
import json
import sys
import csv

# class Tee:
#     def __init__(self, *streams):
#         self.streams = streams

#     def write(self, data):
#         for s in self.streams:
#             s.write(data)
#             s.flush()

#     def flush(self):
#         for s in self.streams:
#             s.flush()

# sys.stdout = Tee(sys.stdout, open("output.txt", "w"))

def example_edit_method(
    model: torch.nn.Module,
    inputs: Tensor,
    labels: Tensor,
    #
    # Editing options:
    layer: int = 0,
    lb: float = -10.,
    ub: float =  10.,
):
    """ This is an example of how to edit a model using sytorch.
        Feel free to modify its parameters/implementation/returns!
    """

    # Create a copy of the model if you don't want to modify the original model.
    model = deepcopy(model)

    # Convert the model to an editable symbolic model. The two models will
    # share the same parameters, so editing one will affect the other.
    editable_model = st.nn.to_editable(model)

    # Comment this line to turn on the gurobi progress output.
    editable_model.solver.verbose_(False)


    """ Set Editable Parameters.
        ========================================================================
        Just like fine-tuning, we can set the parameters of the model we want to
        edit. Editing more parameters will result in a more flexible edit, but
        will also require more computation.

        For any symbolic module like
        - `editable_model[layer]` and `editable_model[layer:]`: a sequence of layers;
        - `editable_model[layer]`: a single layer;
        and any symbolic parameter like
        - `editable_model[layer].weight`: the weight of a layer;
        - `editable_model[layer].bias`: the bias of a layer;
        you can call `.requires_edit_(lb=lb, ub=ub, mask=mask)` to set the
        - floats `lb` and `ub` bounds for the parameters changes, and
        - a boolean tensor `mask` to control the subset of parameters to edit
          (default=None for editing all parameters).
    """
    # Example 1: Edit only the layer editable_model[layer]
    # editable_model[layer].requires_edit_(lb=lb, ub=ub)

    # Example 2: Edit all layers from layer to the end of the model
    # editable_model[layer:].requires_edit_(lb=lb, ub=ub)

    # Example 3: Edit part of the parameters, specified by a boolean mask.
    editable_model[layer].weight.requires_edit_(
        mask=editable_model[layer].weight > 0.,
        lb=lb, ub=ub,
    )


    """ Symbolic Forward Pass
        ========================================================================
        - `editable_model(inputs)` returns an `st.SymbolicTensor` containing
          symbolic variables representing the outputs of the model.
    """

    symbolic_outputs = editable_model(inputs).data

    """ Asserting the Output ArgMax Constraint
        ========================================================================
        - `symbolic_outputs.argmax(-1) == labels` computes the argmax of the
          symbolic outputs along the last dimension, returning the constraints;
        - `output_constraints.assert_()` adds the constraints to the optimization
          problem. Under the hood it is equivalent to
          `editable_model.solver.add_constraints(output_constraints)`.
    """
    output_constraints = symbolic_outputs.argmax(-1) == labels
    output_constraints.assert_()


    """ Example Minimization Objective: parameter difference norm
        ========================================================================
        - `editable_model.param_delta()` returns an `st.SymbolicTensor`
          containing variables of the difference between the original and
          edited model parameters, flattened into an 1d vector.

        - `st.SymbolicTensor.norm_ub('linf+l1n')` computes the (linearized
           upper bound of) the norm of the 1d symbolic tensor. Available
           norms are:
           - 'l1': L1 norm (sum of absolute values).
           - 'l1n': L1 normalized norm (L1 norm divided by the number of elements).
           - 'linf': L_inf norm (max absolute value).
           - mix of the above, e.g. 'l1+l1n', 'linf+l1n', etc.

    """
    objective = editable_model.param_delta().norm_ub('linf+l1n')

    """ Another Minimization Objective: output difference norm
        ========================================================================
        - `symbolic_outputs.dense_delta_1d()` returns an `st.SymbolicTensor`
          containing variables of the difference between the original and
          edited model outputs, flattened into an 1d vector.
    """
    # objective = objective + .5 * symbolic_outputs.dense_delta_1d().norm_ub('linf+l1n')


    """ Optimize the Model
        ========================================================================
        - `editable_model.optimize(minimize=objective)` solves the optimization
          problem with the given objective, returning True if the optimization
          was successful and False otherwise.
    """
    if editable_model.optimize(minimize=objective):
        return model

    else:
        return None

def your_edit(model, inputs, labels, 
              verbose = False,
              layer=-1, 
              multi_layer = False,
              partial_layer = False,
              lb=-10., ub=10.,
              norms=['l1'],
              mask_fn=operator.gt,
              threshold=0
              ):
    # raise NotImplementedError(
    #     "Editing method for `part1.py` not implemented."
    # )
    
    print(f"Trying with: {layer = }, {multi_layer = }, {partial_layer = }, "
        f"{lb = }, {ub = }, {norms = }, {mask_fn.__name__ = }, {threshold = }")
    
    model = deepcopy(model)

    editable_model = st.nn.to_editable(model)

    editable_model.solver.verbose_(verbose)
    
    
    
    if multi_layer:
        if partial_layer:
            if hasattr(editable_model[layer], "weight"):
                mask = mask_fn(editable_model[layer].weight, threshold)
            editable_model[layer:].weight.requires_edit_(mask=mask, lb=lb, ub=ub)
        else:
            editable_model[layer:].requires_edit_(lb=lb, ub=ub)
    else:
        editable_model[layer].requires_edit_(lb=lb, ub=ub)

    
    symbolic_outputs = editable_model(inputs).data
    
    output_constraints = symbolic_outputs.argmax(-1) == labels
    output_constraints.assert_()
    
    objective = editable_model.param_delta().norm_ub('+'.join(norms))
    
    if editable_model.optimize(minimize=objective):
        return model

    else:
        return None
    

def run_hyperparameter_search(model, all_images, all_lables):
    best_models = []
    results = []
    norms_options = [
        ['l1'],
        ['l1n'],
        ['linf'],
        ['l1', 'l1n'],
        ['l1', 'linf'],
        ['l1n', 'linf'],
        ['l1', 'l1n', 'linf'],
    ]
    
    mask_options = [
        (operator.gt, 0),
        (operator.lt, 0),
        (operator.gt, 0.5),
        (operator.lt, -0.5),
        (operator.gt, 1),
        (operator.lt, -1)
    ]

    param_grid = {
        "layer": [-2, -1, 0, 1],
        "multi_layer": [False, True],
        "partial_layer": [True, False],
        "lb": [-1., -10., -100.],
        "ub": [1., 10., 100.],
        "norms": norms_options,
        "mask": mask_options
    }
    
    # norms_options = [
    #     ['l1'],
    # ]
    
    # mask_options = [
    #     (operator.gt, 0),
    # ]

    # param_grid = {
    #     "layer": [-2],
    #     "multi_layer": [True],
    #     "partial_layer": [False],
    #     "lb": [-1.],
    #     "ub": [1.],
    #     "norms": norms_options,
    #     "mask": mask_options
    # }


    from itertools import product

    for norms, (mask_fn, threshold), lb, ub, partial_layer, layer, multi_layer in product(
        param_grid["norms"],
        param_grid["mask"],
        param_grid["lb"],
        param_grid["ub"],
        param_grid["partial_layer"],
        param_grid["layer"],
        param_grid["multi_layer"]):



        try:
            edited_model = your_edit(
                model,
                inputs=all_images,
                labels=all_labels,
                layer=layer,
                multi_layer=multi_layer,
                partial_layer=partial_layer,
                lb=lb,
                ub=ub,
                norms=norms,
                mask_fn=mask_fn,
                threshold=threshold
            )
            if edited_model is not None:
                edit_acc = test(edited_model, edit_dataset)
                test_dataset = load_testset()
                test_acc = test(edited_model, test_dataset)
                
                
                result = {
                    "status": "success",
                    "layer": layer,
                    "multi_layer": multi_layer,
                    "partial_layer": partial_layer,
                    "lb": lb,
                    "ub": ub,
                    "norms": norms,
                    "mask_fn": mask_fn.__name__,
                    "threshold": threshold,
                    "edit_acc": edit_acc,
                    "test_acc": test_acc,
                    "error": ""
                }

                
                model_id = len(best_models)
                base_dir = f"my_model/part1/model_{model_id}"
                os.makedirs(base_dir, exist_ok=True)

                save_model(edited_model, os.path.join(base_dir, "model.pt"))
                with open(os.path.join(base_dir, "params.json"), "w") as f:
                    json.dump(result, f, indent=2)

                best_models.append({"model": edited_model, "params": result})
                
            else:
                result = {
                    "status": "null_model",
                    "layer": layer,
                    "multi_layer": multi_layer,
                    "partial_layer": partial_layer,
                    "lb": lb,
                    "ub": ub,
                    "norms": norms,
                    "mask_fn": mask_fn.__name__,
                    "threshold": threshold,
                    "edit_acc": None,
                    "test_acc": None,
                    "error": "edited_model is None"
                }

        except Exception as e:
            print(f"❌ Failed with error: {e}")
            result = {
                "status": "error",
                "layer": layer,
                "multi_layer": multi_layer,
                "partial_layer": partial_layer,
                "lb": lb,
                "ub": ub,
                "norms": norms,
                "mask_fn": mask_fn.__name__,
                "threshold": threshold,
                "edit_acc": None,
                "test_acc": None,
                "error": str(e)
            }

        results.append(result)

        with open("my_model/part1/full_results.csv", "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "status", "layer", "multi_layer", "partial_layer",
                "lb", "ub", "norms", "mask_fn", "threshold",
                "edit_acc", "test_acc", "error"
            ])
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow({
                "status": result["status"],
                "layer": result["layer"],
                "multi_layer": result["multi_layer"],
                "partial_layer": result["partial_layer"],
                "lb": result["lb"],
                "ub": result["ub"],
                "norms": " ".join(result["norms"]),
                "mask_fn": result["mask_fn"],
                "threshold": result["threshold"],
                "edit_acc": result["edit_acc"],
                "test_acc": result["test_acc"],
                "error": result["error"]
            })



if __name__ == "__main__":

    """ Load the pre-trained model to edit. """
    model = load_model(path='model.pt')

    """ Load the edit dataset and test dataset. """
    edit_dataset = load_edit_dataset_part_1()

    all_images, all_labels = edit_dataset.tensors

    """
    TODO: Implement and call your edit function here (e.g., your_edit).
          The returned edited_model should be correct for all images in the edit_dataset.
          The accuracy of the edited_model should be at least 95% on the test set.

          The example_edit_method shown above should get you started on how to use sytorch.
          The below statement shows how to call this example_edit_method.
            edited_model = example_edit_method(model,
                inputs=all_images,
                labels=all_labels,
                layer=-1,
                lb=-10., ub=10.,
            )
    """
    
    run_hyperparameter_search(model, all_images, all_labels)
    
    # edited_model = your_edit(model, inputs=all_images, labels=all_labels,
    #                          verbose=True,
    #                          layer=-2,
    #                          multi_layer=False,
    #                          partial_layer=False,
    #                          lb = -10., ub = 10.,
    #                          norms=['l1n'],
    #                          mask_fn=operator.lt,
    #                          threshold=0)
    # if edited_model is None:
    #     print("Editing failed.")
    #     exit(1)

    # print("Edited model successfully!\nAccuracy on the edit set should be 100%:")
    # edit_acc = test(edited_model, edit_dataset)

    # print("Accuracy on the test set should >= 95%:")
    # test_dataset = load_testset()
    # test_acc = test(edited_model, test_dataset)

    # """ Required: save the edited model to 'artifacts/edited_model_part1.pt'. """
    # save_model(edited_model, 'artifacts/edited_model_part1.pt')
