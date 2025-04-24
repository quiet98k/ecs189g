import torch
from torch import Tensor
import sytorch as st
from helpers import load_testset, load_edit_dataset_part_1, load_model, save_model, test
from copy import deepcopy

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
    editable_model[layer:].requires_edit_(lb=lb, ub=ub)

    # Example 3: Edit part of the parameters, specified by a boolean mask.
    # editable_model[layer].weight.requires_edit_(
    #     mask=editable_model[layer].weight > 0.,
    #     lb=lb, ub=ub,
    # )


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

def your_edit(model, *args, **kwargs):
    raise NotImplementedError(
        "Editing method for `part1.py` not implemented."
    )

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
    edited_model = your_edit(model,
        inputs=all_images,
        labels=all_labels)

    if edited_model is None:
        print("Editing failed.")
        exit(1)

    print("Edited model successfully!\nAccuracy on the edit set should be 100%:")
    edit_acc = test(edited_model, edit_dataset)

    print("Accuracy on the test set should >= 95%:")
    test_dataset = load_testset()
    test_acc = test(edited_model, test_dataset)

    """ Required: save the edited model to 'artifacts/edited_model_part1.pt'. """
    save_model(edited_model, 'artifacts/edited_model_part1.pt')
