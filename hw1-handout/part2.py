import torch
from torch import Tensor
import sytorch as st
from helpers import load_testset, load_edit_dataset_part_2, load_model, save_model, test
from copy import deepcopy
from part1 import your_edit
import operator
import os
import json
import sys
import csv
from itertools import product
import random

def edit(model, images, labels):
    # raise NotImplementedError(
    #     "Editing method for `part2.py` not implemented."
    best_models = []
    results = []
    best_model = None
    best_result = None
    best_test_acc = -1.0
    model_id = 0 

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
        (operator.gt, 5),
        (operator.lt, -5)
    ]

    param_grid = {
        "layer": list(range(-5, 5)),
        "multi_layer": [False, True],
        "partial_layer": [True, False],
        "lb": [-1., -10., -100.],
        "ub": [1., 10., 100.],
        "norms": norms_options,
        "mask": mask_options,
    } 

    all_combinations = list(product(
        param_grid["norms"],
        param_grid["mask"],
        param_grid["lb"],
        param_grid["ub"],
        param_grid["partial_layer"],
        param_grid["layer"],
        param_grid["multi_layer"]
    ))

    sampled_combinations = random.sample(all_combinations, 500)

    for norms, (mask_fn, threshold), lb, ub, partial_layer, layer, multi_layer in sampled_combinations:
        try:
            edited_model = your_edit(
                model,
                inputs=images,
                labels=labels,
                layer=layer,
                multi_layer=multi_layer,
                partial_layer=partial_layer,
                lb=lb,
                ub=ub,
                norms=norms,
                mask_fn=mask_fn,
                threshold=threshold
            )

            if edited_model is None:
                continue  # skip null edits

            edit_acc = test(edited_model, edit_dataset)
            test_acc = test(edited_model, load_testset())

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
            }

            results.append(result)

            # Update best model
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model = edited_model
                best_result = result

        except Exception as e:
            print(f"❌ Failed with error: {e}")
            continue
    
    # Sort by edit_acc descending, filter out test_acc < 90%
    valid_results = [r for r in results if r["test_acc"] >= 90.0]
    valid_results.sort(key=lambda x: x["edit_acc"], reverse=True)

    # Write all original results to CSV
    csv_path = "my_model/part2/full_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "status", "layer", "multi_layer", "partial_layer",
            "lb", "ub", "norms", "mask_fn", "threshold",
            "edit_acc", "test_acc"
        ])
        writer.writeheader()
        for res in results:
            writer.writerow({
                "status": res["status"],
                "layer": res["layer"],
                "multi_layer": res["multi_layer"],
                "partial_layer": res["partial_layer"],
                "lb": res["lb"],
                "ub": res["ub"],
                "norms": " ".join(res["norms"]),
                "mask_fn": res["mask_fn"],
                "threshold": res["threshold"],
                "edit_acc": res["edit_acc"],
                "test_acc": res["test_acc"]
            })

    # Select best valid result
    best_result = valid_results[0] if valid_results else None

    if best_result:
        print("\nBest Model Configuration (test_acc ≥ 90%):")
        for k, v in best_result.items():
            print(f"{k}: {' '.join(v) if k == 'norms' else v}")
        print(f"\nBest Edit Accuracy: {best_result['edit_acc']:.4f}")
        print(f"Best Test Accuracy: {best_result['test_acc']:.4f}")
    else:
        print("No valid edited model met the accuracy threshold (test_acc ≥ 90%).")

    return best_result["model"] if best_result else None


if __name__ == "__main__":

    """ Load the pre-trained model to edit. """
    model = load_model(path='model.pt')

    """ Load the edit dataset and test dataset. """
    edit_dataset = load_edit_dataset_part_2()

    edit_dataloader = torch.utils.data.DataLoader(
        edit_dataset, batch_size=10, shuffle=False
    )

    images, labels = next(iter(edit_dataloader))

    """ TODO: Implement the edit function. The edited_model returned
        by the function should maximize the accuracy on edit_dataset
        while ensuring that the accuracy on the test set is >= 90%.
    """
    edited_model = edit(model, images, labels)

    if edited_model is None:
        print("Editing failed.")
        exit(1)

    print("Edited model successfully!\nTry to maximize the accuracy on the edit set.")
    edit_acc = test(edited_model, edit_dataset)

    print("Accuracy on the test set should >= 90%:")
    test_dataset = load_testset()
    test_acc = test(edited_model, test_dataset)

    """ Required: save the edited model to 'artifacts/edited_model_part2.pt'. """
    save_model(edited_model, 'artifacts/edited_model_part2.pt')
