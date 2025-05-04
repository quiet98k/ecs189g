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
from itertools import product, islice
import random
from contextlib import contextmanager
import optuna
# optuna.logging.set_verbosity(optuna.logging.WARNING)


# @contextmanager
# def suppress_stdout():
    # original_stdout = sys.stdout
    # sys.stdout = open(os.devnull, 'w')
    # try:
    #     yield
    # finally:
    #     sys.stdout.close()
    #     sys.stdout = original_stdout

def load_batches(edit_dataset, max_batches):
    
    edit_dataloader = torch.utils.data.DataLoader(
        edit_dataset, batch_size=10, shuffle=False
    )

    all_images = []
    all_labels = []

    batch_iter = edit_dataloader if max_batches == 0 else islice(edit_dataloader, max_batches)

    for images, labels in batch_iter:
        all_images.append(images)
        all_labels.append(labels)

    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_images, all_labels

norms_dict = {
    "l1": ['l1'],
    "l1n": ['l1n'],
    "linf": ['linf'],
    "l1_l1n": ['l1', 'l1n'],
    "l1_linf": ['l1', 'linf'],
    "l1n_linf": ['l1n', 'linf'],
    "all": ['l1', 'l1n', 'linf'],
}

mask_dict = {
    "gt_0": (operator.gt, 0),
    "lt_0": (operator.lt, 0),
    "gt_1": (operator.gt, 1),
    "lt_neg1": (operator.lt, -1),
    "gt_5": (operator.gt, 5),
    "lt_neg5": (operator.lt, -5),
    "ge_0": (operator.ge, 0),
    "le_0": (operator.le, 0),
}

def edit(model, max_iteration=500):
    edit_dataset = load_edit_dataset_part_2()

    def objective(trial):
        num_batches = trial.suggest_int("num_batches", 1, 500, log=True)
        layer = trial.suggest_int("layer", -10, 9)
        multi_layer = trial.suggest_categorical("multi_layer", [True, False])
        partial_layer = trial.suggest_categorical("partial_layer", [True, False])
        lb = trial.suggest_categorical("lb", [-0.1, -1., -10., -100., -1000.])
        ub = trial.suggest_categorical("ub", [0.1, 1., 10., 100., 1000.])

        norm_key = trial.suggest_categorical("norms", list(norms_dict.keys()))
        norms = norms_dict[norm_key]

        mask_key = trial.suggest_categorical("mask", list(mask_dict.keys()))
        mask_fn, threshold = mask_dict[mask_key]


        print("\n" + "=" * 80)
        print(f"Trying config | {num_batches = }, {layer = }, {multi_layer = }, {partial_layer = }, "
              f"{lb = }, {ub = }, {norms = }, {mask_fn.__name__ = }, {threshold = }")

        images, labels = load_batches(edit_dataset, num_batches)
        
        try:
            edited_model = your_edit(
                deepcopy(model),
                inputs=images,
                labels=labels,
                layer=layer,
                multi_layer=multi_layer,
                partial_layer=partial_layer,
                lb=lb,
                ub=ub,
                norms=norms,
                mask_fn=mask_fn,
                threshold=threshold,
                print_config=False
            )
        except Exception as e:
            print(f"❌ your_edit failed: {e}")
            print("=" * 80)
            raise optuna.TrialPruned()

        if edited_model is None:
            print("=" * 80)
            raise optuna.TrialPruned()

        # with suppress_stdout():
        edit_acc = test(edited_model, edit_dataset)
        test_acc = test(edited_model, load_testset())

        print(f"{edit_acc = }")
        print(f"{test_acc = }")
        print("=" * 80)

        if test_acc < 0.9:
            raise optuna.TrialPruned()

        trial.set_user_attr("model", edited_model)
        trial.set_user_attr("result", {
            "status": "success",
            "num_batches": num_batches,
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
        })

        return edit_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=max_iteration)

    best_trial = study.best_trial
    best_model = best_trial.user_attrs["model"]
    best_result = best_trial.user_attrs["result"]

    os.makedirs("my_model/part2", exist_ok=True)
    with open("my_model/part2/best_config.json", "w") as f:
        json.dump(best_result, f, indent=2)

    with open("my_model/part2/full_results.csv", "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "status", "num_batches", "layer", "multi_layer", "partial_layer",
            "lb", "ub", "norms", "mask_fn", "threshold", "edit_acc", "test_acc"
        ])
        writer.writeheader()
        for trial in study.trials:
            if "result" in trial.user_attrs:
                row = trial.user_attrs["result"]
                row["norms"] = " ".join(row["norms"])
                writer.writerow(row)

    print("\nBest Model Configuration (test_acc ≥ 90%):")
    for k, v in best_result.items():
        print(f"{k}: {' '.join(v) if k == 'norms' else v}")
    print(f"\nBest Edit Accuracy: {best_result['edit_acc']:.4f}")
    print(f"Best Test Accuracy: {best_result['test_acc']:.4f}")

    return best_model



if __name__ == "__main__":

    """ Load the pre-trained model to edit. """
    model = load_model(path='model.pt')

    """ Load the edit dataset and test dataset. """
    edit_dataset = load_edit_dataset_part_2()

    """ TODO: Implement the edit function. The edited_model returned
        by the function should maximize the accuracy on edit_dataset
        while ensuring that the accuracy on the test set is >= 90%.
    """
    edited_model = edit(model, 50)
    # edited_model = your_edit(model, inputs=all_images, 
    #                          labels=all_labels, 
    #                          layer=-7, 
    #                          multi_layer=True,
    #                          partial_layer=False,
    #                          lb=-10., ub=1.,
    #                          norms=['l1', 'l1n'],
    #                          mask_fn=operator.lt,
    #                          threshold=0)

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
