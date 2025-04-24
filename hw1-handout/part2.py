import torch
from torch import Tensor
import sytorch as st
from helpers import load_testset, load_edit_dataset_part_2, load_model, save_model, test
from copy import deepcopy

def edit(model, *args, **kwargs):
    raise NotImplementedError(
        "Editing method for `part2.py` not implemented."
    )

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
