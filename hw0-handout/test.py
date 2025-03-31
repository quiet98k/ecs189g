from pathlib import Path
import torch
from helpers import Model, AdversarialExamples

artifacts_dir = Path('artifacts')
model_path = artifacts_dir / 'model.pt'
adversarial_examples_path = artifacts_dir / 'adversarial_examples.pt'
edit_last_layer_path = artifacts_dir / 'edited_last_layer.pt'
device = torch.device('cpu')
dtype = torch.float32

@torch.no_grad()
def test():
    print("==================== Testing Part 1/3 ====================")
    if not model_path.exists():
        print(f"TODO: Model parameters not found at '{model_path}'.\n"
              f"TODO: Please complete `part1_training.py` and save the model parameters."
        )
        return

    model = Model().load(model_path).to(device=device,dtype=dtype).eval()
    print(f"(ok) Successfully loaded model from '{model_path}'.")
    print("NOTE: Please ensure that the accuracy of your trained DNN is above 95% on the rotated MNIST test set.")


    print("==================== Testing Part 2/3 ====================")
    if not adversarial_examples_path.exists():
        print(f"TODO: Adversarial examples not found at '{adversarial_examples_path}'.\n"
              f"TODO: Please complete `part2_attacking.py` and save the adversarial examples."
        )
        return

    examples = AdversarialExamples.load(adversarial_examples_path).to(device=device,dtype=dtype)
    print(f"(ok) Successfully loaded adversarial examples from '{adversarial_examples_path}'.")

    """ The distance between the original and adversarial images should be within the specified epsilon. """
    dist_linf = (examples.images - examples.adv_images).norm(torch.inf)
    assert dist_linf <= 1e-2 + 1e-5, \
        f"ERROR: Expected |img - adv|_inf <= {1e-2:.2e}, got {dist_linf:.2e}"
    print(f"(ok) |img - adv|_inf == {dist_linf:.2e} <= {1e-2:.2e}")

    """ The original should correctly classify the original images. """
    predicted_targets = model(examples.images).argmax(-1)
    assert (predicted_targets == examples.targets).all(), \
        f"ERROR: Expected model to correctly classify the original image as {examples.targets[0]}, " \
        f"got {predicted_targets[0]}"
    print(f"(ok) Model correctly classifies the original image as {examples.targets[0]}.")

    """ The original should misclassify the adversarial images. """
    predicted_adv_targets = model(examples.adv_images).argmax(-1)
    assert (predicted_adv_targets != examples.targets).all(), \
        f"ERROR: Expected model to misclassify the adversarial example, " \
        f"got the correct label {examples.targets[0]}."
    print(f"(ok) Model misclassifies the adversarial example as {predicted_adv_targets[0]}.")


    print("==================== Testing Part 3/3 ====================")
    if not edit_last_layer_path.exists():
        print(f"TODO: Edited last layer parameters not found at '{edit_last_layer_path}'.\n"
              f"TODO: Please complete `part3_editing.py` and save the edited last layer parameters."
        )
        return

    """ Load the edited last layer parameters. """
    model = model.load_layer(-1, edit_last_layer_path).to(device=device,dtype=dtype).eval()
    print(f"(ok) Successfully loaded edited last layer parameters from '{edit_last_layer_path}'.")

    """ The edited model should correctly classify the original images. """
    edited_predicted_targets = model(examples.images).argmax(-1)
    assert (edited_predicted_targets == examples.targets).all(), \
        f"ERROR: Expected edited model to correctly classify the original image as {examples.targets[0]}, " \
        f"got {edited_predicted_targets[0]}"
    print(f"(ok) Edited model correctly classifies the original image as {examples.targets[0]}.")

    """ The edited model should correctly classify the adversarial images. """
    edited_predicted_adv_targets = model(examples.adv_images).argmax(-1)
    assert (edited_predicted_adv_targets == examples.targets).all(), \
        f"ERROR: Expected edited model to correctly classify the adversarial example as {examples.targets[0]}, " \
        f"got {edited_predicted_adv_targets[0]}"
    print(f"(ok) Edited model correctly classifies the adversarial example as {examples.targets[0]}.")
    print("NOTE: Please ensure that the drawdown of your edited DNN is below 1% on the rotated MNIST test set.")

if __name__ == '__main__':
    test()
