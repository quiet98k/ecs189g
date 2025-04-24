from pathlib import Path
import torch
from helpers import load_model, load_edit_dataset_part_1, load_edit_dataset_part_2, load_testset, test

artifacts_dir = Path('artifacts')
device = torch.device('cpu')
dtype = torch.float32

@torch.no_grad()
def main():

    testset = load_testset()

    print("==================== Testing Part 1/2 ====================")
    model_path = artifacts_dir / 'edited_model_part1.pt'
    if not model_path.exists():
        print(f"TODO: Model parameters not found at '{model_path}'.\n"
              f"TODO: Please complete `part1.py` and save the model parameters."
        )
        return

    model = load_model(model_path, device=device, dtype=dtype).eval()
    print(f"(ok) Successfully loaded model from '{model_path}'.")

    editset = load_edit_dataset_part_1()
    edit_acc = test(model, editset)
    if edit_acc < 1.:
        print(f"ERROR: Edited model accuracy on the part 1 edit dataset should be 100%. Got {edit_acc:.2%}.")
    else:
        print(f"(ok) Edited model accuracy on the part 1 edit dataset is {edit_acc:.2%}.")

    test_acc = test(model, testset)
    if test_acc < 0.95:
        print(f"ERROR: Edited model accuracy on the MNIST test set should be >= 95%. Got {test_acc:.2%}.")
    else:
        print(f"(ok) Edited model accuracy on the MNIST test set is {test_acc:.2%}.")

    print("==================== Testing Part 2/2 ====================")
    model_path = artifacts_dir / 'edited_model_part2.pt'
    if not model_path.exists():
        print(f"TODO: Model parameters not found at '{model_path}'.\n"
              f"TODO: Please complete `part2.py` and save the model parameters."
        )
        return

    model = load_model(model_path, device=device, dtype=dtype).eval()
    print(f"(ok) Successfully loaded model from '{model_path}'.")

    editset = load_edit_dataset_part_2()
    edit_acc = test(model, editset)
    print(f"(ok) Edited model accuracy on the part 2 edit dataset is {edit_acc:.2%}.")

    test_acc = test(model, testset)
    if test_acc < 0.90:
        print(f"ERROR: Edited model accuracy on the MNIST test set should be >= 90%. Got {test_acc:.2%}.")
    else:
        print(f"(ok) Edited model accuracy on the MNIST test set is {test_acc:.2%}.")


if __name__ == '__main__':
    main()
