from pathlib import Path
import torch
import torch.nn.functional as F
from helpers import load_model, load_dataset, AdversarialExamples

""" ----------------------------
    DO NOT CHANGE THIS FILE.
    ----------------------------
"""

artifacts_dir = Path('artifacts')
device = torch.device('cpu')
dtype = torch.float32

@torch.no_grad()
def test_part1() -> bool:
    model = load_model(path='model.pt')

    dataset = load_dataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)
    inputs, targets = next(iter(dataloader))

    n_steps = 50
    baselines = torch.zeros_like(inputs)

    from captum.attr import IntegratedGradients
    compute_ig = IntegratedGradients(model, multiply_by_inputs=True)
    expected_ig = compute_ig.attribute(inputs, baselines=baselines, target=targets, n_steps=n_steps,
                                       method='riemann_left')

    from part1 import integrated_gradients
    computed_ig = integrated_gradients(model, inputs, baselines, targets=targets, n_steps=n_steps)

    assert computed_ig.allclose(expected_ig), \
        "(ERROR) Integrated Gradients test failed. "

    print("(ok) Integrated Gradients test passed.")

@torch.no_grad()
def test_part2():
    from captum.attr import IntegratedGradients
    model = load_model(path='model.pt')
    compute_ig = IntegratedGradients(model, multiply_by_inputs=True)

    adv_example = AdversarialExamples.load(path='artifacts/adv_examples.pt')
    images = adv_example.images
    adv_images = adv_example.adv_images
    targets = adv_example.targets

    orig_ig = compute_ig.attribute(images, target=targets, n_steps=50, method='riemann_left')
    adv_ig = compute_ig.attribute(adv_images, target=targets, n_steps=50, method='riemann_left')
    cos = F.cosine_similarity(orig_ig, adv_ig, dim=-1).item()
    assert cos <= 0.5, \
        f"(ERROR) Cosine similarity {cos:.2%} > 50%. This means the adversarial example is not valid."
    print(f"(ok) Cosine similarity {cos:.2%} <= 50%.")

if __name__ == "__main__":
    test_part1()
    test_part2()
