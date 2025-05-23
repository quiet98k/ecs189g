import torch
import torch.nn.functional as F
from helpers import load_dataset, load_model, AdversarialExamples
import torchattacks

def find_adv_examples():
    model = load_model(path='model.pt')
    dataset = load_dataset()

    """
    TODO: In this part, you need to find adversarial example training images in
          the @dataset on @model, so that
          - $L_\infty$ distance between the training image and the adversarial
            example (the perturbation) is within `1e-2`,
          - The integrated gradients of the model output of the correct class
            with respect to the input and the adversarial example are not similar
            (cosine similarity < 0.5). You could use `F.cosine_similarity(...)`.

        Use your implementation of integrated gradients from part1.py to compute the
        integrated gradients with
        - `n_steps=50` and
        - an all-zero baseline.
        You are not allowed to use other libraries to compute the integrated gradients.

        Save the original image, expected target and the adversarial example as
        'artifacts/adv_examples.pt' using the following code.
    """

    AdversarialExamples(images=images,
                        adv_images=adv_images,
                        targets=targets,
    ).save('artifacts/adv_examples.pt')

if __name__ == "__main__":
    find_adv_examples()
