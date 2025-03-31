""" Implement this script so that
    ```
    uv run part2_attacking.py
    ```
    will find an adversarial example for your trained DNN from Part 1.

    You could use the following code to load your trained model:
    ```
    from helpers import Model
    model = Model().load('artifacts/model.pt')
    ```

    You could use the following code to save the adversarial example:
    ```
    from helpers import AdversarialExamples
    AdversarialExamples(images=images,
                        adv_images=adv_images,
                        targets=targets,
    ).save('artifacts/adversarial_examples.pt')
    ```

    See README.md for the requirements.
"""
