""" Implement this script so that
    ```
    uv run part3_editing.py
    ```

    will edit the last layer of your trained DNN from Part 1, so that the edited
    DNN correctly classifies both the original and adversarial images you found
    in part 2.

    You could use the following code to load the adversarial example:
    ```
    from helpers import AdversarialExamples
    example = AdversarialExamples.load('artifacts/adversarial_examples.pt')
    ```

    And use the following code to save the edited last layer:
    ```
    model.save_layer(-1, 'artifacts/edited_last_layer.pt')
    ```

    See README.md for the requirements.
"""
