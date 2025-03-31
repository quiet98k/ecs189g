
## Managing Python environment using [`uv`](https://github.com/astral-sh/uv)
In this course we will use `uv` to manage the Python environment for our
homework.

### Install `uv`

We recommend the following ways to install `uv` for the current user. See the
[installation
documentation](https://docs.astral.sh/uv/getting-started/installation/) for
details and alternative installation methods.

```
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```
# On Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Run a script in our Python environment
Under this directory, please use the `uv run` command to run a script in our
Python environment. For example,
```
uv run part1_training.py
```
See the
[documentation](https://docs.astral.sh/uv/guides/scripts/#running-a-script-with-dependencies)
for details and related commands.

### Default Python environment
We have provided a default Python environment specified by the following configuration files:
- `.python-version` specified the Python version (3.12.8);
- `pyproject.toml` specified the required packages.

Please don't modify them manually. See the next section for instructions to
manage package dependencies. You will submit those configuration files, and we
will grade your submission in an environment reproduced using your submitted
configuration.

### (Optional) Install additional Python packages
Please use the `uv add` command instead of `pip install` to install a Python
package to the environment. For example,
```
uv add numpy
```
This command will maintain all installed packages in the `project.dependencies`
of `pyproject.toml`. See the
[documentation](https://docs.astral.sh/uv/concepts/projects/dependencies/#managing-dependencies)
for details and related commands.


## Part 1: Train a DNN on a rotated MNIST dataset.
In this part, you will implement the `part1_training.py` script so that
```
uv run part1_training.py
```
will train a DNN on a ***rotated*** MNIST dataset. Below are the requirements:
- You are required to use the DNN architecture `Model` defined in
  `helpers.py`. You are NOT allowed to modify its definition.

- You are required to prepare the rotated MNIST dataset so that each image is
  rotated by 90 degree clockwise. You could obtain the original MNIST dataset
  using [torchvision.datasets.MNIST](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

- The accuracy of the trained DNN should above 95\% on the rotated MNIST test set.

- Save the trained model ***parameters*** at `artifacts/model.pt`. For example,
  ```
  model.save('artifacts/model.pt')
  ```

- You can define utilities in a `utils.py` script, which are shared with
  other parts of this homework.

You can run the following command to test that the parameters are saved and loaded properly:
```
uv run test.py
```
Note that this `test.py` script does NOT evaluate your DNN on the rotated MNIST test data;
it is your responsibility to make sure that the test accuracy is greater than 95\%.

## Part 2: Attack your trained DNN to find an adversarial example.
In this part, you will implement the `part2_attacking.py` script so that
```
uv run part2_attacking.py
```
will find an adversarial example for your trained DNN. Specifically,
- Take your trained DNN from part 1, use the `torchattacks` package to find an
  adversarial example for any training image $x$ in the rotated MNIST training
  set, so that the $L_\infty$ distance between the training image and the
  adversarial example (the perturbation) is within `1e-2`, your trained DNN
  correctly classifies the training image but misclassifies the adversarial
  example.

- Save the original image, expected target and the adversarial example at
  `artifacts/adversarial_examples.pt` using `AdversarialExamples.save`
  defined in helpers. For example,
    ```
    AdversarialExamples(images=images,
                        adv_images=adv_images,
                        targets=targets,
    ).save('artifacts/adversarial_examples.pt')
    ```

- You can define utilities in a `utils.py` script, which can be shared with
  other parts of this homework.

You can run the following command to test your saved adversarial example:
```
uv run test.py
```

## Part 3: Editing your trained DNN to correctly classify the adversarial example.

> ### Acquiring a Gurobi Academic License
> In this part you need to use the Gurobi LP solver. Please connect to the UC Davis
> Library or COE VPN and [follow the
> instructions](https://www.gurobi.com/academia/academic-program-and-licenses/) to
> acquire an **Academic Named-User License** for your machine. You can download
> the Gurobi licensing tool `grbgetkey` [from
> here](https://support.gurobi.com/hc/en-us/articles/360059842732-How-do-I-set-up-a-license-without-installing-the-full-Gurobi-package).

In this part, you will implement the `part3_editing.py` script so that
```
uv run part3_editing.py
```
will edit the last layer of your trained DNN so that the edited DNN
correctly classifies both the original and adversarial images you found in part 2.
Specifically,

- Take your trained DNN $N$ from Part 1 and the adversarial example you found
   from Part 2, find new weight and bias for the last fully-connected layer
   `model.layers[-1]` using the Gurobi LP solver, so that the edited DNN
   correctly classifies both the original and adversarial images. See
   [Gurobi Python API Reference](https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html)
   for `gurobipy` instructions.

- Save the edited last layer ***parameters*** at `artifacts/edited_last_layer.pt`. For example,
   ```
   model.save_layer(-1, 'artifacts/edited_last_layer.pt')
   ```

- You can define utilities in a `utils.py` script, which can be shared with
  other parts of this homework.

You can run the following command to test your edited DNN:
```
uv run test.py
```

## Submit your solution and environment
Please use the following command to zip the `./hw0-submission.zip` file for submission.
```
bash zip_submission.sh
```
`./hw0-submission.zip` should ONLY contain the following files:
```
.python-version
pyproject.toml
uv.lock
part1_training.py
part2_attacking.py
part3_editing.py
utils.py
artifacts/model.pt
artifacts/adversarial_examples.pt
artifacts/edited_last_layer.pt
```
