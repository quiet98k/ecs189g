
## Preliminaries:
> In this assignment, we will continue use the python environment manager `uv` and
> the gurobi solver, please refer the Homework 0 for instruction.

### Using `sytorch`
In this assignment, we will use a symbolic model editing tool `sytorch` for
editing the pre-trained models. The current version of `sytorch` supports the
following platforms:
- X86-64-based Linux
- ARM-based macOS

If `sytorch` does not support your platform (e.g., Windows), please use a CSIF
machine.

To use `sytorch`in this homework, please
- connect to the UCDavis VPN, and
- put your sytorch license key in the `./sytorch.lic` file. You should have
  received this key via email from the instructors.

## Instructions

Look at the comments in `part1.py` and `part2.py` to understand what is to be
done.

You are NOT allowed to modify `helpers.py`.

You can define utilities in a `utils.py` script, which can be shared by
solutions to both parts of the homework.

You can run the following command to test your solution:
```
uv run test.py
```

## Submit your solution and environment
Please use the following command to zip the `./hw1-submission.zip` file for submission.
```
bash zip_submission.sh
```
`./hw1-submission.zip` should ONLY contain the following files:
```
.python-version
pyproject.toml
uv.lock
part1.py
part2.py
utils.py
artifacts/edited_model_part1.pt
artifacts/edited_model_part2.pt
```
