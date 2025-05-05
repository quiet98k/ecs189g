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

model = load_model("./artifacts/edited_model_part2.pt")

edit_dataset = load_edit_dataset_part_2()
test_dataset = load_testset()


edit_acc = test(model, edit_dataset)
test_acc = test(model, test_dataset)

