import transformers
from datasets import load_dataset
import random
import numpy as np
from tqdm import tqdm

np.random.seed(1111)
random.seed(1111)

dataset1 = load_dataset("wmt19", "ru-en", split = "validation")
dataset2 = load_dataset("wmt19", "fr-de", split = "validation")
dataset3 = load_dataset("wmt19", "kk-en", split = "validation")
dataset4 = load_dataset("wmt19", "lt-en", split = "validation")

tot_pairs = []

for idx, d in tqdm(enumerate(dataset1["translation"])):

    langs = list(d.keys())

    rand = random.randint(0, len(dataset1["translation"]) - 1)
    
    if rand == idx:
        while rand != idx:
            rand = random.randint(0, len(dataset1["translation"]) - 1)

    sent = d[langs[0]] + "\t" + d[langs[1]] + "\t" + dataset1["translation"][rand][langs[0]] + "\t" + dataset1["translation"][rand][langs[1]] + "\t" + langs[0] + "\t" + langs[1]
    tot_pairs.append(sent)

for idx, d in tqdm(enumerate(dataset2["translation"])):

    langs = list(d.keys())

    rand = random.randint(0, len(dataset2["translation"]) - 1)

    if rand == idx:
        while rand != idx:
            rand = random.randint(0, len(dataset2["translation"]) - 1)

    sent = d[langs[0]] + "\t" + d[langs[1]] + "\t" + dataset2["translation"][rand][langs[0]] + "\t" + dataset2["translation"][rand][langs[1]] + "\t" + langs[0] + "\t" + langs[1]
    tot_pairs.append(sent)

for idx, d in tqdm(enumerate(dataset3["translation"])):

    langs = list(d.keys())

    rand = random.randint(0, len(dataset3["translation"]) - 1)

    if rand == idx:
        while rand != idx:
            rand = random.randint(0, len(dataset3["translation"]) - 1)

    sent = d[langs[0]] + "\t" + d[langs[1]] + "\t" + dataset3["translation"][rand][langs[0]] + "\t" + dataset3["translation"][rand][langs[1]] + "\t" + langs[0] + "\t" + langs[1]
    tot_pairs.append(sent)

for idx, d in tqdm(enumerate(dataset4["translation"])):

    langs = list(d.keys())

    rand = random.randint(0, len(dataset4["translation"]) - 1)

    if rand == idx:
        while rand != idx:
            rand = random.randint(0, len(dataset4["translation"]) - 1)


    sent = d[langs[0]] + "\t" + d[langs[1]] + "\t" + dataset4["translation"][rand][langs[0]] + "\t" + dataset4["translation"][rand][langs[1]] + "\t" + langs[0] + "\t" + langs[1]
    tot_pairs.append(sent)

with open("/home/bhanuv/projects/multilingual_text.txt", "w") as f:
    for idx, t in enumerate(tot_pairs):
        if idx + 1 < len(tot_pairs):
            f.write(t + "\n")
        else:
            f.write(t)

print(len(tot_pairs))