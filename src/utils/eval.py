import torch
import numpy as np

def get_accuracy(pred, label, threshold=0.5):
    pred = pred > threshold
    label = label == torch.max(label)
    correct = torch.sum(pred == label)
    all = 1
    for i in range(pred.dim()):
        all *= pred.size(i)
    acc = float(correct) / float(all)

    return acc