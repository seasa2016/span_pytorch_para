import numpy as np
from sklearn.metrics import f1_score
import torch


"""
def softmax_cross_entropy_flatten(y, t):
    t = t.flatten()
    ignore_label = -1
    loss = chaFunc.softmax_cross_entropy(y, t,
                                         enable_double_backprop=True,
                                         ignore_label=ignore_label)
    return loss


def accuracy_flatten(y, t):
    t = t.flatten()
    ignore_label = -1
    acc = chaFunc.accuracy(y, t, ignore_label=ignore_label)
    return acc

def classification_summary_flatten(y, t):
    t = t.flatten()
    summary = chaFunc.classification_summary(y, t)
    return summary
"""


def fscore_binary(ys, ts, trc_len, adu_len, max_n_spans):
    def convert(index, l):
        eye = torch.eye(l, dtype=torch.long)

        root_row = (index == max_n_spans)
        index[root_row] = 0

        data = eye[index]
        data[root_row] = torch.zeros(l, dtype=torch.long)
        for i in range(l):
            data[i, i] = -1
        data = data[data > -1]
        return data

    t_all = []
    y_all = []

    start = 0
    for t, l in zip(trc_len, adu_len):
        t_flat = convert(ts[start:start+t], l).view(-1).tolist()
        t_all.extend(t_flat)

        y_flat = convert(ys[start:start+t], l).view(-1).tolist()
        y_all.extend(y_flat)

        start += l

    f1_link = f1_score(t_all, y_all, pos_label=1)
    f1_no_link = f1_score(t_all, y_all, pos_label=0)
    return f1_link, f1_no_link

def count_prediction(y, i):
    return (y == i).sum()
