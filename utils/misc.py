import random
import sys
import tqdm
from collections import defaultdict, Counter
import numpy as np

import torch

def reset_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if(args.gpu_id != -1 and args.device > 0 and torch.cuda.is_available()):
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

def read_vocabfile(VOCABPATH):
    vocab = defaultdict(lambda: len(vocab))
    vocab["<unk>"]
    with open(VOCABPATH) as f:
        for line in f:
            token, _ = line.split("\t")
            vocab[token.strip().lower()]
            vocab[token.strip()]
    vocab.default_factory = None
    return vocab


def tokeninds2text(token_inds, vocab):
    index2token = {v: k for k, v in vocab.items()}
    text = [index2token[token_id] for token_id in token_inds]
    return text


def get_class_info(train_data, test_data, dev_data):
    train_c = Counter()
    test_c = Counter()
    dev_c = Counter()
    all_c = Counter()

    for x, t in train_data:
        train_c.update(t.tolist())
    for x, t in test_data:
        test_c.update(t.tolist())
    for x, t in dev_data:
        dev_c.update(t.tolist())

    all_c.update(train_c.elements())
    all_c.update(test_c.elements())
    all_c.update(dev_c.elements())

    sys.stderr.write("train class count: {}\n".format(
        "\t".join(["{}:{}".format(cls, count) for cls, count in sorted(train_c.items(), key=lambda x:x[0])])))
    sys.stderr.write("test class count: {}\n".format(
        "\t".join(["{}:{}".format(cls, count) for cls, count in sorted(test_c.items(), key=lambda x:x[0])])))
    sys.stderr.write("dev class count: {}\n\n".format(
        "\t".join(["{}:{}".format(cls, count) for cls, count in sorted(dev_c.items(), key=lambda x:x[0])])))

    classweight = {}
    normalize = sum([1/c for cl, c in all_c.items() if cl != -1])

    for cl, c in all_c.items():
        if cl != -1:
            classweight[cl] = 1 / (c*normalize)

    classweight_list = [weight for cl, weight in sorted(
        classweight.items(), key=lambda x:x[0])]

    print(classweight_list)

    return classweight_list


def test_essay_data(essay_max_n_dict):
    matrix = xp.zeros((28, 28), dtype=xp.int32)
    matrix = matrix - xp.eye(28, dtype=xp.int32)
    matrix[11:, :] = -1
    matrix[:, 11:] = -1
    matrix[1, 0] = 1
    matrix[2, 0] = 1
    matrix[3, 0] = 1
    matrix[4, 5] = 1
    matrix[7, 6] = 1
    matrix[9, 8] = 2
    matrix = matrix.flatten()
    assert len(essay_max_n_dict) == 402
    assert essay_max_n_dict[398]["relation_matrix"].tolist() == matrix.tolist()


def load_glove(path, vocab, embedding_dim):
    sys.stderr.write("Loading glove embeddings...\n")
    embedding_matrix = xp.zeros((len(vocab), embedding_dim)).astype(xp.float32)
    with open(path, "r") as f:
        c = 0
        for line in tqdm.tqdm(f):
            line_list = line.strip().split(" ")
            word = line_list[0]
            if word in vocab:
                vec = xp.array(line_list[1::], dtype=xp.float32)
                embedding_matrix[vocab[word]] = vec
                c += 1
    sys.stderr.write("read {} token embeddings\n".format(c))
    return embedding_matrix


def count_relations(data, max_n_spans):
    ts = [t[-3][:max_n_spans] for t in data]
    t_all = []
    ts = np.vstack(ts)
    mask = ts > -1
    t_len = np.sum(mask, axis=-1).astype(np.int32)
    t_section = np.cumsum(t_len)
    ts = ts[mask]

    start = 0
    for i, end in enumerate(t_section):
        t = np.array(ts[start:end])
        n = len(t)
        t_root_row = np.where(t == max_n_spans)

        t[t == max_n_spans] = 0

        eye = np.identity(n).astype(np.int32)
        t = eye[t]

        t[t_root_row] = np.zeros(n)

        t[np.identity(n).astype(np.bool)] = -1
        t = t[t > -1]

        t = t.flatten().tolist()
        t_all.extend(t)

        start = end

    n_links = sum(t_all)
    n_no_links = len(t_all) - sum(t_all)
    return n_links, n_no_links
