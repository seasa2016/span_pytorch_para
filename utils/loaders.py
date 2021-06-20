import random
import os
from collections import defaultdict
import itertools
import sys
import json
import numpy as np

from .misc import count_relations
import h5py

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import TensorDataset


def read_vocabfile(VOCABFILE):
    token2index = defaultdict(lambda: len(token2index))
    token2index["<unk>"]
    with open(VOCABFILE) as f:
        for line in f:
            token, *_ = line.split("\t")
            token2index[token]
    return token2index


def relation_info2relation_matrix(ac_relations, max_n_spans, n_spans, settings):

    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        ndarray: flatten version of relation matrix
                                 (axis0: source_AC_id,
                                  axis1: target_AC_id,
                                  value: relation type)
    """

    relation_matrix = np.zeros((max_n_spans, max_n_spans)).astype('int32')
    relation_matrix.fill(-1)
    relation_matrix[n_spans:, :] = -1
    relation_matrix[:, n_spans:] = -1
    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_type = combination[2]
        relation_matrix[source_i, target_i] = relation_type

    return relation_matrix.flatten().astype('int32')


def relation_info2target_sequence(ac_relations, ac_types, max_n_spans, n_spans, settings):
    """Summary line.

    Args:
        arg1 (list): list of argumentative information tuple (source, target, relation_type)

    Returns:
        array: array of target ac index
    """

    relation_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    direction_seq = np.zeros(max_n_spans).astype('int32')
    depth_seq = np.zeros(max_n_spans).astype('int32')

    relation_seq.fill(max_n_spans)
    relation_type_seq.fill(2)
    direction_seq.fill(2)
    depth_seq.fill(100)
    relation_seq[n_spans:] = -1
    relation_type_seq[n_spans:] = -1
    direction_seq[n_spans:] = -1
    depth_seq[n_spans:] = -1

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        relation_seq[source_i] = target_i
        relation_type_seq[source_i] = combination[2]

    for i in range(len(relation_seq)):
        depth = 0
        target_i = relation_seq[int(i)]
        if target_i == -1:
            continue
        while(1):
            if target_i == max_n_spans:
                break
            else:
                target_i = relation_seq[int(target_i)]
                depth += 1
        depth_seq[i] = depth

    return relation_seq, relation_type_seq, depth_seq


def relation_info2children_sequence(ac_relations, max_n_spans, n_spans, settings):
    children_list = [[] for _ in range(max_n_spans + 1)]

    for combination in ac_relations:
        source_i = combination[0]
        target_i = combination[1]
        children_list[target_i].append(source_i)
    return children_list


def get_shell_lang_span(start, text, vocab, previous_span_end):

    EOS_tokens_list = [".",
                       "!",
                       "?",
                       "</AC>",
                       "</para-intro>",
                       "</para-body>",
                       "</para-conclusion>",
                       "</essay>"]

    EOS_ids_set = set([vocab[token.lower()]
                       for token in EOS_tokens_list if token.lower() in vocab])
    shell_lang = []
    if start == 0:
        shell_span = (start, start)
        return shell_span

    for i in range(start-1, previous_span_end, -1):
        if text[int(i)] not in EOS_ids_set:
            shell_lang.append(int(i))
        else:
            break
    if shell_lang:
        shell_start = min(shell_lang)
        shell_end = max(shell_lang)
        shell_span = (shell_start, shell_end)
    else:
        shell_span = (start-1, start-1)
    return shell_span


def get_essay_detail(essay_lines, max_n_spans, vocab, settings):

    essay_id = int(essay_lines[0].split("\t")[0])

    # list of (start_idx, end_idx) of each ac span
    ac_spans = []

    # text of each span
    ac_texts = []

    # type of each span (premise, claim, majorclaim)
    ac_types = []

    # in which type of paragraph each ac is (opening, body, ending)
    ac_paratypes = []

    # id of the paragraph where the ac appears
    ac_paras = []

    # id of each ac (in paragraoh)
    ac_positions_in_para = []

    # linked acs (source_ac, target_ac, relation_type)
    ac_relations = []

    # list of (startr_idx, end_idx) of each am span
    shell_spans = []

    relation2id = {"Support": 0, "Attack": 1}
    actype2id = {"Premise": 0, "Claim": 1, "Claim:For": 1, "Claim:Against": 1, "MajorClaim": 2}
    paratype2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    relation_type_seq = np.zeros(max_n_spans).astype('int32')
    relation_type_seq.fill(2)

    text = [vocab[line.strip().split("\t")[-1].lower()]
            if line.strip().split("\t")[-1].lower() in vocab
            else vocab["<UNK>".lower()]
            for line in essay_lines]

    previous_span_end = 0
    for ac_type, lines in itertools.groupby(essay_lines, key=lambda x: x.split("\t")[6]):
        ac_lines = list(lines)

        if ac_lines[0].split("\t")[7] != "-":
            ac_text = [ac_line.split("\t")[-1].strip() for ac_line in ac_lines]
            ac_texts.append(ac_text)

            para_i = int(ac_lines[0].split("\t")[2])
            para_type = ac_lines[0].split("\t")[3]
            ac_i = int(ac_lines[0].split("\t")[6])
            ac_type = ac_lines[0].split("\t")[7]
            start = int(ac_lines[0].split("\t")[11])
            end = int(ac_lines[-1].split("\t")[11])

            ac_positions_in_para.append(ac_i)
            ac_types.append(actype2id[ac_type])
            ac_paratypes.append(paratype2id[para_type])
            ac_paras.append(para_i)

            ac_span = (start, end)
            ac_spans.append(ac_span)

            shell_span = get_shell_lang_span(start, text, vocab, previous_span_end)
            shell_spans.append(shell_span)

            if ac_type == "Claim:For":
                relation_type_seq[ac_i] = 0
            elif ac_type == "Claim:Against":
                relation_type_seq[ac_i] = 1

            if "Claim" not in ac_lines[0].split("\t")[7]:
                ac_relations.append(
                   (ac_i,
                    ac_i + int(ac_lines[0].split("\t")[8]),
                    relation2id[ac_lines[0].split("\t")[9].strip()]))
                relation_type_seq[ac_i] = relation2id[ac_lines[0].split("\t")[9].strip()]
            previous_span_end = end

    assert len(ac_spans) == len(ac_positions_in_para)
    assert len(ac_spans) == len(ac_types)
    assert len(ac_spans) == len(ac_paratypes)
    assert len(ac_spans) == len(ac_paras)
    assert len(ac_spans) == len(shell_spans)
    assert len(relation_type_seq) == max_n_spans

    assert max(relation_type_seq).tolist() <= 2
    assert len(ac_spans) >= len(ac_relations)

    n_acs = len(ac_spans)
    relation_type_seq[n_acs:] = -1

    relation_matrix = relation_info2relation_matrix(ac_relations,
                                                    max_n_spans,
                                                    n_acs,
                                                    settings)

    assert len(relation_matrix) == max_n_spans*max_n_spans

    relation_targets, _, relation_depth = \
        relation_info2target_sequence(ac_relations,
                                      ac_types,
                                      max_n_spans,
                                      n_acs,
                                      settings)

    assert len(relation_targets) == max_n_spans
    assert len(relation_depth) == max_n_spans

    relation_children = relation_info2children_sequence(ac_relations,
                                                        max_n_spans,
                                                        n_acs,
                                                        settings)

    ac_position_info = np.array([
                       ac_positions_in_para,
                       [(i_ac - max(ac_positions_in_para))*(-1)+max_n_spans
                        for i_ac in ac_positions_in_para],
                       [i+2*max_n_spans for i in ac_paratypes]], dtype=np.int32).T

    assert ac_position_info.shape == (n_acs, 3)

    if not len(ac_position_info):
        para_type = ac_lines[0].split("\t")[3]
        ac_position_info = np.array([[0,
                                      0 + max_n_spans, paratype2id[para_type] + max_n_spans*2]],
                                    dtype=np.int32)

    essay_detail_dict = {}
    essay_detail_dict["essay_id"] = essay_id
    essay_detail_dict["text"] = np.array(text, dtype=np.int32)
    essay_detail_dict["ac_spans"] = np.array(ac_spans, dtype=np.int32)
    essay_detail_dict["shell_spans"] = np.array(shell_spans, dtype=np.int32)
    essay_detail_dict["ac_types"] = np.pad(ac_types,
                                           [0, max_n_spans-len(ac_types)],
                                           'constant',
                                           constant_values=(-1, -1))
    essay_detail_dict["ac_paratypes"] = ac_paratypes
    essay_detail_dict["ac_paras"] = ac_paras
    essay_detail_dict["ac_position_info"] = ac_position_info
    essay_detail_dict["relation_matrix"] = relation_matrix
    essay_detail_dict["relation_targets"] = relation_targets
    essay_detail_dict["relation_children"] = relation_children
    essay_detail_dict["ac_relation_types"] = relation_type_seq
    essay_detail_dict["ac_relation_depth"] = relation_depth

    return essay_detail_dict


def get_essay_info_dict(FILENAME, vocab, settings):
    with open(FILENAME) as f:
        n_span_para = []
        n_span_essay = []
        n_para = []
        span_index_column = 6
        for line in f:
            if line.split("\t")[span_index_column] != "-" \
                    and line.split("\t")[5] != "AC_id_in_essay" \
                    and line.split("\t")[6] != "AC_id_in_paragraph":
                n_span_essay.append(int(line.split("\t")[5]))
                n_span_para.append(int(line.split("\t")[6]))
                n_para.append(int(line.split("\t")[2]))

    max_n_spans = max(n_span_para) + 1
    max_n_paras = max(n_para) + 1

    essay_info_dict = {}
    split_column = 1

    essay2parainfo = defaultdict(dict)
    essay2paraids = defaultdict(list)
    para2essayid = dict()

    with open(FILENAME) as f:
        for essay_id, lines in itertools.groupby(f, key=lambda x: x.split("\t")[split_column]):

            if essay_id == "Essay_id" or essay_id == "Paragraph_id" or essay_id == "-":
                continue

            essay_lines = list(lines)
            para_type = essay_lines[0].split("\t")[3]
            essay_id = int(essay_lines[0].split("\t")[0])
            para_id = int(essay_lines[0].split("\t")[1])

            para2essayid[para_id] = essay_id
            essay2paraids[essay_id].append(para_id)
            essay2parainfo[essay_id][para_type] = para_id

            essay_info_dict[int(para_id)] = get_essay_detail(essay_lines,
                                                             max_n_spans,
                                                             vocab,
                                                             settings)

    max_n_tokens = max([len(essay_info_dict[essay_id]["text"])
                        for essay_id in range(len(essay_info_dict))])

    essay_max_n_dict = {}
    essay_max_n_dict["max_n_spans_para"] = max_n_spans
    essay_max_n_dict["max_n_paras"] = max_n_paras
    essay_max_n_dict["max_n_tokens"] = max_n_tokens

    para_info_dict = defaultdict(dict)
    for para_id, essay_id in para2essayid.items():
        para_info_dict[para_id]["prompt"] = essay2parainfo[essay_id]["prompt"]
        if settings.dataset == "PE":
            para_info_dict[para_id]["intro"] = essay2parainfo[essay_id]["intro"]
            para_info_dict[para_id]["conclusion"] = essay2parainfo[essay_id]["conclusion"]
            para_info_dict[para_id]["context"] = essay2paraids[essay_id]

    return essay_info_dict, essay_max_n_dict, para_info_dict


def get_data_dicts(vocab, args):

    SCRIPT_PATH = os.path.dirname(__file__)
    DATA_PATH = os.path.join(SCRIPT_PATH,
                             "../../work/PE_data.tsv")

    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH,
                                                                            vocab,
                                                                            args)

    return essay_info_dict, essay_max_n_dict, para_info_dict


def return_train_dev_test_ids_PE(vocab, essay_info_dict, essay_max_n_dict,
                                 para_info_dict, args, dev_shuffle=True):

    SCRIPT_PATH = os.path.dirname(__file__)
    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]
    max_n_paras = essay_max_n_dict["max_n_paras"]
    max_n_tokens = essay_max_n_dict["max_n_tokens"]

    invalid_para_inds = set([para_i for para_i, info in essay_info_dict.items()
                             if isinstance(info, dict) and len(info["ac_spans"]) < 1])

    sys.stderr.write("max_n_spans_para: {}\tmax_n_paras: {}\tmax_n_tokens: {}\t"\
                     .format(max_n_spans_para,
                             max_n_paras,
                             max_n_tokens))
    sys.stderr.write("n_vocab: {}\n".format(len(vocab)))

    data = {}
    for key in ['train', 'dev', 'test']:
        temp = json.load(open(os.path.join(SCRIPT_PATH, "../../work/{}_paragraph_index.json".format(key))))
        data[key] = []
        for _ in temp:
            if(_[1] not in invalid_para_inds):
                data[key].append(tuple(_))

    train_inds, test_inds, dev_inds = data['train'], data['dev'], data['test']

    n_trains = len(train_inds)
    print(train_inds[:3])
    if(dev_shuffle):
        all_train_inds = train_inds + dev_inds
        random.seed(args.seed)
        random.shuffle(all_train_inds)
        train_inds = all_train_inds[:n_trains]
        dev_inds = all_train_inds[n_trains:]

    assert set(train_inds) & set(test_inds) & set(dev_inds) == set()

    return train_inds, dev_inds, test_inds


def return_train_dev_test_iter_PE(train_inds, dev_inds, test_inds,
                                  essay_info_dict, essay_max_n_dict, args):

    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]

    train_data = load_data(train_inds, essay_info_dict, max_n_spans_para, args)
    test_data = load_data(test_inds, essay_info_dict, max_n_spans_para, args)
    dev_data = load_data(dev_inds, essay_info_dict, max_n_spans_para, args)

    train_n_links, train_n_no_links = count_relations(train_data, max_n_spans_para)
    dev_n_links, dev_n_no_links = count_relations(dev_data, max_n_spans_para)
    test_n_links, test_n_no_links = count_relations(test_data, max_n_spans_para)

    return train_data, dev_data, test_data

def return_train_dev_test_iter_CMV(train_inds, dev_inds, test_inds, data_path, args):
    data, adu_max = load_data_cmv(data_path, args)

    inds = []
    for _ in [train_inds, dev_inds, test_inds]:
        inds.append(torch.tensor(_, dtype=torch.long))

    split_data = []
    for ind in inds:
        split_data.append([_[ind] for _ in data])

    return split_data[0], split_data[1], split_data[2], adu_max


def return_train_dev_test_iter_MT(vocab, args,
                                  iteration_i=0, fold_i=0):

    SCRIPT_PATH = os.path.dirname(__file__)

    # change
    DATA_PATH = os.path.join(SCRIPT_PATH,
                             "../../work/MT_data.tsv")
    essay_info_dict, essay_max_n_dict, para_info_dict = get_essay_info_dict(DATA_PATH,
                                                                            vocab,
                                                                            args)

    max_n_spans_para = essay_max_n_dict["max_n_spans_para"]
    max_n_paras = essay_max_n_dict["max_n_paras"]
    max_n_tokens = essay_max_n_dict["max_n_tokens"]

    sys.stderr.write("max_n_spans_para: {}\tmax_n_paras: {}\tmax_n_tokens: {}\t"\
                     .format(max_n_spans_para,
                             max_n_paras,
                             max_n_tokens))
    sys.stderr.write("n_vocab: {}\n".format(len(vocab)))

    fold_tids = json.load(open(os.path.join(SCRIPT_PATH,
                                            "../../work/folds_author.json")))
    all_train_inds = fold_tids[str(iteration_i)][fold_i][0]
    n_trains = len(all_train_inds)

    train_inds = all_train_inds[:int(n_trains*0.9)]
    dev_inds = all_train_inds[int(n_trains*0.9):]
    test_inds = fold_tids[str(iteration_i)][fold_i][1]

    sys.stderr.write("train_inds: {}\n".format(",".join([str(index)
                                                         for index in train_inds])))
    sys.stderr.write("total_train_essays: {}\n".format(len(train_inds)))

    sys.stderr.write("test inds: {}\n".format(",".join([str(index)
                                                        for index in test_inds])))
    sys.stderr.write("total_test_essays: {}\n".format(len(test_inds)))

    sys.stderr.write("dev inds: {}\n".format(",".join([str(index)
                                                       for index in dev_inds])))
    sys.stderr.write("total_dev_essays: {}\n\n".format(len(dev_inds)))

    assert set(train_inds) & set(test_inds) & set(dev_inds) == set()
    assert len(train_inds) + len(test_inds) + len(dev_inds) == 112


    train_data = load_data(train_inds, essay_info_dict, max_n_spans_para, args)
    test_data = load_data(test_inds, essay_info_dict, max_n_spans_para, args)
    dev_data = load_data(dev_inds, essay_info_dict, max_n_spans_para, args)

    train_n_links, train_n_no_links = count_relations(train_data, max_n_spans_para)
    dev_n_links, dev_n_no_links = count_relations(dev_data, max_n_spans_para)
    test_n_links, test_n_no_links = count_relations(test_data, max_n_spans_para)

    print("n_links: ", str(train_n_links+dev_n_links+test_n_links))
    return train_data, dev_data, test_data, essay_info_dict, essay_max_n_dict, para_info_dict


def load_data(essay_ids, essay_info_dict, max_n_spans, args):
    ts_link = torch.tensor([essay_info_dict[int(i)]["relation_targets"]
                        for _, i in list(essay_ids)], dtype=torch.long)
    ts_type = torch.tensor([essay_info_dict[int(i)]["ac_types"]
                        for _, i in list(essay_ids)], dtype=torch.long)
    ts_link_type = torch.tensor([essay_info_dict[int(i)]["ac_relation_types"]
                             for _, i in list(essay_ids)], dtype=torch.long)

    ################
    # data loading #
    ################
    xs = [essay_info_dict[int(i)]["text"] for _, i in essay_ids]

    x_spans = [essay_info_dict[int(i)]["ac_spans"] for _, i in essay_ids]
    x_spans = [spans if len(spans) else np.array([[1, len(xs[i])-2]],dtype=np.int32)
                for i, spans in enumerate(x_spans)]

    shell_spans = [essay_info_dict[int(i)]["shell_spans"] for _, i in essay_ids]
    shell_spans = [spans if len(spans) else np.array([[1, len(xs[i])-2]],                                                            dtype=np.int32) for i, spans in enumerate(shell_spans)]

    #(batchsize, max_n_spans, (ac id in essay, ac id in paragraph, paragraph id))
    x_position_info = [essay_info_dict[int(i)]["ac_position_info"] for _, i in essay_ids]

    assert len(xs) == len(x_spans)
    assert len(xs) == len(shell_spans)
    assert x_spans[0][0][1] >= x_spans[0][0][0]
    assert shell_spans[0][0][1] >= shell_spans[0][0][0]
    assert len(x_position_info[0][0]) == 3

    ################
    # padding data #
    ################
    xs_len = [ len(_) for _ in xs]
    adu_len = [ len(_) for _ in x_spans]

    max_xs_len, max_adu_len = max(xs_len), 12
    xs = torch.tensor(
        [ (_.tolist()+[0]*(max_xs_len-xs_len[i])) for i, _ in enumerate(xs)]
    )
    x_spans = torch.tensor(
        [ (_.tolist()+[[0, 0]]*(max_adu_len-adu_len[i])) for i, _ in enumerate(x_spans)]
    )
    shell_spans = torch.tensor(
        [ (_.tolist()+[[0, 0]]*(max_adu_len-adu_len[i])) for i, _ in enumerate(shell_spans)]
    )
    x_position_info = torch.tensor(
        [ (_.tolist()+[[0, 0, 0]]*(max_adu_len-adu_len[i])) for i, _ in enumerate(x_position_info)]
    )

    print('xs', xs.shape)
    print('x_spans', x_spans.shape)
    print('shell_spans', shell_spans.shape)
    print('x_position_info', x_position_info.shape)

    print('finish building position information')
    #########################
    # handle elmo embedding #
    #########################
    elmo_embed = h5py.File(args.elmo_path, 'r')
    elmo_embeddings = [elmo_embed.get(str(i))[()] for _, i in essay_ids]
    elmo_len = [_.shape[1] for _ in elmo_embeddings]
    max_elmo_len = max(elmo_len)
    hidden_dim = elmo_embeddings[0].shape[-1]

    elmo_embeddings = torch.tensor([
        np.concatenate([_, np.zeros([3, max_elmo_len-elmo_len[i], hidden_dim], np.float)], axis=1)
        for i, _ in enumerate(elmo_embeddings)
    ], dtype=torch.float)

    topic_embeddings = [elmo_embed.get(str(_))[()] for _, i in essay_ids]
    topic_len = [_.shape[1] for _ in topic_embeddings]
    max_topic_len = max(topic_len)
    hidden_dim = topic_embeddings[0].shape[-1]

    topic_embeddings = torch.tensor([
        np.concatenate([_, np.zeros([3, max_topic_len-topic_len[i], hidden_dim], np.float)], axis=1)
        for i, _ in enumerate(topic_embeddings)
    ], dtype=torch.float)

    xs_len = torch.tensor(xs_len, dtype = torch.long)
    adu_len = torch.tensor(adu_len, dtype = torch.long)
    topic_len = torch.tensor(topic_len, dtype = torch.long)

    return (xs, x_spans, shell_spans, x_position_info,
            elmo_embeddings, topic_embeddings,
            xs_len, adu_len, topic_len,
            ts_link, ts_type, ts_link_type)


def load_data_cmv(data_path, args):
    # load data from json file and add up elmo_embedding
    with open(data_path) as f:
        data = [json.loads(line) for line in f]
    """
    {
        'topic_index':topic_elmo_index,
        'elmo_index':elmo_index,
        'shell_span':shell_span,
        'span':span,
        'adu_label':adu_label,
        'mask':mask,
        'rel_label':rel_label,
        'ref_label':ref_label,
        'ac_position_info':ac_position_info
    }
    """

    # share adu parameter
    adu_len = [ len(_['shell_span']) for _ in data]
    adu_max = max( adu_len )
    """
    for post in data:
        print(len(post['shell_span']), len(post['elmo_index']))
    """
    ##############################
    # handle span and shell span #
    ##############################
    shell_spans = torch.tensor([
        post['shell_span'] + [[0, 0, 0]]*(adu_max-adu_len[i]) for i, post in enumerate(data)
    ])
    spans = torch.tensor([
        post['span'] + [[0, 0, 0]]*(adu_max-adu_len[i]) for i, post in enumerate(data)
    ])

    ################
    # handle label #
    ################
    shell_span = torch.tensor([
        post['shell_span'] + [[0, 0, 0]]*(adu_max-adu_len[i]) for i, post in enumerate(data)
    ])
    span = torch.tensor([
        post['span'] + [[0, 0, 0]]*(adu_max-adu_len[i]) for i, post in enumerate(data)
    ])

    relation2id = {"Support": 0, "Attack": 1}
    actype2id = {"Premise": 0, "Claim": 1, "Claim:For": 1, "Claim:Against": 1, "MajorClaim": 2}
    paratype2id = {"intro": 0, "body": 1, "conclusion": 2, "prompt": 3}

    mapping={
        'adu':{'P':0, 'C':1, -1:-1},
        'rel':{'support':0, 'attack':1, -1:-1}
    }
    adu_label, rel_label, ref_label = [], [], []
    for i, post in enumerate(data):
        adu_label.append([ mapping['adu'][_] for _ in post['adu_label']])
        rel_label.append([ mapping['rel'][_] for _ in post['rel_label']])

        ref_label.append([])
        for ref in post['ref_label']:
            if(ref=='title'):
                ref = adu_max
            ref_label[-1].append(ref)

        adu_label[-1].extend([-1]*(adu_max-adu_len[i]))
        rel_label[-1].extend([-1]*(adu_max-adu_len[i]))
        ref_label[-1].extend([-1]*(adu_max-adu_len[i]))
    adu_label = torch.tensor(adu_label, dtype=torch.long)
    rel_label = torch.tensor(rel_label, dtype=torch.long)
    ref_label = torch.tensor(ref_label, dtype=torch.long)

    #########################
    # handle elmo embedding #
    #########################
    elmo_embed = h5py.File(args.elmo_path, 'r')
    elmo_embeddings = []
    for post in data:
        temp = []
        for index in post['elmo_index']:
            temp.append(elmo_embed.get(str(index))[()])
        elmo_embeddings.append(temp)

    elmo_len = [ [para_embed.shape[1] for para_embed in post_emb] for post_emb in elmo_embeddings ]
    elmo_max = max( [ _ for q in elmo_len for _ in q] )

    para_len = [ len(_['elmo_index']) for _ in data]
    para_max = max(para_len)
    hidden_dim = elmo_embeddings[0][0].shape[-1]
    print('max', para_max, elmo_max)

    temp_emb = []
    for i, post_emb in enumerate(elmo_embeddings):
        temp = []
        for _ in post_emb:
            temp.append(np.concatenate([_, np.zeros([3, elmo_max-_.shape[1], hidden_dim], np.float)], axis=1))
        temp = torch.tensor(temp, dtype=torch.float)
        temp_emb.append(
            torch.cat([temp, torch.zeros(para_max-para_len[i], 3, elmo_max, hidden_dim, dtype=torch.float)]))
    elmo_embeddings = torch.stack(temp_emb)
    print('finish elmo')

    topic_embeddings = [elmo_embed.get(str(post['topic_index']))[()] for post in data]
    topic_len = [_.shape[1] for _ in topic_embeddings]
    topic_max = max(topic_len)
    hidden_dim = topic_embeddings[0].shape[-1]
    topic_embeddings = torch.tensor([
        np.concatenate([_, np.zeros([3, topic_max-_.shape[1], hidden_dim], np.float)], axis=1)
        for _ in topic_embeddings], dtype=torch.float)

    print('finish topic')

    ########################
    # handle position info #
    ########################
    ac_position_info = torch.tensor([ _['ac_position_info']+[[0,0]]*(adu_max-adu_len[i]) for i, _ in enumerate(data)], dtype=torch.long)


    mask = []
    for i, post in enumerate(data):
        mask.append([])
        for _ in post['mask']:
            mask[-1].append( _+[0]*(adu_max-len(_)))
        mask[-1].extend([[0]*adu_max]*(adu_max-adu_len[i]))
    mask = torch.tensor(mask, dtype=torch.long)

    print('finish mask')

    batch_size, adu_dim = rel_label.shape
    for i in range(batch_size):
        for j in range(adu_dim):
            index = ref_label[i,j].item()
            if(index == -1 or index == adu_max):
                continue
            if(mask[i, j, index].item() == 0 or j==index):
                rel_label[i, j] = -1
                ref_label[i, j] = -1

    print('finish cleaning')

    for i, _ in enumerate(elmo_len):
        _.extend([0]*(para_max-para_len[i]))

    xs_len = torch.tensor(elmo_len, dtype=torch.long)
    adu_len = torch.tensor(adu_len, dtype=torch.long)
    topic_len = torch.tensor(topic_len, dtype=torch.long)

    return (spans, shell_spans, ac_position_info,
            elmo_embeddings, topic_embeddings,
            xs_len, adu_len, topic_len,
            ref_label, adu_label, rel_label,
            mask), adu_max

