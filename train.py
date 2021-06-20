import argparse
import sys
import os
import copy
import numpy as np
import collections

import torch
import torch.optim as optim
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)

from data.dataloader import treeDataset, tree_collate_fn
from model.span_selection_model import SpanSelectionParser
from classifier.parsing_loss import FscoreClassifier

from utils.resource import Resource
from optim import Ranger, RAdam

#from utils.converters import convert_hybrid as convert

from utils.misc import (
    reset_seed, read_vocabfile, load_glove
)
from utils.evaluators import fscore_binary, count_prediction

from utils.option import (
    add_dataset_args, add_default_args, add_embed_args,
    add_log_args, add_model_arch_args, add_optim_args,
    add_trainer_args, post_process_args_info
)
best_threshold = [0, 0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_default_args(parser)
    parser = add_dataset_args(parser)
    parser = add_embed_args(parser)
    parser = add_log_args(parser)
    parser = add_model_arch_args(parser)
    parser = add_optim_args(parser)
    parser = add_trainer_args(parser)

    return parser.parse_args()

def log_info(args):
    #resource = Resource(args, train=True)
    #resource.dump_git_info()
    resource.dump_command_info()
    resource.dump_python_info()
    resource.dump_library_info()
    #resource.save_vocab_file()
    resource.save_config_file()
    outdir = resource._return_output_dir()
    return outdir


def load_embeddings(model, vocab, args):
    if args.use_elmo:
        pass
    else:
        #################
        # glove setting #
        #################

        sys.stderr.write("Loading glove embeddings...\n")
        embedding_matrix = load_glove(args.glove_path,
                                      vocab,
                                      args.eDim)
        if args.device:
            embedding_matrix = xp.asarray(embedding_matrix)

        sys.stderr.write("Setting embed matrix...\n")
        model.predictor.set_embed_matrix(embedding_matrix)

def data_update(total, length, output, label):
    """
    update label and pred
    """
    link_sort = output[1].sort(-1, descending=True)[1].tolist()
    
    output[1] = output[1].max(-1)[1]
    output[2] = output[2].max(-1)[1]
    for i in [3]:
        output[i] = (output[i].sigmoid())

    trc_len, adu_len = length
    total['label']['link'].append( label[0] )
    total['label']['type'].append( label[1] )
    total['label']['link_type'].append( label[2] )

    total['pred']['link_sort'].extend( link_sort )

    total['pred']['link_mst'].append(  output[0].view(-1) )
    total['pred']['link'].append( output[1].view(-1) )
    total['pred']['type'].append( output[2].view(-1) )
    total['pred']['link_type'].append( output[3].view(-1) )
    total['trc_len'].extend( trc_len.tolist() )
    total['adu_len'].extend( adu_len.tolist() )



def print_stat(total_stat, keys):
    for key in keys:
        if(key not in total_stat):
            return
        print('{:16}\t{:.4f}'.format(key, total_stat[key]), end='\t', flush=True)
    print()

def convert(data, device, in_data=True):
    temp = []
    if(in_data):
        for key in ['span', 'shell_span', 'ac_position_info', 'elmo_emb', 'topic_emb', 'author', 'elmo_length', 'adu_length', 'topic_length', 'mask']:
            temp.append(data[key].to(device))
    else:
        for key in ['link', 'type', 'link_type']:
            temp.append(data[key].to(device))

    return temp

def train(model, optimizer, data_iter, device, model_path):
    model.predictor.train()
    model.predictor.zero_grad()

    total = {
        'label':collections.defaultdict(list),
        'pred':collections.defaultdict(list),
        'trc_len':[],
        'adu_len':[]
    }

    total_stat = {}
    for i, (data, label) in enumerate(data_iter):
        x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author, xs_len, adu_len, topic_len, mask = convert(data, device, in_data=True)
        ts_link, ts_type, ts_link_type = convert(label, device, in_data=False)

        loss, stat, output, label = model.train(x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author, xs_len, adu_len, topic_len, mask, ts_link, ts_type, ts_link_type, False)

        loss.backward()

        data_update(total, [(ts_link>=0).sum(-1), adu_len], output, label)

        #nn.utils.clip_grad_norm_(model.parameters(), 8)
        optimizer.step()
        model.predictor.zero_grad()
        total_stat.update(stat)
        if(i%2==0):
            for key, _ in total_stat.items():
                if('loss' in key):
                    print(key, _, end='\t')
            print(flush=True)
            
        
    stat = model.evaluate(total, best_threshold=best_threshold)
    print_stat(stat, ['macro_f_type', 'f_premise', 'f_claim'])
    print_stat(stat, ['macro_f_mst_link', 'f_mst_link', 'f_mst_nolink'])
    print_stat(stat, ['macro_f_link', 'f_link', 'f_nolink'])
    print_stat(stat, ['macro_f_link_type', 'f_support', 'f_attack'])
    print_stat(stat, ['mrr_link'])
    print_stat(stat, ['total_macro_f'])

    torch.save(model.predictor.state_dict(), model_path)

def evaluation(model, data_iter, device, mode='dev'):
    model.predictor.eval()

    total = {
        'label':collections.defaultdict(list),
        'pred':collections.defaultdict(list),
        'trc_len':[],
        'adu_len':[]
    }

    total_stat = collections.defaultdict(float)
    for i, (data, label) in enumerate(data_iter):
        x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author, xs_len, adu_len, topic_len, mask = convert(data, device, in_data=True)
        ts_link, ts_type, ts_link_type = convert(label, device, in_data=False)

        loss, stat, output, label = model.train(x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author, xs_len, adu_len, topic_len, mask, ts_link, ts_type, ts_link_type, False)

        data_update(total, [(ts_link>=0).sum(-1), adu_len], output, label)

        for key, _ in stat.items():
            total_stat[key] += _

    stat = model.evaluate(total, best_threshold=best_threshold, mode=mode)
    total_stat.update(stat)

    print_stat(total_stat, ['macro_f_type', 'f_premise', 'f_claim'])
    print_stat(total_stat, ['macro_f_mst_link', 'f_mst_link', 'f_mst_nolink'])
    print_stat(total_stat, ['macro_f_link', 'f_link', 'f_nolink'])
    print_stat(total_stat, ['macro_f_link_type', 'f_support', 'f_attack'])
    print_stat(stat, ['mrr_link'])
    print_stat(total_stat, ['total_macro_f'])
    if(mode=='dev'):
        print('best threshold', best_threshold)

def setting(args):
    print("build up dataloader")
    max_n_spans_para = args.max_n_spans_para
    
    tree = []
    num_worker, batch_size = 0, args.batchsize
    for dtype, shuffle in [('_train', True), ('_dev', False), ('_test', False)]:
        dataset = treeDataset(data_path=args.data_path+dtype, embed_path=args.elmo_path)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_worker, collate_fn=tree_collate_fn)
        tree.append(dataloader)


    print('finish parsing data')
    ################
    # select model #
    ################
    predictor = SpanSelectionParser(
        max_n_spans_para=max_n_spans_para,
        settings=args,
        baseline_heuristic=args.baseline_heuristic,
        use_elmo=args.use_elmo,
        decoder=args.decoder
        )
    print('model', predictor)
    print('finish build model')
    sys.stderr.write("dump setting file...\n")
    model = FscoreClassifier(
                predictor,
                max_n_spans_para,
                args,
                fscore_link_fun=fscore_binary,
                count_prediction=count_prediction,
                ac_type_alpha=args.ac_type_alpha,
                link_type_alpha=args.link_type_alpha
                )

    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Specify GPU ID from command line
    model.predictor.to(device)

    if( os.path.isfile(args.save_dir) ):
        model.predictor.load_state_dict(torch.load(args.save_dir))
    elif( not os.path.exists(args.save_dir)):
        os.mkdir(args.save_dir)

    #############
    # optimizer #
    #############
    para = []
    for _ in model.predictor.parameters():
        para.append(_)

    if(args.optimizer == "Adam"):
        optimizer = optim.Adam(para, lr=args.lr)
    elif(args.optimizer == 'Ranger'):
        optimizer = Ranger(para, lr=args.lr)
    elif(args.optimizer == 'Radam'):
        optimizer = RAdam(para, lr=args.lr)
    else:
        raise NotImplementedError
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    ##################
    # train and eval #
    ##################
    for i in range(args.epoch):
        print(i)
        if(args.train):
            train(model, optimizer, tree[0], device, args.save_dir+'/checkpoint_{}.pt'.format(i))

        if(args.dev):
            print('dev')
            evaluation(model, tree[1], device, mode='dev')

        if(args.test):
            print('test')
            evaluation(model, tree[2], device, mode='test')
        print('-'*10)
        scheduler.step()

def test(model, data_iter, device):
    model.predictor.eval()

    output = None
    for i, data, label in enumerate(data_iter):
        x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, xs_len, adu_len, topic_len, mask = convert(data, device, in_data=True)
        y = model.train(x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, xs_len, adu_len, topic_len, mask, False)

        if(i == 0):
            output = y[1:]
        else:
            for j, _ in enumerate(y[1:]):
                output[j] = torch.cat([output[j], _], dim=0)

    torch.save(output, args.pred_result)

def main():
    args = parse_args()

    reset_seed(args)
    #outdir = log_info(args)

    setting(args)


if __name__ == "__main__":
    main()
