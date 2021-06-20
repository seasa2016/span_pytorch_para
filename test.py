import argparse
import sys
import os
import numpy as np
import json

import torch

from torch.utils.data import DataLoader

from data.dataloader import treeDataset, tree_collate_fn
from model.span_selection_model import SpanSelectionParser

from utils.misc import (
    read_vocabfile, load_glove
)

from utils.option import (
    add_test_args, add_embed_args, add_model_arch_args
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_embed_args(parser)
    parser = add_test_args(parser)
    parser = add_model_arch_args(parser)

    return parser.parse_args()

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



def test(model, data_iter, device, pred_path):
    model.eval()


    with torch.no_grad():
        with open(pred_path, 'w') as f:
            for i, (data, label) in enumerate(data_iter):
                if(i%50==0):
                    print(i)

                for key in data:
                    try:
                        data[key] = data[key].to(device)
                    except:
                        pass
                """
                input:
                    x_spans, shell_spans, x_position_info,
                    elmo_embeddings, topic_embeddings,
                    xs_lens, adu_lens, topic_lens
                output:
                    pair_scores,
                    ac_types,
                    link_types
                """
                try:
                    y = model( data['span'], data['shell_span'], data['ac_position_info'], data['elmo_emb'], data['topic_emb'], data['author'], data['elmo_length'], data['adu_length'], data['topic_length'])
                except:
                    print(i, data['uid'])
                    print(i, data['elmo_length'])
                    print(i, data['adu_length'])
                    print(i, data['topic_length'])
                    None+1

                for j in range(3):
                    y[j] = y[j].detach().cpu().numpy()

                for j in range( y[0].shape[0] ):
                    output = {
                        'uid':data['uid'][j],
                        'link_score':y[0][j].tolist(),
                        'type_score':y[1][j].tolist(),
                        'link_type_score':y[2][j].tolist()
                    }
                    f.write(json.dumps(output))
                    f.write('\n')



def main():
    args = parse_args()

    max_n_spans_para = 128

    dataset = treeDataset(data_path=args.test_data, embed_path=args.test_elmo, max_adu=max_n_spans_para)
    dataloader = DataLoader(dataset, batch_size=args.batchsize,shuffle=False, num_workers=4,collate_fn=tree_collate_fn)

    print('finish parsing data')
    ################
    # select model #
    ################
    model = SpanSelectionParser(
        max_n_spans_para=max_n_spans_para,
        settings=args,
        baseline_heuristic=args.baseline_heuristic,
        use_elmo=args.use_elmo,
        decoder=args.decoder
        )
    print('model', model)
    print('finish build model')


    if(torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # Specify GPU ID from command line
    model.to(device)

    model.load_state_dict(torch.load(args.save_model))

    test(model, dataloader, device, args.result_path)


if __name__ == "__main__":
    main()
