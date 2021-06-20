import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from model import module
import numpy as np
import h5py
import time

class BaseArgStrParser(nn.Module):
    def __init__(self,
                 max_n_spans_para,
                 args, baseline_heuristic=False, use_elmo=True,
                 decoder="proposed"):
        super(BaseArgStrParser, self).__init__()

        ##########################
        # set default attributes #
        ##########################
        self.eDim = args.eDim
        self.hDim = args.hDim
        self.dropout = args.dropout
        self.dropout_lstm = args.dropout_lstm
        self.dropout_embedding = args.dropout_embedding
        self.max_n_spans = max_n_spans_para
        self.decoder = decoder

        self.args = args

        ###############
        # Select LSTM #
        ###############
        self.lstm_ac = args.lstm_ac
        self.lstm_shell = args.lstm_shell
        self.lstm_ac_shell = args.lstm_ac_shell
        self.lstm_type = args.lstm_type

        #######################
        # position information #
        #######################
        ###############################################################3
        self.position_info_max = 16
        self.relative_post_info_max = 4
        self.relative_adu_info_max = 16

        self.position_info_size = 16
        self.relative_post_info_size = 16
        self.relative_adu_info_size = 16
        
        self.pos_post_emb = nn.Embedding(self.position_info_max, self.position_info_size)
        self.pos_para_emb = nn.Embedding(self.position_info_max, self.position_info_size)

        self.dist_post_emb = nn.Embedding(self.relative_post_info_max, self.relative_post_info_size)
        self.dist_para_emb = nn.Embedding(self.relative_adu_info_max, self.relative_adu_info_size)

        self.author_size = 32
        self.author_emb = nn.Embedding(2, self.author_size)
        self.relative_position_info_size = self.relative_adu_info_size + self.relative_post_info_size


        ################
        # elmo setting #
        ################
        self.use_elmo = use_elmo
        if self.use_elmo:
            self.eDim = 1024
            self.elmo_task_gamma = nn.Parameter(torch.ones(1))
            self.elmo_task_s = nn.Parameter(torch.ones(3))

        ##########
        # others #
        ##########
        self.baseline_heuristic = baseline_heuristic

        ##########
        # Default #
        ##########
        #self.Embed_x = nn.Embedding(self.encVocabSize,
        #                                self.eDim)
        self.dropout2d_init = nn.Dropout2d(0.7) 
        self.dropout2d_sec = nn.Dropout2d(0) 
        self.Bilstm = module.lstm(input_dim=self.eDim, output_dim=self.hDim,
                    num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)

        self.Topiclstm = module.lstm(input_dim=self.eDim, output_dim=self.hDim,
                    num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        ##############################
        # hidden representation size #
        ##############################
        self.lstm_out = self.hDim*2

        # the size of representation created with LSTM-minus
        self.span_rep_size = self.lstm_out * 2

        # output of AC layer
        if(self.lstm_ac):
            self.ac_rep_size = self.lstm_out
            self.AcBilstm = module.lstm(input_dim=self.span_rep_size + self.author_size, output_dim=self.hDim,
                    num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        else:
            self.ac_rep_size =  self.span_rep_size

        # output of AM layer
        if(self.lstm_shell):
            self.shell_rep_size = self.lstm_out

            self.ShellBilstm = module.lstm(input_dim=self.span_rep_size + self.author_size, output_dim=self.hDim,
                num_layers=1, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)
        else:
            self.shell_rep_size = self.span_rep_size

        # the size of ADU representation
        n_ac_shell_latm_layers = 1
        self.ac_shell_rep_size_in = self.ac_rep_size + self.shell_rep_size + self.position_info_size + self.position_info_size + self.author_size

        self.AcShellBilstm = module.lstm(input_dim=self.ac_shell_rep_size_in, output_dim=self.hDim,
            num_layers=n_ac_shell_latm_layers, batch_first=True, dropout=self.dropout_lstm, bidirectional=True)

        # output of ADU layer
        self.ac_shell_rep_size_out = self.lstm_out if self.lstm_ac_shell else self.ac_shell_rep_size_in
        self.LastBilstm = module.lstm(input_dim=self.ac_shell_rep_size_out, output_dim=self.hDim,
                            num_layers=1, batch_first=True, dropout=self.dropout_lstm)


        # output of Encoder (ADU-level)
        self.reps_for_type_classification = self.ac_shell_rep_size_out
        self.AcTypeLayer = nn.Linear(in_features=self.reps_for_type_classification,
                                        out_features=5)

        # the size of ADU representations for link identification
        self.type_rep_size = self.lstm_out if self.lstm_type else self.ac_shell_rep_size_out
        #self.type_rep_size += 3

        # the size of ADU pair representation
        self.span_pair_size = self.type_rep_size*3 + self.relative_position_info_size
        
        self.LinkLayer = nn.Sequential(
                            nn.Linear(self.span_pair_size, 64),
                            nn.ReLU(),
                            nn.Dropout(0.1),
                            nn.Linear(64, 2)
                            )
        """
        self.LinkLayer = nn.Linear(self.span_pair_size, 2)
        """
        self.init_para()
        self.start = time.time()
    def init_para(self):
        def linear_init(model):
            for para in model.parameters():
                init.uniform_(para, -0.05, 0.05)

        # linear init
        for _ in [self.AcTypeLayer, self.LinkLayer]:
            linear_init(_)

        if(self.args.bow_feat):
            linear_init(self.BowFCLayer)

        # lstm init
        if self.lstm_ac:
            self.AcBilstm.init()
        if self.lstm_shell:
            self.ShellBilstm.init()
        for _ in [self.Bilstm, self.AcShellBilstm, self.LastBilstm, self.Topiclstm]:
            _.init()

    def sequence_embed(self, embed, xs, is_param=False):
        if is_param:
            ex = embed(xs)
            ex = F.dropout(ex, self.dropout_embedding)
        else:
            ex = embed.W.data[xs]
            ex = F.dropout(ex, self.dropout_embedding)
        return ex

    def set_embed_matrix(self, embedd_matrix):
        self.Embed_x.W.data = embedd_matrix

    def load_elmo(self, elmo_embed):
        if self.args.elmo_layers == "weighted":
            elmo_embeddings = self.elmo_task_s.softmax(-1).view(1, -1, 1, 1) * elmo_embed
            elmo_embeddings = elmo_embeddings.sum(dim=1)
        elif self.args.elmo_layers == "avg":
            elmo_embeddings = elmo_embed.sum(dim=1)/3
        else:
            elmo_embeddings = elmo_embed[:, int(self.args.elmo_layers)-1]

        if self.args.elmo_task_gamma:
            elmo_embeddings = self.elmo_task_gamma * elmo_embeddings

        elmo_embeddings = F.dropout(elmo_embeddings, self.dropout_embedding)

        return elmo_embeddings

    def position2onehot(self, inds, dim):
        inds = inds.long().abs()
        inds = inds.clamp(0, dim-1)
        return inds
        """
        device = inds.device

        eye = torch.eye(dim, dtype=torch.float).to(device)
        onehot = eye[inds]
        return onehot
        """

    def get_span_reps(self, ys_l, x_spans=None, x_len=None):
        """
        x_spans (batchsize, n_spans, 2)
        ys_l: (batchsize, length, rep)
        """
        # print('shape', ys_l.shape)
        span_reps = []
        if(x_spans is not None):
            for row, spans in enumerate(x_spans):
                span_reps.append([])
                for elmo_index, start, end in spans:
                    try:
                        start_hidden_states_forward = ys_l[elmo_index, start-1, :self.hDim]
                        end_hidden_states_forward = ys_l[elmo_index, end, :self.hDim]

                        start_hidden_states_backward = ys_l[elmo_index, end+1, self.hDim:]
                        end_hidden_states_backward = ys_l[elmo_index, start, self.hDim:]

                        span_forward = end_hidden_states_forward - start_hidden_states_forward
                        span_backward = end_hidden_states_backward - start_hidden_states_backward

                        span_reps[-1].append(torch.cat(
                    [span_forward, span_backward, start_hidden_states_forward, start_hidden_states_backward],
                    dim=-1))
                    except:
                        print(row, elmo_index, start, end)
                        print(spans)
                        None+1

                span_reps[-1] = torch.stack(span_reps[-1])
        else:
            # print(type(ys_l), ys_l.shape)
            # print(x_len)
            for row, l in enumerate(x_len):
                # print(row, l)
                span_reps.append(torch.cat(
                [ys_l[row, l-1, :self.hDim], ys_l[row, 0, self.hDim:]],
                dim=-1))

        return torch.stack(span_reps)


    def get_bow_reps(self, xs, xs_embed,
                        x_spans, shell_spans,
                        position_info,
                        xs_len, adu_len):

        assert len(x_spans[0]) == len(shell_spans[0])

        device = xs.device
        # max pooling
        eye = torch.eye(len(self.vocab), dtype=torch.float).to(device)

        max_xs_embed, avg_xs_embed, min_xs_embed = [], [], []
        xs_ids = []
        for i, (x_spans_in_para, shell_spans_in_para) in enumerate(zip(x_spans, shell_spans)):
            temp = [[], [], [], []]

            for x_span, shell_span in zip(x_spans_in_para, shell_spans_in_para):
                start, end = shell_span[0].item(), x_span[1].item()+1
                temp[0].append( xs_embed[i, start:end].max() )
                temp[1].append( (xs_embed[i, start:end].sum())/(end-start) )
                temp[2].append( xs_embed[i, start:end].min() )
                temp[3].append(eye[xs[i][start:end+1]].sum(dim=0)/(end-start))

            xs_ids.append(torch.stack(temp[3]))
            max_xs_embed.append(torch.stack(temp[0]))
            avg_xs_embed.append(torch.stack(temp[1]))
            min_xs_embed.append(torch.stack(temp[2]))

        xs_ids = torch.stack(xs_ids)
        max_xs_embed = torch.stack(max_xs_embed)
        avg_xs_embed = torch.stack(avg_xs_embed)
        min_xs_embed = torch.stack(min_xs_embed)

        #(all_n_spans, feature_vector)
        if(self.use_elmo):
            # We found that pooling-based features with ELMo does not significantly contribute the performance.
            # Then, we only used discrete BoW features for span-based models with ELMo.
            bow_reps = torch.cat([xs_ids], dim=-1)
        else:
            bow_reps = torch.cat([max_xs_embed,
                                min_xs_embed,
                                avg_xs_embed,
                                xs_ids
                                ], dim=-1)

        assert bow_reps.shape[-1] == self.bow_feature_size

        bow_reps = self.BowFCLayer(bow_reps).sigmoid()
        bow_reps = F.dropout(bow_reps, self.dropout)

        return bow_reps

    def hierarchical_encode(self, xs_embed,
                            x_spans, shell_spans, author_embed,
                            position_info, xs_len, adu_len):
        ###########
        # Encoder #
        ###########
        ys_l, _ = self.Bilstm(xs_embed, xs_len)
        

        # need to fix dimension
        if(self.lstm_ac):
            ac_reps = self.get_span_reps(ys_l, x_spans=x_spans)
            ac_reps = self.dropout2d_sec(ac_reps)
            ac_reps = torch.cat([ac_reps , author_embed], dim=-1)
            ac_reps, _ = self.AcBilstm(ac_reps, adu_len)
        else:
            ac_reps = self.get_span_reps(ys_l, x_spans=x_spans)

        if(self.lstm_shell):
            shell_reps = self.get_span_reps(ys_l, x_spans=shell_spans)
            shell_reps = self.dropout2d_sec(shell_reps)
            shell_reps = torch.cat([shell_reps , author_embed], dim=-1)
            shell_reps, _ = self.ShellBilstm(shell_reps, adu_len)
        else:
            shell_reps = self.get_span_reps(ys_l, x_spans=shell_spans)
        
        """
        baseline
        """
        device = shell_reps.device
        shell_reps = torch.zeros_like(shell_reps).to(device)


        ac_shell_reps = torch.cat([ac_reps, shell_reps], dim=-1)


        # print('shape', ac_shell_reps.shape, position_info.shape, span_reps_bow.shape)
        batch_size, n_span, hidden_dim = ac_shell_reps.shape
        _, max_n_span, _ = position_info.shape
        device = ac_shell_reps.device

        ac_shell_reps = torch.cat([ac_shell_reps, torch.zeros(batch_size, max_n_span-n_span, hidden_dim).to(device)], dim=1)

        ac_shell_reps = torch.cat(
            [ac_shell_reps, position_info.float(), author_embed], dim=-1)

        assert ac_shell_reps.shape[-1] == self.ac_shell_rep_size_in

        return ac_shell_reps


    def calc_pair_score(self, span_reps_pad, topic_reps, relative_position_info):
        ###########################
        # for link identification #
        ###########################
        #(batchsize, max_n_spans, span_representation)
        batchsize, max_n_spans, _ = span_reps_pad.shape
        device = span_reps_pad.device

        #(batchsize, max_n_spans, max_n_spans, span_representation)
        span_reps_matrix = span_reps_pad.unsqueeze(1).expand(-1, max_n_spans, -1, -1)

        #(batchsize, max_n_spans, max_n_spans, span_representation)
        span_reps_matrix_t = span_reps_matrix.transpose(2, 1)

        #(batchsize, max_n_spans, max_n_spans, pair_representation)
        pair_reps = torch.cat(
            [span_reps_matrix,
             span_reps_matrix_t,
             span_reps_matrix*span_reps_matrix_t,
             relative_position_info.float()],
            dim=-1)

        #########################
        #### add root object ####
        #########################

        #(batchsize, max_n_spans, span_rep_size)
        #print(span_reps_pad.shape)
        #print(topic_reps.shape)
        root_matrix = topic_reps.unsqueeze(1).expand_as(span_reps_pad)

        #(batchsize, max_n_spans, pair_rep_size)
        pair_reps_with_root = torch.cat([span_reps_pad,
                                        root_matrix,
                                        span_reps_pad*root_matrix,
                                        torch.zeros(batchsize,
                                        max_n_spans,
                                        self.relative_position_info_size, dtype=torch.float).to(device)
                                        ],
                                        dim=-1)

        #(batchsize, max_n_spans, max_n_spans+1, pair_rep_size)
        pair_reps = torch.cat([pair_reps,
                                pair_reps_with_root.view(
                                                batchsize,
                                                max_n_spans,
                                                1,
                                                self.span_pair_size)],
                            dim=2)

        relation_scores, pair_scores = self.LinkLayer(pair_reps).split(1, dim=-1)

        return pair_scores.squeeze(-1), relation_scores.squeeze(-1)

    def mask_link_scores(self, pair_scores, adu_len, batchsize,
                         max_n_spans, mask, mask_type="minus_inf"):
        device = pair_scores.device
        batch_size, max_n_spans, _ = pair_scores.shape

        mask = torch.cat([mask, torch.ones(batch_size, max_n_spans, 1, dtype=torch.long).to(device)], dim=2)

        for i in range(mask.shape[1]):
            mask[:, i, i] = 0

        # mask
        for i, _ in enumerate(adu_len):
            mask[i, :, _:-1] = 0

        if(mask_type == "minus_inf"):
            #matrix for masking
            padding = torch.full((batchsize, max_n_spans, max_n_spans+1), -1e16,
                            dtype=torch.float, device=device)
        else:
            padding = torch.zeros(batchsize, max_n_spans, max_n_spans+1,
                                 dtype=torch.float, device=device)

        # print('shape', mask.shape, pair_scores.shape, padding.shape)
        #(batchsize, max_n_spans, max_n_spans+1, 1)
        masked_pair_scores = torch.where(mask.byte(), pair_scores, padding)

        return masked_pair_scores

    def get_position_info(self, x_position_info):
        # the number of ACs in a batch
        batch_size, max_n_spans, _ = x_position_info.shape

        #(batchsize, 3, max_n_spans)
        #position_info = self.position2onehot(x_position_info, max_n_spans)
        pos_info = self.position2onehot(x_position_info, self.position_info_max)
        pos_emb = torch.cat([
            self.pos_post_emb(pos_info[:,:,0]), self.pos_para_emb(pos_info[:,:,1])
        ], dim=-1)

        #(batchsize, 3*max_n_spans)
        pos_emb = pos_emb.view(batch_size, max_n_spans, self.position_info_size*2)
        return pos_emb

    def get_relative_position_info(self, x_position_info):
        #(batchsize, max_n_spans, 3)
        batch_size, max_n_spans, _ = x_position_info.shape

        span_position_info_matrix = x_position_info.unsqueeze(1).expand(-1, max_n_spans, -1, -1)
        #(batchsize, max_n_spans, max_n_spans, 3)
        span_position_info_matrix_t = span_position_info_matrix.transpose(2, 1)

        #(batchsize, max_n_spans, max_n_spans, 3)
        # relative position information
        span_relative_position_info_matrix = span_position_info_matrix - span_position_info_matrix_t
        """
        #(batchsize*max_n_spans*max_n_spans, 1)
        relative_position = span_relative_position_info_matrix + self.max_n_spans
        """
        # self.relative_adu_info_size = 16
        # self.relative_post_info_size = 4
        #(batchsize*max_n_spans*max_n_spans, relative_position_info_size)
        adu_relative_position_info_matrix =  span_relative_position_info_matrix[:, :, :, 0]
        post_relative_position_info_matrix = span_relative_position_info_matrix[:, :, :, 1]

        relative_adu_info = self.position2onehot(adu_relative_position_info_matrix,
                                                      self.relative_adu_info_max)
        relative_post_info = self.position2onehot(post_relative_position_info_matrix,
                                                      self.relative_post_info_max)
        
        relative_adu_info = self.dist_para_emb(relative_adu_info)
        relative_post_info = self.dist_post_emb(relative_post_info)

        #(batchsize, max_n_spans, max_n_spans, relative_position_info_size)
        relative_adu_info = relative_adu_info.view(
                                        batch_size,
                                        max_n_spans,
                                        max_n_spans,
                                        self.relative_adu_info_size)
        relative_post_info = relative_post_info.view(
                                        batch_size,
                                        max_n_spans,
                                        max_n_spans,
                                        self.relative_post_info_size)

        relative_position_info = torch.cat([relative_adu_info, relative_post_info], dim=-1)
        return relative_position_info

    def majority_voting_to_links(self, position_info):
        pair_scores = torch.zeros(self.batchsize,
                                self.max_n_spans,
                                self.max_n_spans + 1, dtype=torch.float)

        for i, position_info in enumerate(position_info):
            if(position_info[0][-1] == 25):
                pair_scores[i, :, 0] = 1
                pair_scores[i, 0, self.max_n_spans] = 2
            else:
                pair_scores[i, :, self.max_n_spans] = 1
        return pair_scores

    def forward(self, x_spans, shell_spans, x_position_info,
            elmo_embeddings, topic_embeddings, author,
            xs_lens, adu_lens, topic_lens, mask=None):
        """
        Args:
        ids: essay ids

        Return:
        (all_spans, candidates, score)
        """
        batch_size, max_n_spans, _ = x_spans.shape
        device = x_spans.device

        ###################
        # load embeddings #
        ###################
        total_adu, _, sent_dim, hidden_dim = elmo_embeddings.shape
        #print(author, flush=True)
        #author_embed =  self.author_emb(author)
        #author_embed = torch.zeros_like(author_embed)
        author_embed = torch.zeros(batch_size, max_n_spans, 32).to(device)

        xs_embed = self.load_elmo(elmo_embeddings)
        topic_embed = self.load_elmo(topic_embeddings)

        xs_embed = self.dropout2d_init(xs_embed)
        topic_embed = self.dropout2d_init(topic_embed)

        ############################
        # get position information #
        ############################
        position_info = self.get_position_info(x_position_info)
        relative_position_info = self.get_relative_position_info(x_position_info)

        # print(time.time()-self.start)
        # self.time = time.time()
        ###########
        # encoder #
        ###########
        
        topic_reps, _ = self.Topiclstm(topic_embed, topic_lens)
        topic_reps = self.get_span_reps(topic_reps, x_len=topic_lens)

        span_reps = self.hierarchical_encode(
                                             xs_embed,
                                             x_spans,
                                             shell_spans,
                                             author_embed,
                                             position_info,
                                             xs_lens, adu_lens
                                             )

        # print(time.time()-self.start)
        # self.time = time.time()
        ###########
        # decoder #
        ###########
        pair_scores, ac_types, link_types, span_reps_pad =\
            self.decoder_net(span_reps, topic_reps, adu_lens,
                             relative_position_info)
        # print(time.time()-self.start)
        # self.time = time.time()

        if(self.baseline_heuristic):
            raise NotImplementedError('no this function')
            # pair_scores = self.majority_voting_to_links(position_info)

        if(mask is None):
            mask = torch.zeros(batch_size, max_n_spans, max_n_spans, dtype=torch.long).to(device)
            for i, adu_len in enumerate(adu_lens):
                mask[i, :adu_len, :adu_len] = 1

        masked_pair_scores = self.mask_link_scores(pair_scores,
                                                   adu_lens,
                                                   batch_size,
                                                   max_n_spans,
                                                   mask,
                                                   mask_type="minus_inf")
        # print(time.time()-self.start)
        # self.time = time.time()

        return [masked_pair_scores, ac_types, link_types]
