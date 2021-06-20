import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base_model import BaseArgStrParser


class SpanSelectionParser(BaseArgStrParser):
    def __init__(self, max_n_spans_para,
                 settings, baseline_heuristic=False, use_elmo=True,
                 decoder="proposed"):

        super().__init__(
            max_n_spans_para,
            settings, baseline_heuristic, use_elmo, decoder)

        self.decoder_net = self.span_selection_model

    def span_selection_model(self, span_reps, topic_reps, adu_len,
                             relative_position_info):
        ######################
        # AC type prediction #
        ######################
        batch_size, max_n_span, _, _ = relative_position_info.shape
        device = span_reps.device

        if(self.lstm_ac_shell):
            ys_l_ac_shell, _ = self.AcShellBilstm(span_reps, adu_len)
            span_reps = ys_l_ac_shell

            _, n_span, hidden_dim = span_reps.shape
            span_reps = F.dropout(span_reps, self.dropout)
            span_reps = torch.cat([span_reps, torch.zeros(batch_size, max_n_span-n_span, hidden_dim, dtype=torch.float).to(device)], dim=1)


        ac_types = self.AcTypeLayer(span_reps)

        if(self.lstm_type):
            ys_l_last, _ = self.LastBilstm(span_reps, adu_len)
            span_reps = ys_l_last
            _, n_span, hidden_dim = span_reps.shape
            span_reps = F.dropout(span_reps, self.dropout)
            span_reps = torch.cat([span_reps, torch.zeros(batch_size, max_n_span-n_span, hidden_dim, dtype=torch.float).to(device)], dim=1)

        #span_reps = torch.cat([span_reps, ac_types.softmax(-1)], dim=-1)
        ############################
        # span pair representation #
        ############################
        pair_scores, link_types = self.calc_pair_score(span_reps, topic_reps,
                                           relative_position_info)

        return pair_scores, ac_types, link_types, span_reps

