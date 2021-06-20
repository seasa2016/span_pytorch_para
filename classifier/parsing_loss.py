import math
from .decode import decode_mst
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

class FscoreClassifier():

    compute_accuracy = True

    def __init__(self,
                 predictor,
                 max_n_spans,
                 args,
                 fscore_link_fun=None,
                 count_prediction=None,
                 ac_type_alpha=0,
                 link_type_alpha=0):

        super(FscoreClassifier, self).__init__()

        self.predictor = predictor
        self.max_n_spans = max_n_spans
        self.settings = args
        self.fscore_link_fun = fscore_link_fun

        self.count_prediction = count_prediction
        self.ac_type_alpha = ac_type_alpha
        self.link_type_alpha = link_type_alpha

    def train(self, x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author,
            xs_len, adu_len, topic_len, mask=None,
            ts_link=None, ts_type=None, ts_link_type=None, train=True):
        device = x_spans.device
        
        y_link, y_type, y_link_type = self.predictor(x_spans, shell_spans, x_position_info, elmo_embeddings, topic_embeddings, author, xs_len, adu_len, topic_len, mask)
              
        output, label = [], []
        if(train):
            batch_size, max_n_spans, link_len = y_link.shape
            y_link_mst = y_link.view(-1, link_len).max(dim=-1)[1].detach().to(device)
        else:
            y_link_mst = decode_mst(y_link.detach(), ts_link)
        
        output = [y_link_mst, y_link, y_type, y_link_type]
        if(ts_link is not None):
            loss = 0
            stat = {}
            label_mask = (ts_link>=0).view(-1)
            link_index = torch.arange(0, label_mask.shape[0], dtype=torch.long).to(device).masked_select(label_mask)

            label_mask = (ts_type>=0).view(-1)
            type_index = torch.arange(0, label_mask.shape[0], dtype=torch.long).to(device).masked_select(label_mask)
            
            # link
            temp_link = ts_link.clone()

            criterion = nn.CrossEntropyLoss()
            batch_size, max_n_spans, link_len = y_link.shape
            y_link_mst = y_link_mst.view(-1)[link_index]
            y_link = y_link.view(-1, link_len)[link_index]

            ts_link = ts_link.view(-1)[link_index]
            #print(y_link.shape)
            #print(ts_link)
            loss_link = criterion(y_link, ts_link)

            stat['loss_link'] = loss_link.detach().cpu().item()
            loss += (1 - self.ac_type_alpha -
                        self.link_type_alpha)*loss_link


            assert self.ac_type_alpha + self.link_type_alpha <= 1
            # type
            batch_size, n_spans, link_len = y_type.shape
            y_type = y_type.view(-1, link_len)[type_index]
            ts_type = ts_type.view(-1)[type_index]
            loss_type = criterion(y_type, ts_type)
            stat['loss_type'] = loss_type.detach().cpu().item()

            if(self.ac_type_alpha):
                loss += self.ac_type_alpha*loss_type

            # y_link_type contains all pair's relation score, we should extract out what we need
            criterion = nn.BCEWithLogitsLoss()
            batch_size, n_spans, _ = y_link_type.shape

            second_index = (ts_link_type.view(-1)[link_index]<2)
            if(second_index.sum().cpu().item() == 0):
                stat['loss_link_type'] = 0
            else:
                y_link_type = y_link_type.view(-1, _)[link_index]

                ex_index = ((torch.arange(0, batch_size, dtype=torch.long).to(device)*_).unsqueeze(-1) + temp_link).view(-1)[link_index]

                y_link_type = y_link_type.view(-1)[ex_index][second_index]
                ts_link_type = ts_link_type.view(-1)[link_index][second_index]

                loss_link_type = criterion(y_link_type, ts_link_type.float())
                stat['loss_link_type'] = loss_link_type.detach().cpu().item()

            if(self.link_type_alpha):
                if(second_index.sum().cpu().item() != 0):
                    loss += self.link_type_alpha*loss_link_type

            return loss, stat, [y_link_mst, y_link.detach(), y_type.detach(), y_link_type.detach()], [ts_link.detach(), ts_type.detach(), ts_link_type.detach()]
        
        else:
            return out
    def find(self, pred, gold, dtype=False):
        best = [0, 0]
        if(dtype):
            for i in range(10, 40):
                temp_pred = (pred>(i/100)).long()
                f_type = [f1_score(gold==i, temp_pred==i, average='binary') for i in range(2)]
                macro = sum(f_type)/2
                print(i, macro, end=' ')
                if(macro>best[0]):
                    best = (macro, i/100)
        
        else:
            for i in range(10):
                temp_pred = (pred>(i/10)).long()
                f_type = [f1_score(gold==i, temp_pred==i, average='binary') for i in range(2)]
                macro = sum(f_type)/2
                print(i, macro, end=' ')
                if(macro>best[0]):
                    best = (macro, i/10)
        print()        
        return best[1]

    def evaluate(self, total, best_threshold=[0.5, 0.5], mode='train'):
        stat = {}

        for key in ['link', 'link_type', 'type']:
            total['pred'][key] = torch.cat(total['pred'][key], dim=-1)
            total['label'][key] = torch.cat(total['label'][key], dim=-1)
        total['pred']['link_mst'] = torch.cat(total['pred']['link_mst'], dim=-1)

        pred_link_sorts = total['pred']['link_sort']
        pred_link_mst = total['pred']['link_mst'].detach().cpu()
        pred_link = total['pred']['link'].detach().cpu()
        pred_type = total['pred']['type'].detach().cpu()
        pred_link_type  = total['pred']['link_type'].detach().cpu()
        ts_link = total['label']['link'].detach().cpu()
        ts_type = total['label']['type'].detach().cpu()
        ts_link_type = total['label']['link_type'].detach().cpu()
        adu_len = total['adu_len']
        trc_len = total['trc_len']

        macro_f_scores = []
        ###########################
        # link prediction results #
        ###########################
        accuracy_link_mst = (pred_link_mst == ts_link).float().mean().item()
        stat['accuracy_link_mst'] = accuracy_link_mst

        stat['mrr_link'] = 0
        for t, pred_link_sort in zip(ts_link, pred_link_sorts):
            #print(pred_link_sort)
            for index, pred_ans in enumerate(pred_link_sort):
                if(pred_ans == t):
                    stat['mrr_link'] += 1/(index+1)
        stat['mrr_link'] /= len(pred_link_sorts)

        accuracy_link = (pred_link == ts_link).float().mean().item()
        stat['accuracy_link'] = accuracy_link

        f_link, f_nolink = accuracy_link, 0
        #f_link, f_nolink = self.fscore_link_fun(pred_link_mst,
        #                        ts_link.cpu(), trc_len, adu_len, self.max_n_spans)
        macro_f_link = (f_link+f_nolink)/2
        macro_f_scores.append(macro_f_link)
        stat['f_mst_link'] = f_link
        stat['f_mst_nolink'] = f_nolink
        stat['macro_f_mst_link'] = macro_f_link

        f_link, f_nolink = accuracy_link, 0
        #f_link, f_nolink = self.fscore_link_fun(pred_link.cpu(),
        #                        ts_link.cpu(), trc_len, adu_len, self.max_n_spans)
        macro_f_link = (f_link+f_nolink)/2
        macro_f_scores.append(macro_f_link)
        stat['f_link'] = f_link
        stat['f_nolink'] = f_nolink
        stat['macro_f_link'] = macro_f_link

        ##############################
        # ac_type prediction results #
        ##############################

        if(mode=='dev'):
            best_threshold[0] = self.find(pred_type, ts_type)
    
        #pred_type = (pred_type>best_threshold[0]).long()
        f_type = [f1_score(ts_type==i, pred_type==i, average='macro') for i in range(2)]

        accuracy_type = (pred_type == ts_type).float().mean().item()
        stat['accuracy_type'] = accuracy_type

        support_type = 2
        f_premise, f_claim = f_type
        macro_f_type = sum(f_type)/len(f_type)
    
        if(self.ac_type_alpha):
            macro_f_scores.append(macro_f_type)

        stat['f_premise'] = f_premise
        stat['f_claim'] = f_claim
        stat['macro_f_type'] =  sum(f_type)/len(f_type)
        for i , val in enumerate(range(support_type)):
            stat['ac_type_predicted_class_{}'.format(i)] =  self.count_prediction(pred_type, i)
        """
        for i, val in enumerate(support_type):
            stat['ac_type_gold_class_{}'.format(i)] = val
        """
        ################################
        # link type prediction results #
        ################################

        if(mode=='dev'):
            best_threshold[1] = self.find(pred_link_type, ts_link_type, True)
        
        pred_link_type = (pred_link_type>best_threshold[1]).long()
        f_link_type = [f1_score(ts_link_type==i, pred_link_type==i, average='binary') for i in range(2)]
        
        accuracy_type = (pred_link_type == ts_link_type).float().mean()
        stat['accuracy_link_type'] = accuracy_type

        support_type = 2
        macro_f_link_type = sum(f_link_type)/len(f_link_type)

        if(self.link_type_alpha):
            macro_f_scores.append(macro_f_link_type)

        stat['f_support'], stat['f_attack'] = f_link_type
        stat['macro_f_link_type'] = sum(f_link_type)/len(f_link_type)

        for i, val in enumerate(range(support_type)):
            stat['link_type_predicted_class_{}'.format(i)]=\
                    self.count_prediction(pred_link_type, i)
            stat['link_type_gold_class_{}'.format(i)] = val

        stat['total_macro_f'] = sum(macro_f_scores)/len(macro_f_scores)
        return stat
