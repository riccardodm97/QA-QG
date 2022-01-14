
import torch 
import numpy as np
import wandb

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict, defaultdict

from lib.evaluate import QA_evaluate
import lib.globals as globals

from typing import Tuple

import logging 

logger = logging.getLogger(globals.LOG_NAME)


class QATrainer :

    def __init__(self, model : nn.Module, optimizer : optim.Optimizer, criterion, param : dict, device): 

        self.model = model 
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device               
        self.param = param
    
    def train_loop(self, iterator):

        metrics = defaultdict(list)

        self.model.train()

        for batch_id, batch in enumerate(iterator):

            #zero the gradients 
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()        

            pred_start_raw, pred_end_raw = self.model(batch)

            true_start, true_end = batch['label_token_start'], batch['label_token_end']

            loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)

            #backward pass 
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.param['grad_clipping'])     #TODO CHE VALORE METTERE COME MAX_NORM ??

            #update the gradients
            self.optimizer.step()

            # batch_loss += loss.item()    #accumulate batch loss 

            pred_start, pred_end = self.compute_predictions(pred_start_raw,pred_end_raw)

            to_eval = OrderedDict ({
                'pred_start' : pred_start.cpu(),
                'pred_end' : pred_end.cpu(),
                'true_start' : true_start.cpu(),
                'true_end' : true_end.cpu(),
                'context' : batch['context_text'],
                'offsets' : batch['context_offsets'],
                'answer' : batch['answer_text']
                })

            batch_metrics = QA_evaluate(to_eval)

            batch_metrics['loss'] = loss.item()
            
            #append all values of batch metrics to the corresponid element in metrics 
            for k,v in batch_metrics.items():
                metrics[k].append(v)

        return {self.f("train",k): np.mean(v).round(2) for k,v in metrics.items()}

    
    def val_loop(self, iterator):

        metrics = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            
            for batch_id, batch in enumerate(iterator):

                pred_start_raw, pred_end_raw = self.model(batch)

                true_start, true_end = batch['label_token_start'], batch['label_token_end']

                loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)

                pred_start, pred_end = self.compute_predictions(pred_start_raw,pred_end_raw)

                to_eval = OrderedDict ({
                    'pred_start' : pred_start.cpu(),
                    'pred_end' : pred_end.cpu(),
                    'true_start' : true_start.cpu(),
                    'true_end' : true_end.cpu(),
                    'context' : batch['context_text'],
                    'offsets' : batch['context_offsets'],
                    'answer' : batch['answer_text']
                    })

                batch_metrics = QA_evaluate(to_eval)

                batch_metrics['loss'] = loss.item()
                
                #append all values of batch metrics to the corresponid element in metrics 
                for k,v in batch_metrics.items():
                    metrics[k].append(v)

        return {self.f("val",k): np.mean(v).round(2) for k,v in metrics.items()}

    
    def train_and_eval(self, dataloaders : Tuple[DataLoader,...]):

        train_dataloader, val_dataloader = dataloaders

        for epoch in range(self.param['n_epochs']):

            t = self.train_loop(train_dataloader)
            v = self.val_loop(val_dataloader)

            logger.info('TRAIN EPOCH %d: loss %f, accuracy %f, f1 %f, em %f, s_dist %f, e_dist %f',epoch+1,t['loss'],t['accuracy'],t['f1'],t['em'],t['mean_start_dist'],t['mean_end_dist'])
            logger.info('VAL EPOCH %d: loss %f, accuracy %f, f1 %f, em %f, s_dist %f, e_dist %f',epoch+1,v['loss'],v['accuracy'],v['f1'],v['em'],v['mean_start_dist'],v['mean_end_dist'])
            wandb.log(t, step=epoch)
            wandb.log(v, step=epoch)

   

    def compute_predictions(self,starts,ends):

        pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)

        pred_start, pred_end = torch.argmax(pred_start_logit,dim=1), torch.argmax(pred_end_logit,dim=1)

        return pred_start, pred_end

    def f(self, prep : str, key : str):
        return prep+'/'+key