
import torch 
import numpy as np

from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict, defaultdict

from evaluate import QA_evaluate

from typing import Tuple


class QATrainer :

    def __init__(self, model : nn.Module, optimizer : optim.Optimizer, criterion,  param : dict(), device): 

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

            true_start, true_end = batch['label_token_start'].squeeze(), batch['label_token_end'].squeeze()

            loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)

            #backward pass 
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)     #TODO CHE VALORE METTERE COME MAX_NORM ??

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

        return {k: np.mean(v) for k,v in metrics.items()}

    
    def val_loop(self, iterator):

        metrics = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            
            for batch_id, batch in enumerate(iterator):

                pred_start_raw, pred_end_raw = self.model(batch)

                true_start, true_end = batch['label_token_start'].squeeze(), batch['label_token_end'].squeeze()

                loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)

                pred_start, pred_end = self.compute_predictions(pred_start_raw,pred_end_raw)

                to_eval = OrderedDict ({
                    'pred_start' : pred_start.cpu(),
                    'pred_end' : pred_end.cpu(),
                    'true_start' : true_start.cpu(),
                    'true_end' : true_end.cpu(),
                    'context' : batch['context_text'],
                    'offsets' : batch['offsets'],
                    'answer' : batch['answer']
                    })

                batch_metrics = QA_evaluate(to_eval)

                batch_metrics['loss'] = loss.item()
                
                #append all values of batch metrics to the corresponid element in metrics 
                for k,v in batch_metrics.items():
                    metrics[k].append(v)

        return {k: np.mean(v) for k,v in metrics.items()}

    
    def train_and_eval(self, dataloaders : Tuple[DataLoader,...], device):

        self.model.to(device)

        train_dataloader, val_dataloader = dataloaders

        for epoch in range(self.param['n_epochs']):

            train_metrics = self.train_loop(train_dataloader)
            val_metrics = self.val_loop(val_dataloader)
        
        
        
        
        return 
    

    def compute_predictions(starts,ends):

        pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)

        pred_start, pred_end = torch.argmax(pred_start_logit,dim=1), torch.argmax(pred_end_logit,dim=1)

        return pred_start, pred_end
