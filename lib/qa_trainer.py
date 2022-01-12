

import torch 
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from collections import OrderedDict

from evaluate import QA_evaluate


class QATrainer :

    def __init__(self, model : nn.Module, optimizer : optim.Optimizer, criterion,  param : dict(), device): 

        self.model = model 
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.param = param 

    
    def train_loop(self, iterator):

        batch_loss = 0.0

        metrics = OrderedDict({

        })


        self.model.train()

        for batch_id, batch in enumerate(iterator):

            #zero the gradients 
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()        

            predictions = self.model(batch)

            pred_start_raw, pred_end_raw = predictions 

            true_start, true_end = batch['label_token_start'].squeeze(), batch['label_token_end'].squeeze()

            loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)

            #backward pass 
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()    #accumulate batch loss 

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
            
            #TODO APPEND VALUES OF BATCH_METRICS TO METRICS 


        #TODO AVERAGE ALL RESULTS IN METRICS 

        metrics['epoch_loss'] = batch_loss/(batch_id+1) 

        return metrics

    
    def val_loop(self, iterator):

        batch_loss = 0

        self.model.eval()

        with torch.no_grad():
            
            for batch_id, batch in enumerate(iterator):

                predictions = self.model(batch)

                pred_start, pred_end = predictions 

                true_start, true_end = batch['label_token_start'].squeeze(), batch['label_token_end'].squeeze()

                loss = self.criterion(pred_start,true_start) + self.criterion(pred_end,true_end)

                batch_loss += loss.item()   #accumulate batch loss 
                

        epoch_loss = batch_loss/(batch_id+1) 

        return epoch_loss

    
    def train_and_eval(self, dataloaders : tuple[DataLoader,...], device):


        train_dataloader, val_dataloader = dataloaders

        for epoch in range(self.param['n_epochs']):

            train_metrics = self.train_loop(train_dataloader)
            val_metrics = self.val_loop(val_dataloader)
        
        
        
        
        return 
    

    def compute_predictions(starts,ends):

        pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)

        pred_start, pred_end = torch.argmax(pred_start_logit,dim=1), torch.argmax(pred_end_logit,dim=1)

        return pred_start, pred_end
