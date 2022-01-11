import torch 
from torch import nn, optim

from torch.utils.data import DataLoader


class Trainer :

    def __init__(self, model : nn.Module, optimizer : optim.Optimizer, criterion,  param : dict(), device): 

        self.model = model 
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.param = param 

    
    def train_loop(self, iterator):

        batch_loss = 0

        self.model.train()

        for batch_id, batch in enumerate(iterator):

            #zero the gradients 
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()        

            predictions = self.model(batch)

            pred_start, pred_end = predictions 

            true_start, true_end = batch['label_token_start'].squeeze(), batch['label_token_end'].squeeze()

            loss = self.criterion(pred_start,true_start) + self.criterion(pred_end,true_end)

            #backward pass 
            loss.backward()
            self.optimizer.step()

            batch_loss += loss.item()    #accumulate batch loss 


        epoch_loss = batch_loss/(batch_id+1) 

        return epoch_loss 

    
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

            train_boh = self.train_loop(train_dataloader)
            val_boh = self.val_loop(val_dataloader)
        
        
        
        
        return 
