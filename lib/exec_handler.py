
import os
import logging
import time 
from collections import OrderedDict, defaultdict

import torch
import torch.optim as optim
import torch.nn as nn 
import wandb

from lib.data_handler import RawSquadDataset, DataManager, RecurrentDataManager
import lib.model as models
import  lib.globals as globals
import lib.utils as utils 
from lib.evaluate import QA_evaluate


logger = logging.getLogger(globals.LOG_NAME)

class QA_handler : 
     
    def __init__(self,model_name, dataset, device):
    
        dataset_path = os.path.join(globals.DATA_FOLDER,dataset)

        squad_dataset = RawSquadDataset(dataset_path)

        if model_name == 'DrQA' : 

            self.data_manager : DataManager = RecurrentDataManager(squad_dataset,device)

            HIDDEN_DIM = 128
            LSTM_LAYER = 3
            DROPOUT = 0.3
            N_EPOCHS = 5
            GRAD_CLIPPING = 10
            BATCH_SIZE = 32
            RANDOM_BATCH = False
            
            self.model = models.DrQA(HIDDEN_DIM,LSTM_LAYER,DROPOUT,self.data_manager.emb_model.vectors,self.data_manager.vocab[globals.PAD_TOKEN],device)

            self.optimizer = optim.Adamax(self.model.parameters())
            self.criterion = nn.CrossEntropyLoss().to(device)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING
            }
    
            self.dataloaders = self.data_manager.get_dataloaders(BATCH_SIZE,RANDOM_BATCH)
        
        elif model_name == 'BERT' :
            raise NotImplementedError()
    
    def train_loop(self, iterator):

        start = time.perf_counter()
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_param['grad_clipping'])     #TODO che valore mettere come max norm ? 

            #update the gradients
            self.optimizer.step()

            pred_start, pred_end = utils.compute_predictions(pred_start_raw,pred_end_raw)

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
        
        end = time.perf_counter()
        metrics['epoch_time'] = start-end

        return utils.compute_avg_dict('train',metrics)

    
    def val_loop(self, iterator):

        start = time.perf_counter()
        metrics = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            
            for batch_id, batch in enumerate(iterator):

                pred_start_raw, pred_end_raw = self.model(batch)

                true_start, true_end = batch['label_token_start'], batch['label_token_end']

                loss = self.criterion(pred_start_raw,true_start) + self.criterion(pred_end_raw,true_end)       #TODO come calcolarla ? 

                pred_start, pred_end = utils.compute_predictions(pred_start_raw,pred_end_raw)

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
        
        end = time.perf_counter()
        metrics['epoch_time'] = start-end

        return utils.compute_avg_dict('val',metrics)

    
    def train_and_eval(self):

        train_dataloader, val_dataloader = self.dataloaders

        for epoch in range(self.run_param['n_epochs']):

            start_time = time.perf_counter()

            train_metrics = self.train_loop(train_dataloader)
            val_metrics = self.val_loop(val_dataloader)

            end_time = time.perf_counter()

            logger.info('epoch %d, tot time for train and eval: %f',epoch+1,start_time-end_time)
        
        #TODO do something with those metrics as save model somewhere 

    



        
        
