import os
import logging
import time 
from collections import OrderedDict, defaultdict

import torch
import torch.optim as optim
import torch.nn as nn 
import wandb
from tqdm import tqdm

from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from src.data_handler import RawSquadDataset, DataManager, RecurrentDataManager, TransformerDataManager
import src.model as models
import  src.globals as globals
import src.utils as utils 
from src.evaluate import QA_evaluate


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
            N_EPOCHS = 10
            GRAD_CLIPPING = 10
            BATCH_SIZE = 32
            LR = 0.002
            RANDOM_BATCH = False
            LR_SCHEDULER = False

            #log model configuration   
            wandb.config.hidden_dim = HIDDEN_DIM
            wandb.config.lstm_layer = LSTM_LAYER
            wandb.config.dropout = DROPOUT
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER
            
            
            self.model = models.DrQA(HIDDEN_DIM,LSTM_LAYER,DROPOUT,self.data_manager.emb_model.vectors,self.data_manager.vocab[globals.PAD_TOKEN],device)

            self.optimizer = optim.Adamax(self.model.parameters(), lr=LR)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }
        
        elif model_name == 'BERT' :
            
            self.data_manager : DataManager = TransformerDataManager(squad_dataset,device)

        
            N_EPOCHS = 5
            BATCH_SIZE = 8
            LR = 2e-5
            EPS = 1e-08
            DROPOUT = 0.3
            WEIGHT_DECAY = 0.01
            RANDOM_BATCH = False
            GRAD_CLIPPING = 2.0
            LR_SCHEDULER = True
            WARMUP = 2000

            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.epsilon = EPS
            wandb.config.weight_decay = WEIGHT_DECAY
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER
            wandb.config.warmup = WARMUP
            
            
            self.model = models.BertQA(device,DROPOUT)

            self.optimizer = AdamW(self.model.parameters(), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)
        
            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }
    
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.dataloaders = self.data_manager.get_dataloader('train',BATCH_SIZE,RANDOM_BATCH), self.data_manager.get_dataloader('val',BATCH_SIZE,RANDOM_BATCH)
        self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=WARMUP, num_training_steps=N_EPOCHS * len(self.dataloaders[0]))

        wandb.watch(self.model, self.criterion)

    
    def train_loop(self, iterator):

        start_time = time.perf_counter()
        metrics = defaultdict(list)

        self.model.train()

        for batch_id, batch in enumerate(tqdm(iterator)):

            #zero the gradients 
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()        

            pred_start_raw, pred_end_raw = self.model(batch)

            true_start, true_end = batch['label_token_start'], batch['label_token_end']

            start_loss = self.criterion(pred_start_raw,true_start) 
            end_loss = self.criterion(pred_end_raw,true_end)

            total_loss = (start_loss + end_loss) #/2       #TODO come calcolarla ? 

            #backward pass 
            total_loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_param['grad_clipping'])     #TODO che valore mettere come max norm ?   

            #update the gradients
            self.optimizer.step()

            #update the learning rate
            if self.run_param['lr_scheduler']:
                l = self.lr_scheduler.step()
                wandb.log({"lr": l, "batch": batch_id}, commit=False)

            pred_start, pred_end = utils.compute_predictions(pred_start_raw,pred_end_raw)

            to_eval = OrderedDict ({
                'pred_start' : pred_start.cpu(),
                'pred_end' : pred_end.cpu(),
                'true_start' : true_start.cpu(),
                'true_end' : true_end.cpu(),
                'offsets' : batch['offsets'],
                'context' : batch['context_text'],
                'answer' : batch['answer_text']
                })

            batch_metrics = QA_evaluate(to_eval)

            batch_metrics['loss'] = total_loss.item()
            
            #append all values of batch metrics to the corresponid element in metrics 
            for k,v in batch_metrics.items():
                metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        return utils.compute_avg_dict('train',metrics)

    
    def val_loop(self, iterator):

        start_time = time.perf_counter()
        metrics = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            
            for batch in tqdm(iterator):

                pred_start_raw, pred_end_raw = self.model(batch)

                true_start, true_end = batch['label_token_start'], batch['label_token_end']

                start_loss = self.criterion(pred_start_raw,true_start) 
                end_loss = self.criterion(pred_end_raw,true_end)

                total_loss = (start_loss + end_loss) #/2

                pred_start, pred_end = utils.compute_predictions(pred_start_raw,pred_end_raw)

                to_eval = OrderedDict ({
                    'pred_start' : pred_start.cpu(),
                    'pred_end' : pred_end.cpu(),
                    'true_start' : true_start.cpu(),
                    'true_end' : true_end.cpu(),
                    'offsets' : batch['offsets'],
                    'context' : batch['context_text'],
                    'answer' : batch['answer_text']
                    })

                batch_metrics = QA_evaluate(to_eval)

                batch_metrics['loss'] = total_loss.item()
                
                #append all values of batch metrics to the corresponid element in metrics 
                for k,v in batch_metrics.items():
                    metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        return utils.compute_avg_dict('val',metrics)

    
    def train_and_eval(self):

        best_val_f1 = 0.0
        model_save_path = 'models/'+self.model.get_model_name()+'.pt'

        train_dataloader, val_dataloader = self.dataloaders

        for epoch in range(self.run_param['n_epochs']):

            logger.info('starting epoch %d',epoch+1)
            start_time = time.perf_counter()

            train_metrics = self.train_loop(train_dataloader)
            val_metrics = self.val_loop(val_dataloader)

            end_time = time.perf_counter()

            logger.info('epoch %d, tot time for train and eval: %f',epoch+1,end_time-start_time)
            logger.info('train: loss %f, accuracy %f, f1 %f, em %f, s_dist %f, e_dist %f, num_acc_s %f, num_acc_e %f',
                        train_metrics["train/loss"], train_metrics["train/accuracy"],train_metrics["train/f1"], train_metrics["train/em"], 
                        train_metrics["train/mean_start_dist"],train_metrics["train/mean_end_dist"],
                        train_metrics['train/numerical_accuracy_start'],train_metrics['train/numerical_accuracy_end'])
            logger.info('val: loss %f, accuracy %f, f1 %f, em %f, s_dist %f, e_dist %f, num_acc_s %f, num_acc_e %f',
                        val_metrics["val/loss"], val_metrics["val/accuracy"],val_metrics["val/f1"], val_metrics["val/em"], 
                        val_metrics["val/mean_start_dist"],val_metrics["val/mean_end_dist"],
                        val_metrics['val/numerical_accuracy_start'],val_metrics['val/numerical_accuracy_end'])           

            wandb.log(train_metrics)
            wandb.log(val_metrics)
            wandb.log({'epoch_num': epoch})
        
            #TODO save model somewhere 
            if val_metrics['val/f1'] >= best_val_f1:
                best_val_f1 = val_metrics['val/f1']
                if not os.path.exists('models'):        
                    os.makedirs('models')
                torch.save(self.model.state_dict(),model_save_path)
            
        wandb.save(model_save_path)