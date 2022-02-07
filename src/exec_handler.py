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

from src.data_handler import RawSquadDataset, DataManager, RnnDataManagerQA, RnnDataManagerQG,  TransformerDataManagerQA, BertDataManagerQG
import src.model as models
import  src.globals as globals
import src.utils as utils 
from src.evaluation import qa_evaluate, qg_evaluate

logger = logging.getLogger(globals.LOG_NAME)

class QA_handler : 
     
    def __init__(self, model_name, dataset_path, device):
    
        squad_dataset = RawSquadDataset(train_dataset_path = dataset_path)  # load and wrap dataset 

        if model_name == 'DrQA' : 

            self.data_manager : DataManager = RnnDataManagerQA(squad_dataset, device)   #let the datamanager manage all the data pipeline 

            HIDDEN_DIM = 128
            LSTM_LAYER = 3
            DROPOUT = 0.3
            N_EPOCHS = 15
            GRAD_CLIPPING = 2.0
            BATCH_SIZE = 32
            LR = 0.002
            RANDOM_BATCH = False
            LR_SCHEDULER = False
            FREEZE = False

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
            wandb.config.freeze_emb = FREEZE

            pad_idx = self.data_manager.vocab[globals.PAD_TOKEN]
            vectors = self.data_manager.emb_model.vectors
            
            self.model = models.DrQA(HIDDEN_DIM,LSTM_LAYER,DROPOUT,FREEZE,vectors,pad_idx,device)

            self.optimizer = optim.Adamax(self.model.parameters(), lr=LR)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }
        
        elif model_name == 'Bert' :
            
            self.data_manager : DataManager = TransformerDataManagerQA(squad_dataset, device)

            N_EPOCHS = 3
            BATCH_SIZE = 8
            LR = 3e-5
            EPS = 1e-08
            DROPOUT = 0.1
            WEIGHT_DECAY = 0.01
            RANDOM_BATCH = False
            GRAD_CLIPPING = 2.0
            LR_SCHEDULER = True
            WARMUP = 2000

            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.dropout = DROPOUT
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.epsilon = EPS
            wandb.config.weight_decay = WEIGHT_DECAY
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER
            wandb.config.warmup = WARMUP
            
            self.model = models.BertQA(device, DROPOUT)

            self.optimizer = AdamW(self.model.parameters(), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)
        
            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }

        elif model_name == 'Electra' :
            
            self.data_manager : DataManager = TransformerDataManagerQA(squad_dataset, device)

            N_EPOCHS = 3
            BATCH_SIZE = 8
            LR = 5e-5 #1e-4
            EPS = 1e-12 #1e-06
            DROPOUT = 0.1
            WEIGHT_DECAY = 0.01
            RANDOM_BATCH = False
            GRAD_CLIPPING = 2.0
            LR_SCHEDULER = True
            WARMUP = 2000
            HIDDEN_DIM = 384
            FREEZE = False

            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.dropout = DROPOUT
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.epsilon = EPS
            wandb.config.weight_decay = WEIGHT_DECAY
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER
            wandb.config.warmup = WARMUP
            wandb.config.hidden_dim = HIDDEN_DIM
            wandb.config.freeze = FREEZE
            
            self.model = models.ElectraQA(device, HIDDEN_DIM, FREEZE, dropout= DROPOUT)

            self.optimizer = AdamW(self.model.parameters(), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)
        
            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }
    
        self.dataloaders = self.data_manager.get_dataloader('train', BATCH_SIZE, RANDOM_BATCH), self.data_manager.get_dataloader('val', BATCH_SIZE, RANDOM_BATCH)
        if LR_SCHEDULER: self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=WARMUP, num_training_steps=N_EPOCHS * len(self.dataloaders[0]))

        self.criterion = nn.CrossEntropyLoss().to(device)


    
    def train_loop(self, iterator):

        start_time = time.perf_counter()
        metrics = defaultdict(list)

        self.model.train()

        for batch in tqdm(iterator):

            #zero the gradients 
            self.model.zero_grad(set_to_none=True)
            self.optimizer.zero_grad()        

            pred_start_raw, pred_end_raw = self.model(batch)  #get predictions 

            true_start, true_end = batch['label_token_start'], batch['label_token_end']   #get ground truth 

            start_loss = self.criterion(pred_start_raw,true_start) 
            end_loss = self.criterion(pred_end_raw,true_end)

            total_loss = (start_loss + end_loss) / 2       

            #backward pass 
            total_loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_param['grad_clipping'])      

            #update the gradients
            self.optimizer.step()

            #update the learning rate
            if self.run_param['lr_scheduler']:
                self.lr_scheduler.step()

            pred_start, pred_end = utils.compute_qa_predictions(pred_start_raw,pred_end_raw)

            to_eval = OrderedDict ({
                'pred_start' : pred_start.cpu(),
                'pred_end' : pred_end.cpu(),
                'true_start' : true_start.cpu(),
                'true_end' : true_end.cpu(),
                'offsets' : batch['offsets'],
                'context' : batch['context_text'],
                'answer' : batch['answer_text']
                })

            batch_metrics = qa_evaluate(to_eval)

            batch_metrics['loss'] = total_loss.item()
            
            #append all values of batch metrics to the corresponid element in metrics 
            for k,v in batch_metrics.items():
                metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        return utils.build_avg_dict('train',metrics)  #build dictionary suited for logging 

    
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

                total_loss = (start_loss + end_loss) / 2

                pred_start, pred_end = utils.compute_qa_predictions(pred_start_raw,pred_end_raw)

                to_eval = OrderedDict ({
                    'pred_start' : pred_start.cpu(),
                    'pred_end' : pred_end.cpu(),
                    'true_start' : true_start.cpu(),
                    'true_end' : true_end.cpu(),
                    'offsets' : batch['offsets'],
                    'context' : batch['context_text'],
                    'answer' : batch['answer_text']
                    })

                batch_metrics = qa_evaluate(to_eval)

                batch_metrics['loss'] = total_loss.item()
                
                #append all values of batch metrics to the corresponid element in metrics 
                for k,v in batch_metrics.items():
                    metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        return utils.build_avg_dict('val',metrics)

    
    def train_and_eval(self):

        best_val_f1 = 0.0
        model_save_path = 'models/'+self.model.get_model_name()+'.pt'

        train_dataloader, val_dataloader = self.dataloaders

        for epoch in range(self.run_param['n_epochs']):

            metrics = {}

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

            metrics.update(train_metrics)
            metrics.update(val_metrics)
            metrics['epoch'] = epoch+1 

            wandb.log(metrics)
            
            # save the best model 
            if val_metrics['val/f1'] >= best_val_f1:

                best_val_f1 = val_metrics['val/f1']
                if not os.path.exists('models'):        
                    os.makedirs('models')
            
                logger.info('saving model at epoch %d',epoch+1)
                torch.save(self.model.state_dict(), model_save_path)
            
        wandb.save(model_save_path)

class QG_handler : 

    def __init__(self, model_name, dataset_path, device):
    
        squad_dataset = RawSquadDataset(train_dataset_path = dataset_path)

        if model_name == 'BaseQG':

            self.data_manager : DataManager = RnnDataManagerQG(squad_dataset, device)

            N_EPOCHS = 15
            ENC_HIDDEN = 512
            DEC_HIDDEN = 512
            GRAD_CLIPPING = 1.0
            BATCH_SIZE = 64
            LR = 0.001
            DROPOUT = 0.3
            RANDOM_BATCH = False
            LR_SCHEDULER = False


            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.dropout = DROPOUT
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER

            pad_idx = self.data_manager.dec_tokenizer.token_to_id(globals.PAD_TOKEN)
            vocab_size = self.data_manager.dec_tokenizer.get_vocab_size()
            enc_embeddings = self.data_manager.enc_vectors
            dec_embeddings = self.data_manager.dec_vectors
            
            self.model = models.BaselineQg(enc_embeddings,dec_embeddings,ENC_HIDDEN,DEC_HIDDEN,vocab_size,pad_idx,DROPOUT,device)

            self.optimizer = optim.Adam(self.model.parameters(),lr=LR)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }

        elif model_name == 'RefNetQG':

            self.data_manager : DataManager = RnnDataManagerQG(squad_dataset, device)

            N_EPOCHS = 15
            ENC_HIDDEN = 256
            DEC_HIDDEN = 256
            GRAD_CLIPPING = 1.0
            BATCH_SIZE = 64
            LR = 0.001
            EPS = 1e-08
            DROPOUT = 0.5
            WEIGHT_DECAY = 0.01
            RANDOM_BATCH = False
            LR_SCHEDULER = False
            WARMUP = 2000

            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.epsilon = EPS
            wandb.config.dropout = DROPOUT
            wandb.config.weight_decay = WEIGHT_DECAY
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.lr_scheduler = LR_SCHEDULER
            wandb.config.warmup = WARMUP

            pad_idx = self.data_manager.dec_tokenizer.token_to_id(globals.PAD_TOKEN)
            vocab_size = self.data_manager.dec_tokenizer.get_vocab_size()
            enc_embeddings = self.data_manager.enc_vectors
            dec_embeddings = self.data_manager.dec_vectors
            
            self.model = models.RefNetQG(enc_embeddings,dec_embeddings,ENC_HIDDEN,DEC_HIDDEN,vocab_size,pad_idx,DROPOUT,device)

            self.optimizer = optim.Adam(self.model.parameters(), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }
        
        elif model_name == 'BertQG':

            self.data_manager : DataManager = BertDataManagerQG(squad_dataset, device) 
            
            N_EPOCHS = 4
            DEC_HIDDEN = 768
            GRAD_CLIPPING = 1.0
            BATCH_SIZE = 8
            LR = 3e-5
            EPS = 1e-08
            DROPOUT = 0.1
            WEIGHT_DECAY = 0.01
            RANDOM_BATCH = False
            LR_SCHEDULER = True
            WARMUP = 0
            FREEZE_ENC = False

            #log model configuration   
            wandb.config.n_epochs = N_EPOCHS
            wandb.config.grad_clipping = GRAD_CLIPPING
            wandb.config.batch_size = BATCH_SIZE
            wandb.config.learning_rate = LR
            wandb.config.epsilon = EPS
            wandb.config.dropout = DROPOUT
            wandb.config.weight_decay = WEIGHT_DECAY
            wandb.config.random_batch = RANDOM_BATCH
            wandb.config.dec_hidden = DEC_HIDDEN
            wandb.config.lr_scheduler = LR_SCHEDULER
            wandb.config.warmup = WARMUP
            wandb.config.freeze_enc = FREEZE_ENC

            pad_idx = self.data_manager.dec_tokenizer.token_to_id(globals.PAD_TOKEN)
            vocab_size = self.data_manager.dec_tokenizer.get_vocab_size()
            dec_embeddings = self.data_manager.dec_vectors


            self.model = models.BertQG(dec_embeddings, DEC_HIDDEN, vocab_size, FREEZE_ENC, pad_idx, DROPOUT, device)

            self.optimizer = AdamW(self.model.parameters(), lr=LR, eps=EPS, weight_decay=WEIGHT_DECAY)

            self.run_param = {
                'n_epochs' : N_EPOCHS,
                'grad_clipping' : GRAD_CLIPPING,
                'lr_scheduler' : LR_SCHEDULER
            }

        self.dataloaders = self.data_manager.get_dataloader('train', BATCH_SIZE, RANDOM_BATCH), self.data_manager.get_dataloader('val', BATCH_SIZE, RANDOM_BATCH)
        if LR_SCHEDULER: self.lr_scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=WARMUP, num_training_steps=N_EPOCHS * len(self.dataloaders[0]))
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)

        
    def train_loop(self, iterator):

        start_time = time.perf_counter()
        metrics = defaultdict(list)

        self.model.train()

        for batch in tqdm(iterator):

            #zero the gradients 
            self.optimizer.zero_grad()        

            raw_pred = self.model(batch)

            predictions = raw_pred[:,1:].contiguous().view(-1,raw_pred.shape[-1])
            ground_truth = batch['question_ids'][:,1:].contiguous().view(-1)

            loss = self.criterion(predictions,ground_truth) 
            
            #backward pass 
            loss.backward()
            
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.run_param['grad_clipping'])      

            #update the gradients
            self.optimizer.step()

            #update the learning rate
            if self.run_param['lr_scheduler']:
                self.lr_scheduler.step()

            pred = utils.compute_qg_predictions(raw_pred[:,1:])   

            batch_metrics = qg_evaluate(pred.cpu(),batch['question_ids'][:,1:].cpu(),batch['question_mask'][:,1:].cpu(),self.data_manager.dec_tokenizer)

            batch_metrics['loss'] = loss.item()
            
            #append all values of batch metrics to the corresponid element in metrics 
            for k,v in batch_metrics.items():
                metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        metrics['perplexity'] = torch.exp(torch.mean(torch.tensor(metrics['loss']))).item()

        return utils.build_avg_dict('train', metrics)

    
    def val_loop(self, iterator):

        start_time = time.perf_counter()
        metrics = defaultdict(list)

        self.model.eval()

        with torch.no_grad():
            
            for batch in tqdm(iterator):

                raw_pred = self.model(batch,teacher_force_ratio = 0)

                predictions = raw_pred[:,1:].contiguous().view(-1,raw_pred.shape[-1])
                ground_truth = batch['question_ids'][:,1:].contiguous().view(-1)

                loss = self.criterion(predictions,ground_truth) 

                pred = utils.compute_qg_predictions(raw_pred[:,1:])   

                batch_metrics = qg_evaluate(pred.cpu(),batch['question_ids'][:,1:].cpu(),batch['question_mask'][:,1:].cpu(),self.data_manager.dec_tokenizer)

                batch_metrics['loss'] = loss.item()
                
                #append all values of batch metrics to the corresponid element in metrics 
                for k,v in batch_metrics.items():
                    metrics[k].append(v)
        
        end_time = time.perf_counter()
        metrics['epoch_time'] = end_time-start_time

        metrics['perplexity'] = torch.exp(torch.mean(torch.tensor(metrics['loss']))).item()

        return utils.build_avg_dict('val', metrics)

    
    def train_and_eval(self):

        best_val_f1 = 0.0
        model_save_path = 'models/'+self.model.get_model_name()+'.pt'

        train_dataloader, val_dataloader = self.dataloaders

        for epoch in range(self.run_param['n_epochs']):

            metrics = {}

            logger.info('starting epoch %d',epoch+1)
            start_time = time.perf_counter()

            train_metrics = self.train_loop(train_dataloader)
            val_metrics = self.val_loop(val_dataloader)

            end_time = time.perf_counter()

            logger.info('epoch %d, tot time for train and eval: %f',epoch+1,end_time-start_time)
            logger.info('train: loss %f, f1 %f, num_acc %f, bleu %f, perplexity %f',
                        train_metrics["train/loss"], train_metrics["train/f1"],train_metrics["train/num_acc"], train_metrics["train/bleu"], 
                        train_metrics["train/perplexity"])
            logger.info('val: loss %f, f1 %f, num_acc %f, bleu %f, perplexity %f',
                        val_metrics["val/loss"], val_metrics["val/f1"],val_metrics["val/num_acc"], val_metrics["val/bleu"], 
                        val_metrics["val/perplexity"])           

            metrics.update(train_metrics)
            metrics.update(val_metrics)
            metrics['epoch'] = epoch+1 

            wandb.log(metrics)
        
            if val_metrics['val/f1'] >= best_val_f1:

                best_val_f1 = val_metrics['val/f1']
                if not os.path.exists('models'):        
                    os.makedirs('models')
            
                logger.info('saving model at epoch %d',epoch+1)
                torch.save(self.model.state_dict(), model_save_path)
            
        wandb.save(model_save_path)

