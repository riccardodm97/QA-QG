import os 
import logging 

import torch 
import wandb 

from argparse import ArgumentParser


import lib.globals as globals
import lib.utils as utils 
from lib.data_handling import RawSquadDataset, QA_DataManager, RecurrentDataManager
import lib.model as models
from lib.trainer import QATrainer

import torch.optim as optim
import torch.nn as nn 


def qa_trainer(model_name, dataset, device):

    dataset_path = os.path.join(globals.DATA_FOLDER,dataset)
    assert os.path.splitext(dataset_path)[1] == '.json', 'The dataset file should be in json format'

    squad_dataset = RawSquadDataset(dataset_path)

    if model_name == 'DrQA' : 

        data_manager : QA_DataManager = RecurrentDataManager(squad_dataset,device)

        HIDDEN_DIM = 128
        LSTM_LAYER = 3
        DROPUT = 0.3
        N_EPOCHS = 15
        GRAD_CLIPPING = 1
        BATCH_SIZE = 32
        RANDOM_BATCH = False
        
        model = models.DrQA(HIDDEN_DIM,LSTM_LAYER,DROPUT,data_manager.emb_model.vectors,data_manager.vocab[globals.PAD_TOKEN],device)

        optimizer = optim.Adamax(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)

        param = {
            'n_epochs' : N_EPOCHS,
            'grad_clipping' : GRAD_CLIPPING
        }

        trainer = QATrainer(model,optimizer,criterion,param,device)

        dataloaders = data_manager.get_dataloaders(BATCH_SIZE,RANDOM_BATCH)

        trainer.train_and_eval(dataloaders)
    
    elif model_name == 'BERT' :
        raise NotImplementedError()
    
    else :
        raise ValueError('The name of the model is neither DrQA nor BERT, those are the two only possibilities')



def main(task : str, model_name : str, dataset : str, log : bool):


    if task == 'qg' and model_name == 'DrQA' : 
        raise ValueError('Question Generation task cannot be performed with DrQA model, use BERT as value')

    #setup logging 
    log_path = os.path.join(globals.DATA_FOLDER, "log.txt")
    logger = logging.getLogger(globals.LOG_NAME)
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(log_path)
    #fileHandler.setLevel(logging.INFO)                      #TODO 
    formatter = logging.Formatter("%(name)s: %(message)s")
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #setup wandb 
    config = {

    }
    wandb.init(config = config, project="squad", entity="qa-qg")
    wandb.run.name = wandb.run.id        #set run name to run id 

    #TODO DISABLE WANDB SISYEM-WISE IF LOG IS FALSE 

    if task == 'qa':
        qa_trainer(model_name, dataset, device)
    elif task == 'qg':
        #qgtrainer()
        raise NotImplementedError()
    else :
        raise ValueError('The only two possible tasks are qg and qa, check the spelling')



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-t",  "--task", dest="task", help="Task to perform [Question Answering or Question Generation]", choices=['qa','qg'], required=True)
    parser.add_argument("-m", "--model", dest="model", help="Model to be trained", choices=['DrQA','BERT'], required=True)
    parser.add_argument("-d", "--dataset", dest="dataset", help ="the name of the file which contains the dataset", required=True, type = str)
    parser.add_argument("-l",  "--log", dest="log", help="Wheter to log on wandb or not", default=False, action='store_true')
    args = parser.parse_args()


    main(args.task,args.model,args.dataset,args.log)