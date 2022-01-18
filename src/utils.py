
import os
import sys
import random 
import logging
import time 
from datetime import datetime
import requests

import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler
import torch.nn.functional as F

import gensim.downloader as gloader
from gensim.models import KeyedVectors

from src.data_handler import RawSquadDataset
import src.globals as globals 


logger = logging.getLogger(globals.LOG_NAME)

def load_embedding_model():
    """
    Loads a pre-trained word embedding model via gensim library

    """
    start_time = time.perf_counter()

    model_name = "glove-wiki-gigaword-{}".format(globals.EMBEDDING_DIMENSION)
    glove_model_path = os.path.join(globals.DATA_FOLDER, f"glove_vectors_{globals.EMBEDDING_DIMENSION}.txt")

    try:

        #if already stored in data, retrieve it 
        if os.path.exists(glove_model_path): 

            logger.info('loading embedding vectors (dim = %s) from file',globals.EMBEDDING_DIMENSION)
            embedding_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=True)
        
        else:
            logger.info('downloading glove model (dim = %s)...',globals.EMBEDDING_DIMENSION)
            embedding_model : KeyedVectors = gloader.load(model_name)
            logger.info('glove loaded')

            # unknown vector as the mean of all vectors
            assert globals.UNK_TOKEN not in embedding_model, f"{globals.UNK_TOKEN} key already present"
            unk = np.mean(embedding_model.vectors, axis=0)
            if unk in embedding_model.vectors:
                np.random.uniform(low=-0.05, high=0.05, size=globals.EMBEDDING_DIMENSION)      

            # pad vector as a zero vector
            assert globals.PAD_TOKEN not in embedding_model, f"{globals.PAD_TOKEN} key already present"
            pad = np.zeros((embedding_model.vectors.shape[1],))

            #add newly created vectors to the model
            embedding_model.add_vectors([globals.UNK_TOKEN,globals.PAD_TOKEN], [unk,pad])

            embedding_model.allocate_vecattrs()  #TODO why ? library bug ? 

            embedding_model.save_word2vec_format(glove_model_path, binary=True)
            logger.info('glove model saved to file in data directory')
        
        end_time = time.perf_counter()
        logger.info('loading time: %f',end_time-start_time)

        return embedding_model, embedding_model.key_to_index
        
    except Exception as e:
        raise e('Error')
    

def set_random_seed():

    torch.manual_seed(globals.RND_SEED)
    random.seed(globals.RND_SEED)
    np.random.seed(globals.RND_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

 
def build_dataloader(dataset, batch_size : int, random : bool):

    if random : 
        sampler = BatchSampler(RandomSampler(dataset), batch_size=batch_size, drop_last=False)
    else:
        sampler = BatchSampler(SequentialSampler(dataset), batch_size=batch_size, drop_last=False)
    
    return DataLoader(dataset,sampler=sampler,batch_size=None)


def get_embedding_layer(weights_matrix : np.ndarray , pad_idx : int, device = 'cpu'):

    matrix = torch.from_numpy(weights_matrix).to(device) 
        
    _ , embedding_dim = matrix.shape
    embedding_layer = nn.Embedding.from_pretrained(matrix, freeze = False, padding_idx = pad_idx)   #load pretrained weights in the layer and make it non-trainable

    def tune_embedding(grad, words=1000):
            grad[words:] = 0
            return grad
        
    embedding_layer.weight.register_hook(tune_embedding)   

    return embedding_layer, embedding_dim


def setup_logging():
    """
    Setup logging, mainly for debug purposes 
    """

    log_path = os.path.join(globals.DATA_FOLDER, "log.txt")
    logger = logging.getLogger(globals.LOG_NAME)
    logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler(log_path, mode='w')
    consoleHandler = logging.StreamHandler(sys.stdout) 
    formatter = logging.Formatter("%(message)s")
    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger 


def get_device():
    """
    Return a CUDA device, if available, or CPU otherwise
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def get_run_id():

    return datetime.now().strftime("%d/%m/%Y %H:%M:%S") 

 
def compute_predictions(starts,ends):    #TODO come calcolarle ? 

    # pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)
    # s_idx, e_idx = torch.argmax(pred_start_logit,dim=1), torch.argmax(pred_end_logit,dim=1)


    batch_size, c_len = starts.size()
    ls = nn.LogSoftmax(dim=1)
    mask = (torch.ones(c_len, c_len) * float('-inf')).to(get_device()).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)    
    
    score = (ls(starts).unsqueeze(2) + ls(ends).unsqueeze(1)) + mask
    score, s_idx = score.max(dim=1)
    score, e_idx = score.max(dim=1)
    s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

    return s_idx, e_idx


def compute_avg_dict(mode : str, metrics : dict) -> dict :

    def prepend_mode(key : str):
        return mode + '/' + key
    
    def cond_mean(value):
        if isinstance(value,list):
            return np.mean(value).round(3)
        else : return np.round(value,3)

    return {prepend_mode(k): cond_mean(v) for k,v in metrics.items()}


def remove_errors(dataset : RawSquadDataset):

    error_ids = open(os.path.join(globals.DATA_FOLDER,'error_ids.txt')).read().splitlines()
    
    return dataset.train_df[~dataset.train_df['question_id'].isin(error_ids)]


def load_bert_vocab():
    VOCAB_PATH = os.path.join(globals.DATA_FOLDER,globals.BERT_PRETRAINED+'-vocab.txt')

    response = requests.get("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")
    with open(VOCAB_PATH, mode='wb') as localfile:
        localfile.write(response.content)
