import os
import sys
import random 
import logging
import time 
from datetime import datetime
import requests
import pytz 

import pandas as pd 
import numpy as np
import torch 
import torch.nn.functional as F

import gensim.downloader as gloader
from gensim.models import KeyedVectors

import src.globals as globals 


logger = logging.getLogger(globals.LOG_NAME)


def load_glove_embedding():
    """
    Loads a pre-trained word embedding model via gensim library

    """
    start_time = time.perf_counter()

    model_name = "glove-wiki-gigaword-{}".format(globals.EMBEDDING_DIMENSION)

    logger.info('downloading glove model (dim = %s)...',globals.EMBEDDING_DIMENSION)
    embedding_model : KeyedVectors = gloader.load(model_name)
    logger.info('glove loaded')
    
    end_time = time.perf_counter()
    logger.info('downloaded in: %f',end_time-start_time)

    return embedding_model

def get_Glove_model_and_vocab():
    """
    Retrive GloVe embedding model, adds special tokens to the vocab and returns it 

    """
    glove_model_path = os.path.join(globals.DATA_FOLDER, f"glove_vectors_{globals.EMBEDDING_DIMENSION}.txt")
    
    #if already stored in data, retrieve it 
    if os.path.exists(glove_model_path): 

        logger.info('loading vectors (dim = %s) from file',globals.EMBEDDING_DIMENSION)
        embedding_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=True)
    
    else:
        embedding_model = load_glove_embedding()

        # unknown vector as the mean of all vectors
        assert globals.UNK_TOKEN not in embedding_model, f"{globals.UNK_TOKEN} key already present"
        unk = np.mean(embedding_model.vectors, axis=0)
        if unk in embedding_model.vectors:
            unk = np.random.uniform(low=-0.05, high=0.05, size=globals.EMBEDDING_DIMENSION)      

        # pad vector as a zero vector
        assert globals.PAD_TOKEN not in embedding_model, f"{globals.PAD_TOKEN} key already present"
        pad = np.zeros((embedding_model.vectors.shape[1],))

        #add newly created vectors to the model
        embedding_model.add_vectors([globals.UNK_TOKEN,globals.PAD_TOKEN], [unk,pad])

        embedding_model.allocate_vecattrs()    # library bug ?? 

        embedding_model.save_word2vec_format(glove_model_path, binary=True)
        logger.info('glove model saved to file in data directory')
    
    return embedding_model, embedding_model.key_to_index


def build_embedding_matrix(type : str, vocab : dict) -> np.ndarray:
    '''
    Builds an embedding matrix from GloVe vectors given an already defined vocabulary 
    '''

    assert type in ['encoder','decoder']
    emb_matrix_path = os.path.join(globals.DATA_FOLDER, f"{type}_emb_matrix.npy")

    if os.path.exists(emb_matrix_path): 
        logger.info(f'loading {type} embedding matrix from file')
        embedding_matrix = np.load(emb_matrix_path,allow_pickle=True)
    
    else : 
        logger.info(f'building {type} embedding matrix...')

        emb_model = load_glove_embedding()
        assert emb_model is not None, 'WARNING: empty embeddings model'

        embedding_dimension = emb_model.vector_size      #how many numbers each emb vector is composed of                                                           
        embedding_matrix = np.zeros((len(vocab), embedding_dimension), dtype=np.float32)   #create a matrix initialized with all zeros 

        for word, idx in vocab.items():
            if idx<1 : continue      #skip the pad token 
            try:
                embedding_vector = emb_model[word]
            except (KeyError, TypeError):
                embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

            embedding_matrix[idx] = embedding_vector    #assign the retrived or the generated vector to the corresponding index 
        
        unk = np.mean(emb_model.vectors, axis=0)
        if unk in emb_model.vectors:
            unk = np.random.uniform(low=-0.05, high=0.05,size=embedding_dimension)    

        embedding_matrix[vocab[globals.UNK_TOKEN]] = unk      # add the unk token embedding  

        logger.info(f"built embedding matrix with shape: {embedding_matrix.shape}")

        np.save(emb_matrix_path,embedding_matrix,allow_pickle=True)
        logger.info('embedding matrix saved to file in data directory')

    return embedding_matrix


def load_bert_vocab():

    logger.info('downloading BERT vocab from huggingface ...')
    VOCAB_PATH = os.path.join(globals.DATA_FOLDER,globals.BERT_PRETRAINED+'-vocab.txt')

    response = requests.get("https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt")
    with open(VOCAB_PATH, mode='wb') as localfile:
        localfile.write(response.content)

    logger.info('loaded and stored in data folder')


def set_random_seed():
    '''
    Set the random seed for the entire environment 
    '''

    torch.manual_seed(globals.RND_SEED)
    random.seed(globals.RND_SEED)
    np.random.seed(globals.RND_SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    '''
    Create a run id based on current date and time for wandb logging puroposes 
    '''
    return datetime.now(tz = pytz.timezone('Europe/Rome')).strftime("%d/%m/%Y %H:%M:%S") 

 
def compute_qa_predictions(starts,ends):   
    '''
    Generates single numerical predictions from raw distribution coming from the model output 
    ''' 
    
    _, len_txt = starts.size()

    pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)

    sum_on_row = pred_start_logit.unsqueeze(2) + pred_end_logit.unsqueeze(1)        # [batch_size, text_length, text_length]

    mask = (torch.ones(len_txt, len_txt) * float('-inf')).to(get_device()).tril(-1)      # -inf on lower diagonal and 0 on upper diagonal

    out = sum_on_row + mask                         #mask illegal positions (start > end)

    maximum_row, _ = torch.max(out, dim=2)          #maximum values on row
    maximum_col, _ = torch.max(out, dim=1)          #maximum values on columns
    s_idx = torch.argmax(maximum_row, dim=1)        #row index of maximum per example in batch 
    e_idx = torch.argmax(maximum_col, dim=1)        #column index of maximum per example in batch

    # # alternative 
    # pred_start_logit, pred_end_logit = F.log_softmax(starts,dim=1), F.log_softmax(ends,dim=1)
    # s_idx, e_idx = torch.argmax(pred_start_logit,dim=1), torch.argmax(pred_end_logit,dim=1)

    return s_idx, e_idx

def compute_qg_predictions(raw_pred):

    pred_logit = F.softmax(raw_pred, dim=2)

    return pred_logit.argmax(dim=2) 


def build_avg_dict(mode : str, metrics : dict) -> dict :
    '''
    Construct a dictionary which average down all the lists to a single value and adds to the keys the name of the mode (train-val) for logging purposes
    '''

    def prepend_mode(key : str):
        return mode + '/' + key
    
    def cond_mean(value):
        if isinstance(value,list):
            return np.mean(value).round(3)
        else : return np.round(value,5)

    return {prepend_mode(k): cond_mean(v) for k,v in metrics.items()}


def remove_errors(df : pd.DataFrame):
    '''
    Remove uncorrect rows from the passed dataframe and returns the cleaned one
    '''

    error_ids = open(os.path.join(globals.DATA_FOLDER,'error_ids.txt')).read().splitlines()
    
    return df[~df['question_id'].isin(error_ids)]



