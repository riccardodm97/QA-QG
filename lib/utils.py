
import os 

import lib.globals as globals 

import numpy as np
import torch 
import random 
import logging

from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler

import gensim.downloader as gloader
from gensim.models import KeyedVectors

logger = logging.getLogger(globals.LOG_NAME)

def load_embedding_model():
    """
    Loads a pre-trained word embedding model via gensim library

    """
    model_name = "glove-wiki-gigaword-{}".format(globals.EMBEDDING_DIMENSION)
    glove_model_path = os.path.join(globals.DATA_FOLDER, "glove_vectors.txt")

    try:

        #if already stored in data, retrieve it 
        if os.path.exists(glove_model_path): 

            logger.info('loading embedding vectors from file')
            embedding_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=True)
        
        else:
            logger.info('downloading glove model...')
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

            embedding_model.allocate_vecattrs()  #TODO perchè ? 

            embedding_model.save_word2vec_format(glove_model_path, binary=True)
            logger.info('glove model saved to file in data directory')

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

        return embedding_layer, embedding_dim

