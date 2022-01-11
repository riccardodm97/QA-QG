
import os 

import lib.globals as globals 

import numpy as np
import gensim.downloader as gloader
from gensim.models import KeyedVectors

import logging



logger = logging.getLogger(__name__)

def load_embedding_model():
    """
    Loads a pre-trained word embedding model via gensim library

    """
    model_name = "glove-wiki-gigaword-{}".format(globals.EMBEDDING_DIMENSION)
    glove_model_path = os.path.join(globals.DATA_FOLDER, "glove_vectors.txt") 

    try:

        #if already stored in data, retrieve it 
        if os.path.exists(glove_model_path): 

            logging.info('loading embedding vectors from file')
            embedding_model = KeyedVectors.load_word2vec_format(glove_model_path, binary=True)
        
        else:
            embedding_model : KeyedVectors = gloader.load(model_name)

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

        return embedding_model, embedding_model.key_to_index
        
    except Exception as e:
        raise e('Error')