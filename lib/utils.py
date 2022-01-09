
import lib.globals as glob

import numpy as np

import gensim.downloader as gloader
from gensim.models import KeyedVectors


def load_embedding_model():
    """
    Loads a pre-trained word embedding model via gensim library
    """

    model = "glove-wiki-gigaword-{}".format(glob.EMBEDDING_DIMENSION)
    try:
        embedding_model : KeyedVectors = gloader.load(model)

        # unknown vector as the mean of all vectors
        assert glob.UNK_TOKEN not in embedding_model, f"{glob.UNK_TOKEN} key already present"
        unk = np.mean(embedding_model.vectors, axis=0)
        if unk in embedding_model.vectors:
            np.random.uniform(low=-0.05, high=0.05, size=glob.EMBEDDING_DIMENSION)      

        # pad vector as a zero vector
        assert glob.PAD_TOKEN not in embedding_model, f"{glob.PAD_TOKEN} key already present"
        pad = np.zeros((embedding_model.vectors.shape[1],))

        embedding_model.add_vectors([glob.UNK_TOKEN,glob.PAD_TOKEN], [unk,pad])

        return embedding_model, embedding_model.key_to_index
        
    except Exception as e:
        raise e('Error')