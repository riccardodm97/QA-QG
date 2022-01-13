import os 
import logging 
import lib.globals as globals
import lib.utils as utils 

def run():

    log_path = os.path.join(globals.DATA_FOLDER,'log.txt')
    logging.basicConfig(filename=log_path,level='DEBUG')

    utils.load_embedding_model()


run()
