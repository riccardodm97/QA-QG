import os 
import logging 
import lib.globals as globals
import lib.utils as utils 

def run():

    logger = logging.getLogger(globals.LOG_NAME)
    logger.setLevel(logging.INFO)

    log_path = os.path.join(globals.DATA_FOLDER, "log.txt")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(name)s: %(message)s")

    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    utils.load_embedding_model()


run()
