import os  
import wandb 

from argparse import ArgumentParser

import lib.utils as utils 
import lib.exec_handler as exec


def main(task : str, model_name : str, dataset : str, log : bool):

    #checks on input
    assert os.path.splitext(dataset)[1] == '.json', 'The dataset file should be in json format'
    assert model_name == 'DrQA' or model_name == 'BERT', 'The only two possibilities for model name are DrQA or BERT'
    assert task == 'qa' or task == 'qg', 'The only two tasks available are qa (Question-Answering) or qg (Question-Generation)'
    assert not(task == 'qg' and model_name == 'DrQA'), 'Question Generation task cannot be performed with DrQA model, use BERT as value instead'
    
    #setups 
    utils.set_random_seed()
    logger = utils.setup_logging()
    device = utils.get_device() 

    #setup wandb 
    mode = None if not log else 'disabled'

    config = {
        'device': device,
        'task': task,
        'model_name': model_name,
        'dataset_file': dataset
    }

    wandb.init(config = config, project="squad", entity="qa-qg", mode=mode)
    wandb.run.name = utils.get_run_id()     #set run name 

    logger.info('starting run -> task: %s, model: %s , dataset file: %s, wandb enabled: %s',task,model_name,dataset,str(log))

    if task == 'qa':
        run_handler = exec.QA_handler(model_name, dataset, device)
        run_handler.train_and_eval()

    elif task == 'qg':
        raise NotImplementedError()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-t",  "--task", dest="task", help="Task to perform [Question Answering or Question Generation]", choices=['qa','qg'], required=True)
    parser.add_argument("-m", "--model", dest="model", help="Model to be trained", choices=['DrQA','BERT'], required=True)
    parser.add_argument("-d", "--dataset", dest="dataset", help ="the name of the file which contains the dataset", required=True, type = str)
    parser.add_argument("-l",  "--log", dest="log", help="Wheter to log on wandb or not", action='store_true')
    args = parser.parse_args()


    main(args.task,args.model,args.dataset,args.log)