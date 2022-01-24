import os  
from argparse import ArgumentParser
from tqdm import tqdm
import json
import logging
from collections import namedtuple
import torch
import torch.nn as nn 

from src.data_handler import RawSquadDataset, DataManager, TransformerDataManager, RecurrentDataManager
import src.utils as utils 
import src.model as models
import src.globals as globals

logger = logging.getLogger(globals.LOG_NAME)

Container = namedtuple('Container', ['model','datamanager'])

QA_OBJECTS = {
    'DrQA': Container(models.DrQA, RecurrentDataManager),
    'BERTQA': Container(models.BertQA, TransformerDataManager),
    'ElectraQA': Container(models.ElectraQA, TransformerDataManager)   
}

def get_model_params(model_name : str):
    if model_name == 'DrQA':
        return {} #TODO 
    elif model_name == 'BERT':
        return {'device':utils.get_device()}
    elif model_name == 'Electra':
        return {'device':utils.get_device(),'hidden_dim':384,'freeze':False}


def generate_predictions(model : nn.Module , iterator):

    logger.info('model predictions will be saved in predictions.json file')
    predictions_file_path = os.path.join(globals.DATA_FOLDER,'predictions.json')
    
    predictions = {}

    model.eval()

    with torch.no_grad():

        for batch in tqdm(iterator):

            pred_start_raw, pred_end_raw = model(batch)

            pred_start, pred_end = utils.compute_predictions(pred_start_raw,pred_end_raw)

            pred_start_char = torch.gather(batch['offsets'][:,:,0],1,pred_start.unsqueeze(-1)).squeeze()
            pred_end_char = torch.gather(batch['offsets'][:,:,1],1,pred_end.unsqueeze(-1)).squeeze()

            pred_answers = [txt[s:e] for s,e,txt in zip(pred_start_char,pred_end_char,batch['context_text'])]

            batch_predictions = dict(zip(batch['question_alpha'],pred_answers))

            predictions.update(batch_predictions)


    logger.info('saving predictions.json file in data folder')
    with open(predictions_file_path,'w') as file:
        json.dump(predictions,file)
    logger.info('saved')


def main(dataset_path: str, model_name : str):

    assert os.path.splitext(dataset_path)[1] == '.json', 'The dataset file should be in json format'
    assert os.path.exists(f'models/{model_name}.pt'), f'The {model_name} trained model should be present in the models folder'

    #setups 
    utils.set_random_seed()
    logger = utils.setup_logging()
    device = utils.get_device()

    logger.info('starting evaluation pipeline')

    d = QA_OBJECTS[model_name].datamanager    #TODO change names 
    m = QA_OBJECTS[model_name].model

    test_dataset = RawSquadDataset(test_dataset_path = dataset_path)

    data_manager : DataManager = d(test_dataset,device) 

    test_dataloader = data_manager.get_dataloader('test', 8)

    model_param = get_model_params(model_name)

    logger.info('loading the model')
    best_model = m(**model_param)    
    best_model.load_state_dict(torch.load(f'models/{model_name}.pt', map_location=device))

    logger.info('runnning evaluation script on dataset')
    generate_predictions(best_model, test_dataloader)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_path',  help='path to json dataset file')
    parser.add_argument('-m', '--model', dest='model', choices=['DrQA','BERTQA','ElectraQA'], default='ElectraQA')
    args = parser.parse_args()

    main(args.dataset_path, args.model)