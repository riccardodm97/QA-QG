import os  
from argparse import ArgumentParser
from tqdm import tqdm
import json
import logging

import torch
import torch.nn as nn 

from src.data_handler import RawSquadDataset, DataManager, RecurrentDataManager, TransformerDataManager
import src.utils as utils 
import src.model as models
import src.globals as globals

logger = logging.getLogger(globals.LOG_NAME)


def generate_predictions(model : nn.Module , iterator):

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

            batch_predictions = dict(zip(batch['question_ids'],pred_answers))

            predictions.update(batch_predictions)


    with open(predictions_file_path,'w') as file:
        json.dump(predictions,file)



def main(dataset_path: str):

    assert os.path.splitext(dataset_path)[1] == '.json', 'The dataset file should be in json format'

    #setups 
    utils.set_random_seed()
    logger = utils.setup_logging()
    device = utils.get_device()

    logger.info('running evaluation script')

    test_dataset = RawSquadDataset(test_dataset_path = dataset_path)

    data_manager : DataManager = TransformerDataManager(test_dataset, device)

    best_model = models.BertQA(device)

    test_dataloader = data_manager.get_dataloader('test', 8)

    generate_predictions(best_model, test_dataloader)



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_path',  help='path to json dataset file')
    args = parser.parse_args()

    main(args.dataset_path)