import os  

from argparse import ArgumentParser

from src.data_handler import RawSquadDataset, DataManager, RecurrentDataManager, TransformerDataManager
import src.utils as utils 


def main(dataset_path: str):

    assert os.path.splitext(dataset_path)[1] == '.json', 'The dataset file should be in json format'

    #setups 
    utils.set_random_seed()
    logger = utils.setup_logging()
    device = utils.get_device()

    logger.info('running evaluation script')

    test_dataset = RawSquadDataset(test_dataset_path = dataset_path)

    data_manager : DataManager = TransformerDataManager(test_dataset, device)















if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_path',  help='path to json dataset file')
    args = parser.parse_args()


    main(args.dataset_path)