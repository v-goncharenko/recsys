from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.quick_start import run_recbole


if __name__ == '__main__':
    config = Config(model='EASE', config_file_list=['lastfm.yaml'])
    run_recbole(model=config['model'], dataset=config['dataset'], config_file_list=['lastfm.yaml'])
