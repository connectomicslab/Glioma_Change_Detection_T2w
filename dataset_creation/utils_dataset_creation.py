import argparse
import json


__author__ = "Tommaso Di Noto"
__version__ = "0.0.1"
__email__ = "tommydino@hotmail.it"
__status__ = "Prototype"


def get_parser():
    """This function creates a parser for handling input arguments"""
    p = argparse.ArgumentParser(description='Aneurysm_Net')
    p.add_argument('--config', type=str, required=True, help='Path to json configuration file.')
    return p


def load_config_file():
    """This function loads the input config file
    Returns:
        config_dictionary (dict): it contains the input arguments
    """
    parser = get_parser()  # create parser
    args = parser.parse_args()  # convert argument strings to objects
    with open(args.config, 'r') as f:
        config_dictionary = json.load(f)

    return config_dictionary
