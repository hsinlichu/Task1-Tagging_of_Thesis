import os
from pprint import pprint

import pandas as pd
import pickle
import argparse
import logging
import math


training_data_save_path = os.path.join("data", "lmft_train.txt")
validation_data_save_path = os.path.join("data", "lmft_val.txt")


def main(args):
    
    data = args.train
    if data != None and os.path.isfile(data):
        logging.info('Process Data...')

        Val_ratio = 0.1
        data_df = pd.read_csv(data)
        data_select = data_df[['Abstract']].sample(frac=1).reset_index(drop=True) # select only Abstruct column and shuffle 
        train_data = ""
        val_data = ""
        dataset_len = len(data_select)

        for i in range(dataset_len):
            article = data_select["Abstract"][i].replace("$$$", " ").lower()
            if i < Val_ratio * dataset_len:
                val_data += " " + article
            else:
                train_data += " " article

        with open(training_data_save_path, "w") as f:
            f.write(train_data)
            logging.info("Processed training data save to {}".format(training_data_save_path))
        with open(validation_data_save_path, "w") as f:
            f.write(val_data)
            logging.info("Processed validation data save to {}".format(validation_data_save_path))

         
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('--train', default=None, type=str,
                      help='path to data (default: None)')
    args = parser.parse_args()

    main(args)
