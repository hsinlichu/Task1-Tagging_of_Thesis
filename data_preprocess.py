import os
from pprint import pprint

import pandas as pd
import pickle
import argparse
import logging
import math


label_encoding = {"BACKGROUND": 0,"OBJECTIVES": 1,"METHODS": 2,"RESULTS": 3, "CONCLUSIONS": 4,"OTHERS": 5}

training_data_save_path = os.path.join("data", "train_processed.pkl")
testing_data_save_path = os.path.join("data", "test_processed.pkl")


def main(args):
    trainingdata = args.train
    if trainingdata != None and os.path.isfile(trainingdata):
        logging.info('Process Training Data...')

        data_processed = []
        data_df = pd.read_csv(trainingdata)
        data_select = data_df[['Abstract', 'Task 1']] 
        data_select = data_select.sample(frac=1, random_state=123).reset_index(drop=True) # Shuffle training data with fixed seed

        for index, row in data_select.iterrows():
            sentence_array = row['Abstract'].split("$$$")
            label_array = [[label_encoding[label] for label in labels.split("/")] for labels in row['Task 1'].split(" ")]

            article = [{"number": "T{:05d}_S{:03d}".format(index + 1, sentence_index + 1) ,"sentence": sentence, "label": label} for sentence_index, sentence, label in zip(range(len(sentence_array)), sentence_array, label_array)]   
            data_processed.append(article)

        train_processed = data_processed

        #pprint(train_processed[:1])
        logging.info("Number of training data: {}".format(len(train_processed)))
        with open(training_data_save_path, "wb") as f:
            pickle.dump(train_processed, f)
            logging.info("Processed training data save to {}".format(training_data_save_path))
                            
    testingdata = args.test
    if testingdata != None and os.path.isfile(testingdata):
        logging.info('Process Testing Data...')

        test_processed = []
        test_df = pd.read_csv(testingdata)
        test_select = test_df[['Abstract']] 

        for index, row in test_select.iterrows():
            sentence_array = row['Abstract'].split("$$$")
            article = [{"number": "T{:05d}_S{:03d}".format(index + 1, sentence_index + 1) ,"sentence": sentence} for sentence_index, sentence in zip(range(len(sentence_array)), sentence_array)]   
            test_processed.append(article)

        #pprint(test_processed[:1])
        logging.info("Number of testing data: {}".format(len(test_processed)))

        with open(testing_data_save_path, "wb") as f:
            pickle.dump(test_processed, f)
            logging.info("Processed testing data save to {}".format(testing_data_save_path))
         
                
         


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    parser = argparse.ArgumentParser(description='Preprocess Data')
    parser.add_argument('--train', default=None, type=str,
                      help='path to training data (default: None)')
    parser.add_argument('--test', default=None, type=str,
                      help='path to testing data (default: None)')
    args = parser.parse_args()

    main(args)
