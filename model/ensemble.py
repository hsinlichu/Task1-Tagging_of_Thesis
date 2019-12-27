import os
import pandas as pd
import pickle
import datetime

import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

        predict = []
        for i, batch in enumerate(tqdm(data_loader)):
            data = batch["sentence"]
            number = batch["number"]

            if not isinstance(data, list):   
                data = data.to(device)

            output = model(data)
            if isinstance(output, list):   
                output = torch.cat(output, dim=0).to(device)

            predict.append(output)

        predict_all = torch.cat(predict)
        predict_class = (predict_all > 0.5).type(torch.LongTensor).tolist()
        maxclass = torch.argmax(predict_all, dim=1).tolist() # make sure every sentence predicted to at least one class

    logger.info("Convert output array to submission format. ")
    submission = pd.read_csv(config["test"]["sample_submission_file_path"])
    logger.info("predict array len: {}".format(len(predict_class)))

    for i in tqdm(range(len(predict_class))):
        predict_class[i][maxclass[i]] = 1
    submission.iloc[:len(predict_class),1:] = predict_class

    now = datetime.datetime.now()
    output_path = now.strftime("%m%d%H%M")+ "-predict.csv"
    logger.info("Submission file save to {}".format(output_path))
    submission.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
