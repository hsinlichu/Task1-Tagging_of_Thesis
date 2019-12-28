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
    logger = config.get_logger('valid')

    # setup data_loader instances
    with open("valid_dataloader.pkl", "rb") as fin:
        data_loader = pickle.load(fin)

    # build model architecture
    model = config.init_obj('arch', module_arch, embedding=None)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        gt = []
        predict = []
        for i, batch in enumerate(tqdm(data_loader)):
            data = batch["sentence"]
            number = batch["number"]
            label = batch["label"]
            gt += label

            if not isinstance(data, list):   
                data = data.to(device)

            output = model(data)
            if isinstance(output, list):   
                output = torch.cat(output, dim=0).to(device)

            predict.append(output)

        predict_all = torch.cat(predict).cpu()

        output_gt = False
        if output_gt:
            #print(gt)
            gt_all = torch.cat(gt)
            gt_path = "valid_gt.pkl"
            with open(gt_path, "wb") as fout:
                print("Save validation ground truth file to {}".format(gt_path))
                pickle.dump(gt_all, fout)

    logger.info("Convert output array to submission format. ")
    submission = pd.read_csv(config["test"]["sample_submission_file_path"])
    logger.info("predict array len: {}".format(len(predict_all)))

    submission.iloc[:predict_all.size(0),1:] = predict_all

    now = datetime.datetime.now()
    output_path = now.strftime("%m%d%H%M")+ "-valid_predict.csv"
    logger.info("Validation predict file save to {}".format(output_path))
    submission.to_csv(output_path, index=False)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')

    config = ConfigParser.from_args(args)
    main(config)
