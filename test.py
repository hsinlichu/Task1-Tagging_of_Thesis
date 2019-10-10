import os
import pickle
import csv
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

    # setup word embedding
    embedding_pkl_path = config["embedding"]["pkl_path"]
    if os.path.isfile(embedding_pkl_path):
        with open(embedding_pkl_path, "rb") as f:
            embedding = pickle.load(f)


    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        train_data_path=None,
        test_data_path=config['data_loader']['args']['test_data_path'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        num_workers=2,
        training=False,
        num_classes=6,
        embedding=embedding
    )

    # build model architecture
    model = config.init_obj('arch', module_arch, embedding=embedding)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        predict = [["order_id","BACKGROUND","OBJECTIVES","METHODS","RESULTS","CONCLUSIONS","OTHERS"]]
        for i, batch in enumerate(tqdm(data_loader)):
            data = batch["sentence"].to(device)
            number = batch["number"]
            output = model(data)
            predict_class = (output > 0.5).type(torch.LongTensor).tolist()
            result = [[i] + p_c for i, p_c in zip(number, predict_class) ]
            predict += result

        now = datetime.datetime.now()
        output_path = now.strftime("%m%d%H%M")+ "-predict.csv"

        with open(output_path, 'w', newline='\n') as myfile:
            wr = csv.writer(myfile)
            for r in predict:
                wr.writerow(r)

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
