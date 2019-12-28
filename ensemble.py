import os
import sys
import pandas as pd
import datetime
import pickle

import torch
from tqdm import tqdm

from model.metric import microF1

valid = True


def main():
    model_name = ["12290059-valid_predict.csv"]#["robertalarge_ped65.csv", "bertbase0.001.csv"]
    prob = [1]
    submission = pd.read_csv(sys.argv[1])


    for weight, name in zip(prob, model_name):
        current_csv = pd.read_csv(name)
        submission.iloc[:,1:] += current_csv.iloc[:,1:] * weight

    maxclass = submission.iloc[:,1:].idxmax(1) # make sure every sentence predicted to at least one class
    #predict_class = (predict_all > 0.5).type(torch.LongTensor).tolist()

    for i in tqdm(range(submission.shape[0])):
        submission[maxclass.iloc[i]][i] = 1
    submission.iloc[:,1:] = (submission.iloc[:,1:] > 0.5).astype(int)



    if valid: 
        gt_path = "./valid_gt.pkl"
        with open(gt_path, "rb") as fin:
            print("Open validation ground truth file from {}".format(gt_path))
            gt_all = pickle.load(fin)
        #print(gt_all)

        print("How many validation data: {}".format(gt_all.size(0)))
        predict = torch.tensor(submission.iloc[:gt_all.size(0), 1:].values)
        #print(predict)

        score = microF1(predict, gt_all)
        print("Your Score: {}".format(score))
    else:
        
        now = datetime.datetime.now()
        output_path = now.strftime("%m%d%H%M")+ "-predict_ensemble.csv"
        print("Submission file save to {}".format(output_path))
        submission.to_csv(output_path, index=True)

if __name__ == '__main__':
    main()
