import os
import sys
import pandas as pd
import datetime
import pickle

import torch
from tqdm import tqdm

from model.metric import microF1

valid = False
threshold = 0.6
submission_path = "./data/task1_sample_submission.csv"


def main():
    #model_name = ["9sentence.csv", "7sentence.csv", "5sentence.csv", "3sentence.csv", "pad40.csv", "robertalarge_pad65_fb.csv", "robertalarge_pad65_ff.csv"]#["robertalarge_ped65.csv", "bertbase0.001.csv"]
    model_name = ["9sentence_prob.csv", "7sentence_prob.csv", "5sentence_prob.csv", "3sentence_prob.csv", "pad40_prob.csv", "roberta_pad65_ff-predict_prob.csv", "roberta_pad65_fg-predict_prob.csv"]#["robertalarge_ped65.csv", "bertbase0.001.csv"]
    #prob = [0.40, 0.40, 0.15, 0.15]
    #prob = [0.20, 0.20, 0.20, 0.20, 0.20]
    #prob = [0.25, 0.25, 0.25, 0.25, 0.00]
    prob = [1/7, 1/7, 1/7, 1/7, 1/7, 1/7, 1/7]
    #prob = [0, 0, 0 , 0 , 0, 0, 1]
    submission = pd.read_csv(submission_path)


    for weight, name in zip(prob, model_name):
        print("{} * {}".format(weight, name))
        current_csv = pd.read_csv(name)
        #print(current_csv)
        submission.iloc[:,1:] += current_csv.iloc[:,1:] * weight

    maxclass = submission.iloc[:,1:].idxmax(1) # make sure every sentence predicted to at least one class
    #predict_class = (predict_all > 0.5).type(torch.LongTensor).tolist()

    other_cnt = 0
    omit = 0
    for i in tqdm(range(submission.shape[0])):
        submission[maxclass.iloc[i]][i] = 1
        if submission["OTHERS"][i] == 1:
            other_cnt += 1
            #print(submission.iloc[i,:])
            submission.iloc[i,1:6] = 0
            #print(submission.iloc[i,:])
        else:
            if submission["OTHERS"][i] > threshold:
                omit += 1
            submission["OTHERS"][i] = 0
            

    print(other_cnt, omit)

    print("Current threshold: {}".format(threshold))
    submission.iloc[:,1:] = (submission.iloc[:,1:] > threshold).astype(int)


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
