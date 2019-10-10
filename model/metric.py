import torch


def accuracy(output, target):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        predict = (output > 0.5).type(torch.LongTensor)
        predict, target = predict.to(device), target.to(device)

        correct = 0
        for p_r, t_r in zip(predict, target):
            if torch.equal(t_r, p_r):
                correct += 1
        ret = correct / len(target)
    return ret


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)
