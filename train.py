# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/19
# File Name: train.py
# Edit Author: lnest
# ------------------------------------
import json
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from model.qa_net import QANet
from utility.data_loader import get_dataloader
from utility.use_logger import set_log_level

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_log_level(logging.DEBUG)
logger = logging.getLogger()


def train():
    # word_map = json.load(open('word_map.json'))
    word_mat = json.load(open('./data/emb_mat.json'))

    logger.info('get dataloader....')
    train_dataset = get_dataloader('./data/dl_train.json')
    dev_dataset = get_dataloader('./data/dl_dev.json')
    logger.info('Complete!')
    model = QANet(word_mat).to(device)
    criterion = nn.CrossEntropyLoss()

    parameters = filter(lambda param: param.requires_grad, model.parameters())
    optimizer = optim.SGD(parameters, lr=1e-2)

    for epoch in range(20):
        optimizer.zero_grad()
        running_loss = .0
        for i, batch_data in enumerate(train_dataset):
            passage = batch_data['passage'].to(device)
            query = batch_data['query'].to(device)
            answer = batch_data['answer'].to(device)
            logits = model(passage, query)
            loss = criterion(logits, answer)
            loss.backward()
            running_loss += loss.item()
            if i % 2000 == 1999:
                predict = model.predict(passage, query)
                score = evaluate(predict, answer)
                print('[%d, %5d] loss: %.3f\tscore: %.5f' % (epoch + 1, i + 1, running_loss / 2000, score))
                running_loss = 0.0
        dev_score = dataset_evel(dev_dataset, model)
        print('epoch: %d\tloss: %.3f\tdev_score: %.5f' % (epoch + 1, running_loss / 2000, dev_score))
        torch.save(model.state_dict(), './model/encoder.pkl_' % epoch)


def evaluate(predict, answer):
    cnt = 0
    match = 0
    for pre, ans in zip(predict, answer):
        if pre == ans:
            match += 1
        cnt += 1
    return match / cnt


def dataset_evel(dataset, model):
    score = 0.0
    batch_idx = 0
    for batch_data in dataset:
        passage = batch_data['passage'].to(device)
        query = batch_data['query'].to(device)
        answer = batch_data['answer'].to(device)
        predict = model.predict(passage, query)
        b_score = evaluate(predict, answer)
        score += b_score
        batch_idx += 1
    return score / (batch_idx + 1)


if __name__ == '__main__':
    train()
