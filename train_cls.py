# -*- coding=utf-8 -*-

import shutil
import os
import argparse
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import BertClsDataset
from models.downstream import BertClassifier
from models.bert import Bert
from preprocessing import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)  # 사용할 gpu 선택
parser.add_argument('--model', '-m', type=str, default='bertcls')  # model 선택
parser.add_argument('--epochs', '-e', type=int, default=20)
parser.add_argument('--batch_size', '-bs', type=int, default=16)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-2)
parser.add_argument('--model_path', type=str, default='./cls_weights')  # 모델 저장 위치
parser.add_argument('--max_num_tokens', type=int, default=64)
parser.add_argument('--max_num_seq', type=int, default=50)
parser.add_argument('--d_hidden', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--n_groups', type=int, default=2)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=2000)
parser.add_argument('--n_class', type=int, default=5)
parser.add_argument('--bertlm_weights', type=str, default='lm_weights/bertlm_best.model')
parser.add_argument('--tokenizer_name', type=str, default='okt')

args = parser.parse_args()

# gpu 설정
if torch.cuda.is_available():
    device = 'cuda'
    torch.cuda.manual_seed_all(0)
else:
    device = 'cpu'
torch.manual_seed(0)
print('device:', device)

# 모델 세팅
base_model = Bert(args.d_hidden, args.n_heads, args.n_layers, args.n_groups,
                  args.vocab_size, args.max_num_tokens).to(device)
model = BertClassifier(base_model, args.d_hidden, args.n_class).to(device)

# pretrained 모델 로드
state_dict = torch.load(args.bertlm_weights)
if 'net' in state_dict:
    state_dict = state_dict['net']
bert_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('bert'):
        bert_state_dict[k[len('bert.'):]] = v

base_model.load_state_dict(bert_state_dict)  ## load pre-trained model

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# 학습 하이퍼파라미터 설정
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

# data loader 설정
suffix = f'v{args.vocab_size}_t{args.max_num_tokens}'
tokenizer = Tokenizer(tokenizer_name=args.tokenizer_name, prefix='vocab_{}'.format(args.vocab_size))

train_dataset = BertClsDataset(f'bertcls_train_{suffix}.json', tokenizer=tokenizer,
                               max_num_seq=args.max_num_seq, vocab_size=args.vocab_size, is_train=True)
val_dataset = BertClsDataset(f'bertcls_val_{suffix}.json', tokenizer=tokenizer,
                             max_num_seq=args.max_num_seq, vocab_size=args.vocab_size, is_train=False)


def collate(batch):
    #docs = [item[0][:args.max_num_seq] for item in batch]
    docs = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    labels = torch.LongTensor(labels)
    return [docs, labels]


train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

print(args)
# print(model)


def test():
    '''
    test_data_loader로 [Acc, loss] 테스트 수행
    '''
    model.eval()
    n_batch = len(val_data_loader)
    avg_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for i, (docs, labels) in enumerate(val_data_loader):  # iteration
            docs = [doc.to(device) for doc in docs]
            labels = labels.to(device)

            out = model(docs)
            loss = criterion(out, labels)  # cross_entropy
            avg_loss += loss.item() / n_batch

            correct += sum(torch.argmax(out, -1).cpu() == labels.cpu()).item()
            count += len(labels)
    return correct / count, avg_loss


def train():
    '''
    모델 학습 및 평가 수행
    '''
    min_loss = 999
    max_acc = 0
    st = time.time()
    for epoch in range(args.epochs):  # epoch
        n_batch = len(train_data_loader)
        avg_loss = 0
        model.train()
        for i, (docs, labels) in enumerate(train_data_loader):  # iteration
            docs = [doc.to(device) for doc in docs]
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(docs)
            loss = criterion(out, labels)  # cross_entropy
            loss.backward()
            avg_loss += loss.item() / n_batch
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            if i % 100 == 0:
                et = time.time()
                left_time = (et - st) * (args.epochs - epoch) * n_batch / 100
                print('\titer {}/{}, loss: {:.5f} [left time: {}h {}m {}s]'.format(i, n_batch, loss.item(),
                      int(left_time//3600), int((left_time%3600)//60), int(left_time%60)))
                st = time.time()
        print('epoch {}/{}, train_loss: {:.5f}'.format(epoch, args.epochs, avg_loss))

        # 모델 테스트
        acc, avg_loss = test()
        print('[TEST] acc: {:.5f}, loss: {:.5f}'.format(acc, avg_loss))

        if min_loss > avg_loss:  # 정확도 갱신 시, 모델 저장
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.model_path, f'{args.model}_best_t{args.max_num_tokens}.model'))
            print(f'\t - save model: {args.model}_best_t{args.max_num_tokens}.model, (loss: {min_loss} -> {avg_loss})')
            min_loss = avg_loss
        if acc > max_acc:  # 정확도 갱신 시, 모델 저장
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.model_path, f'{args.model}_best_acc.model'))
            print(f'\t - save model: {args.model}_best_acc.model, (acc: {max_acc} -> {acc})')
            max_acc = acc
        print('')
    # 학습 완료 후 모델 저장
    torch.save({
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.model_path, f'{args.model}_e{args.epochs}.model'))
    print(f'\t - save model: {args.model}_e{args.epochs}.model')


if __name__ == '__main__':
    train()