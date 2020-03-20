# -*- coding=utf-8 -*-

import shutil
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets import BertLMDataset
from models.downstream import BertLM
from models.bert import Bert
from utils.scheduler import ScheduledOptim
from preprocessing import Tokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)  # 사용할 gpu 선택 (data parallel 사용 시 무시됨)
parser.add_argument('--model', '-m', type=str, default='bertlm')  # model 선택
parser.add_argument('--epochs', '-e', type=int, default=1000)
parser.add_argument('--batch_size', '-bs', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-3)
parser.add_argument('--model_path', type=str, default='./lm_weights')  # 모델 저장 위치
parser.add_argument('--max_num_tokens', type=int, default=64)
parser.add_argument('--d_hidden', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--n_groups', type=int, default=2)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=2000)
parser.add_argument('--warmup_steps', type=int, default=10000)
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
model = BertLM(base_model, args.d_hidden, args.vocab_size).to(device)

if not os.path.exists(args.model_path):
    os.makedirs(args.model_path)

# 학습 하이퍼파라미터 설정
optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optim_schedule = ScheduledOptim(optimizer, args.d_hidden, n_warmup_steps=args.warmup_steps)
def mask_ce(logits, labels, mask):
    x = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(x.transpose(1,2), labels, reduction='none') * mask
    return loss.sum() / (mask.sum() + 1e-5)
criterion = mask_ce

# data loader 설정
suffix = f'v{args.vocab_size}_t{args.max_num_tokens}'

tokenizer = Tokenizer(tokenizer_name=args.tokenizer_name, prefix='vocab_{}'.format(args.vocab_size))

train_dataset = BertLMDataset(f'bertlm_train_{suffix}.json', tokenizer=tokenizer)
val_dataset = BertLMDataset(f'bertlm_val_{suffix}.json', tokenizer=tokenizer)

train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_data_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

print(args)
# print(model)


def test():
    '''
    test_data_loader로 [Acc, loss] 테스트 수행
    '''
    model.eval()
    n_batch = len(val_data_loader)
    avg_loss = 0
    with torch.no_grad():
        for i, (tokens, mask, ans) in enumerate(val_data_loader):  # iteration
            tokens = tokens.to(device)
            mask = mask.to(device)
            ans = ans.to(device)

            out = model(tokens)
            loss = criterion(out, ans, mask)  # cross_entropy
            avg_loss += loss.item() / n_batch
    return avg_loss


def train():
    '''
    모델 학습 및 평가 수행
    '''
    min_loss = 999
    for epoch in range(args.epochs):  # epoch
        n_batch = len(train_data_loader)
        avg_loss = 0
        model.train()
        for i, (tokens, mask, ans) in enumerate(train_data_loader):  # iteration
            tokens = tokens.to(device)
            mask = mask.to(device)
            ans = ans.to(device)

            optim_schedule.zero_grad()
            out = model(tokens)
            loss = criterion(out, ans, mask)  # cross_entropy
            loss.backward()
            avg_loss += loss.item() / n_batch
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optim_schedule.step_and_update_lr()

            if i % 100 == 0:
                print('\titer {}/{}, loss: {:.5f}'.format(i, n_batch, loss.item()))
        print('epoch {}/{}, train_loss: {:.5f}'.format(epoch, args.epochs, avg_loss))

        # 모델 테스트
        avg_loss = test()
        print('[TEST] test_loss: {:.5f}'.format(avg_loss))

        if avg_loss < min_loss:  # 정확도 갱신 시, 모델 저장
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.model_path, f'{args.model}_best.model'))
            print(f'\t - save model: {args.model}_best.model, (loss: {min_loss} -> {avg_loss})')
            min_loss = avg_loss
        if (epoch + 1) % 50 == 0:
            torch.save({
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, os.path.join(args.model_path, f'{args.model}_e{epoch}.model'))
            print(f'\t - save model: {args.model}_e{epoch}.model')
        print('')
    # 학습 완료 후 모델 저장
    torch.save({
        'net': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, os.path.join(args.model_path, f'{args.model}_e{args.epochs}.model'))
    print(f'\t - save model: {args.model}_e{args.epochs}.model')


if __name__ == '__main__':
    train()