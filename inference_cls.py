# -*- coding=utf-8 -*-

import shutil
import os
import argparse
import time
from collections import OrderedDict
import csv

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
parser.add_argument('--gpu', type=int, default=0)  # 사용할 gpu 선택 (data parallel 사용 시 무시됨)
parser.add_argument('--seq_ensemble', default=True, action='store_true')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_num_seq', type=int, default=50)
parser.add_argument('--d_hidden', type=int, default=256)
parser.add_argument('--n_heads', type=int, default=2)
parser.add_argument('--n_groups', type=int, default=2)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--vocab_size', type=int, default=5000)
parser.add_argument('--n_class', type=int, default=5)
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
def build_model(args):
    base_model = Bert(args.d_hidden, args.n_heads, args.n_layers, args.n_groups,
                      args.vocab_size, args.max_num_tokens).to(device)
    model = BertClassifier(base_model, args.d_hidden, args.n_class).to(device)

    # pretrained 모델 로드
    state_dict = torch.load(args.bertcls_weights)
    if 'net' in state_dict:
        state_dict = state_dict['net']
    model.load_state_dict(state_dict)

    return model

# data loader 설정
def load_dataset(args):
    suffix = f'v{args.vocab_size}_t{args.max_num_tokens}'
    tokenizer = Tokenizer(tokenizer_name=args.tokenizer_name, prefix='vocab_{}'.format(args.vocab_size))

    dataset = BertClsDataset(f'bertcls_test_{suffix}.json', tokenizer=tokenizer,
                             inference=True)
    print('dataset#:', len(dataset))

    def collate(batch):
        if args.seq_ensemble:  # batch_size 1 가정
            docs = []
            item = batch[0]
            if len(item[0]) > args.max_num_seq:
                for i in range(0, len(item[0]) - args.max_num_seq, 5):
                    docs.append(item[0][i:i + args.max_num_seq])
            else:
                docs.append(item[0][:args.max_num_seq])
        else:
            docs = [item[0][:args.max_num_seq] for item in batch]
        labels = [item[1] for item in batch]
        return [docs, labels]

    test_data_loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate)

    return test_data_loader


def inference(output_file, ensemble=[64]):
    fw = open(output_file, 'w', encoding='utf-8', newline='')
    writer = csv.writer(fw)
    writer.writerow(['label'])
    outputs = [[] for _ in range(len(ensemble))]
    for i, num_token in enumerate(ensemble):
        args.max_num_tokens = num_token
        args.bertcls_weights = f'cls_weights/bertcls_best_t{num_token}.model'

        model = build_model(args)
        model.eval()

        test_data_loader = load_dataset(args)
        with torch.no_grad():
            for (docs, _) in test_data_loader:  # iteration
                docs = [doc.to(device) for doc in docs]

                out = model(docs)
                preds = out.cpu().numpy()
                if args.seq_ensemble and len(preds) > 1:
                    outputs[i].append(preds.mean(0))
                else:
                    for pred in preds:
                        outputs[i].append(pred)

    outputs = np.array(outputs).transpose(1, 0, 2)
    for all_out in outputs:
        writer.writerow([all_out.mean(0).argmax() + 1])
    fw.close()


if __name__ == '__main__':
    inference(output_file='output.csv', ensemble=[16, 32, 64, 128])
    print('complete inference')
