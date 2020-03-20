# -*- coding:utf-8 -*-

import os
import csv
import random
import json

import sentencepiece as spm


def st_preprocessing(st):
    st = ' '.join(st.split())  # 중복된 공백, 개행, 탭 제거
    return st


# SentencePiece를 사용하기 위해 text 파일로 변환
def csv_to_text(input, output):
    writer = open(output, 'w', encoding='utf-8')
    with open(input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            title = st_preprocessing(line[0])
            content = st_preprocessing(line[1])
            #label = int(line[2])
            writer.write(title + ' &&& ' + content + '\n')  # &&&: 제목과 내용 분리 토큰
    writer.close()
    print('complete csv_to_text (file: {})'.format(output))


# SentencePiece tokenizer 학습
def make_tokenizer(input, vocab_size):
    prefix = 'vocab_' + str(vocab_size)
    user_defined_symbols = '[PAD],[UNK],[CLS],[SEP],[MASK]'
    args = """--input={} \
              --model_prefix={} \
              --vocab_size={} \
              --model_type=bpe \
              --user_defined_symbols={}""".format(input, prefix, vocab_size, user_defined_symbols)
    spm.SentencePieceTrainer.train(args)
    tokenize_sample(prefix=prefix)
    print('complete make_tokenizer (prefix: {}, vocab: {})'.format(prefix, vocab_size))
    return prefix


def tokenize_sample(prefix='vocab_10000'):
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load('{}.model'.format(prefix))
    tokens = tokenizer.EncodeAsPieces('웹 게시글 페이징 하려고 알아본건데. 알려주실 수 있을까요? ``` <게시글 검색 쿼리> String sql = "SELECT * from ( "; sql +=" SELECT ')
    print(tokens)


def tokenize_sample_test():
    from konlpy.tag import Kkma, Okt, Hannanum, Komoran
    seqs = [
        '웹 게시글 페이징 하려고 알아본건데. 알려주실 수 있을까요? ``` <게시글 검색 쿼리> String sql = "SELECT * from ( "; sql +=" SELECT ',
        ' '.join('''''
                            <script type=""text/$$$"">
                        (function () {
                            window.item_list = {{item_list|safe}};
                        })();
                        $(""#search"").autocomplete({
                            source: window.item_list
                        });
                    </script>
        '''.split()),
        '[1]: https://res.cloudinary.com/eightcruz/image/upload/v1503145822/aniod28b6vzmc0jpebze.png"'
    ]

    for i, seq in enumerate(seqs):
        print(f'# test {i}')
        tokenizer = Okt()
        tokens = tokenizer.morphs(seq)
        print('Okt:', len(tokens), tokens)

        tokenizer = Kkma()
        tokens = tokenizer.morphs(seq)
        print('Kkma:', len(tokens), tokens)

        tokenizer = Hannanum()
        tokens = tokenizer.morphs(seq)
        print('Hannanum:', len(tokens), tokens)

        # tokenizer = Komoran()
        # tokens = tokenizer.morphs(seq)
        # print('Komoran:', len(tokens), tokens)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load('{}.model'.format('vocab_2500'))
        tokens = tokenizer.EncodeAsPieces(seq)
        print('SP_2.5k:', len(tokens), tokens)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load('{}.model'.format('vocab_5000'))
        tokens = tokenizer.EncodeAsPieces(seq)
        print('SP_5k:', len(tokens), tokens)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load('{}.model'.format('vocab_10000'))
        tokens = tokenizer.EncodeAsPieces(seq)
        print('SP_10k:', len(tokens), tokens)

        tokenizer = spm.SentencePieceProcessor()
        tokenizer.Load('{}.model'.format('vocab_20000'))
        tokens = tokenizer.EncodeAsPieces(seq)
        print('SP_20k:', len(tokens), tokens)
        print('')


def make_vocabulary(input, vocab_size):
    from konlpy.tag import Okt
    tokenizer = Okt()

    except_words = ['http', '![']
    replace_words = ['javascript', 'java', 'android', 'org', 'python']
    word_dict = {}
    with open(input, 'r', encoding='utf-8') as f:
        data = f.readlines()
    for d in data:
        words = d.split()
        f_words = []
        for w in words:
            w = w.lower()
            for ew in except_words:
                if ew in w:
                    break
            else:
                for rw in replace_words:
                    if rw in w:
                        f_words.append(rw)
                        break
                else:
                    f_words.append(w)
        d = ' '.join(f_words)
        tokens = tokenizer.morphs(d)
        for t in tokens:
            if t in word_dict:
                word_dict[t] += 1
            else:
                word_dict[t] = 1
    kv = sorted(word_dict.items(), key=lambda kv: kv[1], reverse=True)
    word_dict = {}
    for k, v in kv:
        word_dict[str(k)] = v
    with open('word_book.json', 'w', encoding='utf-8') as f:
        json.dump(word_dict, f, indent='\t', ensure_ascii=False)
    with open('vocabulary.txt', 'w', encoding='utf-8') as f:
        special = ['[CLS]', '[SEP]', '[PAD]', '[MASK]', '[UNK]']
        f.writelines('\n'.join(special))
        f.writelines('\n')
        f.writelines('\n'.join(list(word_dict.keys())[:vocab_size-len(special)]))

    print('complete make_vocabulary (file: {})'.format('vocabulary.txt'))


def make_bertlm_dataset(input, suffix, tokenizer, max_num_tokens=128, val_rate=0.2):
    data = {'doc_id':[], 'content':[], 'label':[]}
    with open(input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            title = st_preprocessing(line[0])
            content = st_preprocessing(line[1])
            paragraph = title + ' &&& ' + content
            tokens = tokenizer.tokenize(paragraph)  # tokenizing

            # 최대 토큰 개수에 따라 작은 문장들로 분리
            for j in range(0, len(tokens) // (max_num_tokens - 2) + 1):
                t_tokens = ['[CLS]'] + tokens[j * (max_num_tokens - 2):(j + 1) * (max_num_tokens - 2)] + ['[SEP]']
                t_tokens += ['[PAD]'] * (max_num_tokens - len(t_tokens))  # PAD 토큰 추가
                sub_st = ' '.join(t_tokens)  # str 형태로 변환

                data['doc_id'].append(i-1)
                data['content'].append(sub_st)

    index_dict = {}
    indices = list(range(len(data['doc_id'])))
    targets = ['train', 'val']
    random.shuffle(indices)
    index_dict['val'] = indices[:int(len(indices) * val_rate)]
    index_dict['train'] = indices[int(len(indices) * val_rate):]

    for target in targets:
        sub_data = []
        for i in index_dict[target]:
            sub_data.append(
                {
                    'doc_id': data['doc_id'][i],
                    'content': data['content'][i],
                }
            )
        with open(f'bertlm_{target}_{suffix}.json', 'w', encoding='utf-8') as f:
            json.dump(sub_data, f, indent='\t')

    print('complete make_bertlm_dataset (max_num_tokens: {})'.format(max_num_tokens))


def make_bertcls_dataset(input, suffix, tokenizer, max_num_tokens=128, val_rate=0.2, inference=False):
    data = {'doc_id':[], 'content':[], 'label':[]}
    with open(input, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for i, line in enumerate(reader):
            if i == 0:
                continue
            title = st_preprocessing(line[0])
            content = st_preprocessing(line[1])
            label = int(line[2])-1 if not inference else -1  # label: [0, 4]
            paragraph = title + ' &&& ' + content
            tokens = tokenizer.tokenize(paragraph)  # tokenizing

            data['doc_id'].append(i - 1)
            data['label'].append(label)
            sub_sts = []
            # 최대 토큰 개수에 따라 작은 문장들로 분리
            for j in range(0, len(tokens) // (max_num_tokens - 2) + 1):
                t_tokens = ['[CLS]'] + tokens[j * (max_num_tokens - 2):(j + 1) * (max_num_tokens - 2)] + ['[SEP]']
                t_tokens += ['[PAD]'] * (max_num_tokens - len(t_tokens))  # PAD 토큰 추가
                sub_st = ' '.join(t_tokens)  # str 형태로 변환
                sub_sts.append(sub_st)
            data['content'].append(sub_sts)

    index_dict = {}
    indices = list(range(len(data['doc_id'])))
    if inference:
        targets = ['test']
        index_dict['test'] = indices
    else:
        targets = ['train', 'val']
        random.shuffle(indices)
        index_dict['val'] = indices[:int(len(indices) * val_rate)]
        index_dict['train'] = indices[int(len(indices) * val_rate):]

    for target in targets:
        sub_data = []
        for i in index_dict[target]:
            sub_data.append(
                {
                    'doc_id': data['doc_id'][i],
                    'content': data['content'][i],
                    'label': data['label'][i]
                }
            )
        with open(f'bertcls_{target}_{suffix}.json', 'w', encoding='utf-8') as f:
            json.dump(sub_data, f, indent='\t')

    print('complete make_bertcls_dataset (max_num_tokens: {})'.format(max_num_tokens))


class Tokenizer:
    def __init__(self, tokenizer_name, prefix=None):
        self.tokenizer = None
        self._load_tokenizer(tokenizer_name, prefix)

        if not isinstance(self.tokenizer, spm.SentencePieceProcessor):
            with open('vocabulary.txt', 'r', encoding='utf-8') as f:
                self.vocabulary = f.read().splitlines()

            self.except_words = ['http', '![']
            self.replace_words = ['javascript', 'java', 'android', 'org', 'python']

    def _load_tokenizer(self, tokenizer_name, prefix='vocab_5000'):
        if tokenizer_name == 'sentencepiece':
            make_tokenizer('train.txt', vocab_size=vocab_size)
            self.tokenizer = spm.SentencePieceProcessor()
            self.tokenizer.Load('{}.model'.format(prefix))
        else:
            from konlpy.tag import Okt
            self.tokenizer = Okt()

    def _preprocessing(self, x):
        f_x = []
        for w in x.split():
            w = w.lower()
            for ew in self.except_words:
                if ew in w:
                    break
            else:
                for rw in self.replace_words:
                    if rw in w:
                        f_x.append(rw)
                        break
                else:
                    f_x.append(w)
        return ' '.join(f_x)

    def tokenize(self, x):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            x = self.tokenizer.EncodeAsPieces(x)
        else:
            x = self._preprocessing(x)
            x = self.tokenizer.morphs(x)
            for i, w in enumerate(x):
                if w not in self.vocabulary:
                    x[i] = '[UNK]'
        return x

    def tokens_to_ids(self, tokens):
        if not isinstance(tokens, list):
            tokens = tokens.split()
        ids = list(map(self.token_to_id, tokens))
        return ids

    def token_to_id(self, token):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            id = self.tokenizer.PieceToId(token)
        else:
            if token in self.vocabulary:
                id = self.vocabulary.index(token)
            else:
                id = self.vocabulary.index('[UNK]')
        return id

    def get_tokens(self, vocab_prefix=None, for_masking=True):
        if isinstance(self.tokenizer, spm.SentencePieceProcessor):
            with open('{}.vocab'.format(vocab_prefix), 'r', encoding='utf-8') as f:
                tokens = [doc.strip().split("\t")[0] for doc in f]
        else:
            tokens = self.vocabulary[:]

        if for_masking:
            for special in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']:
                tokens.remove(special)
        return tokens


if __name__ == '__main__':
    vocab_size = 2000
    prefix = f'vocab_{vocab_size}'
    max_num_tokens = 64
    val_rate = [0.2, 0.2]
    suffix = f'v{vocab_size}_t{max_num_tokens}'
    # os.makedirs('./data', exist_ok=True)
    tokenizer_name = 'okt' # 'sentencepiece'

    # csv_to_text('hashcode_classification2020_train.csv', 'train.txt')

    if tokenizer_name == 'sentencepiece':
        make_tokenizer('train.txt', vocab_size=vocab_size)
    else:
        make_vocabulary('train.txt', vocab_size=vocab_size)

    tokenizer = Tokenizer(tokenizer_name)

    make_bertlm_dataset('hashcode_classification2020_train.csv', suffix, tokenizer,
                        max_num_tokens=max_num_tokens, val_rate=val_rate[0])
    make_bertcls_dataset('hashcode_classification2020_train.csv', suffix, tokenizer,
                         max_num_tokens=max_num_tokens, val_rate=val_rate[1], )
    make_bertcls_dataset('hashcode_classification2020_test.csv', suffix, tokenizer,
                         max_num_tokens=max_num_tokens, inference=True)
