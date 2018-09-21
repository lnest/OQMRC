# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/17
# File Name: data_process
# Edit Author: lnest
# ------------------------------------
import json
import jieba
import logging
import numpy as np
from tqdm import tqdm
import multiprocessing
from collections import Counter
from multiprocessing.pool import ThreadPool
from utility.time_eval import time_count, TimeCountBlock
from utility.use_logger import set_log_level


set_log_level(logging.DEBUG)
logger = logging.getLogger()


def words_cut(str_data):
    segged = jieba.cut(str_data, cut_all=False)
    return list(segged)


def trans_ans_to_tag(ans):
    if not ans:
        return -1
    if '不' in ans:
        return 0
    elif ans == '无法确定':
        return 2
    else:
        return 1


def thread_parse(json_str):
    json_str = json_str.strip()
    json_obj = json.loads(json_str)

    query = json_obj.get('query')
    query = words_cut(query)
    json_obj['query'] = query

    answer = json_obj.get('answer') if 'answer' in json_obj else ''
    json_obj['answer'] = trans_ans_to_tag(answer)

    passage = json_obj.get('passage')
    passage = words_cut(passage)
    json_obj['passage'] = passage
    return json.dumps(json_obj, ensure_ascii=False)
    # return json_obj


def parse(data, thread_num=2):
    thread_pool = ThreadPool(processes=thread_num)
    threads_res = []
    for json_str in data:
        threads_res.append(thread_pool.apply_async(thread_parse, args=(json_str,)))
    thread_pool.close()
    thread_pool.join()

    parse_res = list()
    for _res in threads_res:
        parse_res.append(_res.get())
    return parse_res


def process(corpus_data, process_num=5, thread_num=2):
    pool = multiprocessing.Pool(processes=process_num)
    pool_res = []
    batch_size = (len(corpus_data) // process_num) + 1

    for i in range(process_num):
        start_pos = i * batch_size
        end_pos = min((i + 1) * batch_size, len(corpus_data))
        pool_res.append(pool.apply_async(func=parse, args=(corpus_data[start_pos: end_pos], thread_num,)))
    pool.close()
    pool.join()
    result = list()
    for parse_res in tqdm(pool_res):
        result.extend(parse_res.get())

    return result


@time_count
def cut_corpus(corpus_path, res_path, process_num=5, thread_num=2):
    with TimeCountBlock('read_file'):
        with open(corpus_path, 'r') as cfr:
            corpus_data = cfr.readlines()
    result = process(corpus_data, process_num, thread_num)
    with open(res_path, 'w') as rfw:
        rfw.write('\n'.join(result))

    return len(result)


def words_count(file_name, words_cnt, corpus_len):
    with open(file_name, 'r') as fr:
        for line in tqdm(fr):
            json_obj = json.loads(line)
            passage = json_obj.get('passage')
            question = json_obj.get('query')
            corpus_id = json_obj.get('query_id')
            corpus_len[corpus_id] = {'para_len': len(passage), 'query_len': len(question)}
            for word in passage + question:
                words_cnt[word] += 1

PAD = '--NULL--'
OOV = '--OOV--'


def get_embedding(counter, limit=-1, emb_file=None, vec_size=None):
    """
    从知乎词向量中读取预训练的embedding，或者随机初始化embedding
    :param counter: 
    :param limit: 
    :param emb_file: 
    :param vec_size:  
    :return: 
    """
    print("Generating embedding...")
    word_emb_dict = {}

    # 去除低频词
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        with open(emb_file, "r", encoding="utf-8") as fh:
            first_line = fh.readline()
            size, vec_size = list(map(int, first_line.strip().split()))
            logger.debug('size: %d, vec_size: %d' % (size, vec_size))
            for line in tqdm(fh, total=size):
                word_emd = line.strip().split()
                word = "".join(word_emd[0:-vec_size])
                embeddings = list(map(float, word_emd[-vec_size:]))
                if word in counter and counter[word] > limit:
                    word_emb_dict[word] = embeddings
        print("{} / {} tokens have corresponding embedding vector".format(
            len(word_emb_dict), len(filtered_elements)))
    else:
        # 如果预训练词向量文件为空，随机生成词向量
        assert vec_size is not None
        for token in filtered_elements:
            word_emb_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    token2idx_dict = {token: idx for idx, token in enumerate(word_emb_dict.keys(), 2)}  # {'单词': 单词索引}
    token2idx_dict[PAD] = 0
    token2idx_dict[OOV] = 1
    idx2token = dict([(token2idx_dict[token], token) for token in token2idx_dict])

    word_emb_dict[PAD] = [0. for _ in range(vec_size)]  # PAD 和 OOV词向量为0
    word_emb_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: word_emb_dict[token]
                    for token, idx in token2idx_dict.items()}   # 与token2idx_dict中的idx一致

    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]   # 文章中所有词的矩阵
    word_map = {'idx2word': idx2token, 'word2idx': token2idx_dict}
    return emb_mat, word_map


def save(filename, obj):
    logger.info('write file...: %s' % filename)
    with open(filename, "w") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def sentence2id(sentence, word_map, uniform_size):
    word2id = word_map['word2idx']
    sentence_index = [int(word2id.get(word, '1')) for word in sentence]
    sentence_index = add_pad(sentence_index, uniform_size=uniform_size)
    return sentence_index


def add_pad(ids, uniform_size):
    if len(ids) < uniform_size:
        for i in range(uniform_size - len(ids)):
            ids.append(0)
    else:
        ids = ids[:uniform_size]
    return ids


def corpus2idx(file_name, word_map, passage_len, query_len):
    corpus_idx = list()
    with open(file_name) as fr:
        for line in tqdm(fr):
            json_obj = json.loads(line.strip())
            json_obj['passage'] = sentence2id(json_obj.get('passage'), word_map, passage_len)
            json_obj['query'] = sentence2id(json_obj.get('query'), word_map, query_len)
            corpus_idx.append(json_obj)
    return corpus_idx


# def run_data_flow():
if __name__ == '__main__':
    # cut_corpus(corpus_path='./data/ai_challenger_oqmrc_trainingset_20180816/ai_challenger_oqmrc_trainingset.json', res_path='./data/train.json')
    # cut_corpus(corpus_path='./data/ai_challenger_oqmrc_validationset_20180816/ai_challenger_oqmrc_validationset.json', res_path='./data/dev.json')
    # cut_corpus(corpus_path='./data/ai_challenger_oqmrc_testa_20180816/ai_challenger_oqmrc_testa.json', res_path='./data/test.json')

    # words_cnt = Counter()
    # corpus_len_dist = dict
    # words_count('./data/train.json', words_cnt, corpus_len=corpus_len_dist)
    # words_count('./data/dev.json', words_cnt, corpus_len=corpus_len_dist)
    # words_count('./data/test.json', words_cnt, corpus_len_dist)
    # save('./data/word_df.json', {'df': words_cnt})
    #
    # emb_mat, word_map = get_embedding(words_cnt, limit=1, emb_file='./data/embedding/sgns.zhihu.char')
    # save('./data/emb_mat.json', emb_mat)
    # save('./data/word_map.json', word_map)
    # save('./data/word_df.json', {'df': words_cnt})
    # save('./data/corpus_len.json', corpus_len_dist)

    word_map = json.load(open('./data/word_map.json'))
    dev_data = corpus2idx('./data/dev.json', word_map, 400, query_len=30)
    test_data = corpus2idx('./data/test.json', word_map, 400, query_len=30)
    train_data = corpus2idx('./data/train.json', word_map, 400, query_len=30)

    save('./data/dl_train.json', train_data)
    save('./data/dl_test.json', test_data)
    save('./data/dl_dev.json', dev_data)