from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
from transition.Instance import Instance

def read_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile, vocab):
            data.append(sentence)
    return data

def read_train_corpus(file_path, vocab=None):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readDepTree(infile, vocab):
            if DepTree(sentence).isProj():
                data.append(sentence)
    return data

def sentences_numberize(sentences, vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab)

def sentence2id(sentence, vocab):
    result = []
    for idx in range(sentence.size):
        wordid = vocab.word2id(sentence.words[idx])
        extwordid = vocab.extword2id(sentence.words[idx])
        tagid = vocab.tag2id(sentence.tags[idx])
        head = sentence.heads[idx]
        relid = vocab.rel2id(sentence.rels[idx])
        result.append([wordid, extwordid, tagid, head, relid])
    return result



def batch_slice(data, batch_size):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        yield sentences


def data_iter(data, batch_size, shuffle=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab):
    batch_size = len(batch)
    batch_insts = []
    for fields in batch:
        inst = Instance(fields[0], vocab)
        batch_insts.append(inst)

    length = batch_insts[0].size
    for b in range(1, batch_size):
        if batch_insts[b].size > length: length = batch_insts[b].size

    words = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    tags = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)
    masks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    heads = []
    rels = []
    lengths = []

    b = 0
    for sentence in sentences_numberize(batch_insts, vocab):
        index = 0
        length = len(sentence)
        lengths.append(length)
        head = np.zeros((length), dtype=np.int32)
        rel = np.zeros((length), dtype=np.int32)
        for dep in sentence:
            words[b, index] = dep[0]
            extwords[b, index] = dep[1]
            tags[b, index] = dep[2]
            head[index] = dep[3]
            rel[index] = dep[4]
            masks[b, index] = 1
            index += 1
        b += 1
        heads.append(head)
        rels.append(rel)



    batch_sent = []
    for inst in batch:
        batch_sent.append(inst[0])
    return words, extwords, tags, heads, rels, lengths, masks, batch_sent

def batch_actions_variable(batch, vocab):
    batch_actions = []
    batch_feats = []
    batch_candid = []
    for inst in batch:
        batch_actions.append(inst[1])
        batch_feats.append(inst[2])
        batch_candid.append(inst[3])
    acs = []
    max_step = -1
    for actions in batch_actions:
        ac = np.zeros(len(actions), dtype=np.int32)
        tmp_step = len(ac)
        if max_step < tmp_step:
            max_step = tmp_step
        for (idx, action) in enumerate(actions):
            ac[idx] = vocab.ac2id(action)
        acs.append(ac)

    batch_step_actions = []
    for idx in range(0, max_step):
        step_actions = []
        for actions in batch_actions:
            if idx < len(actions):
                step_actions.append(actions[idx])
        batch_step_actions.append(step_actions)

    return batch_actions, acs, \
           batch_step_actions, batch_feats, batch_candid


