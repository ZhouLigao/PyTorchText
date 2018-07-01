#coding:utf8

'''
将embedding.txt 转成numpy矩阵
'''


import word2vec
import numpy as np
from numpy import linalg as LA


def main_old(em_file, em_result):
    '''
    embedding ->numpy
    '''

    em = word2vec.load(em_file)
    vec = (em.vectors)
    word2id = em.vocab_hash
    # d = dict(vector = vec, word2id = word2id)
    # t.save(d,em_result)
    np.savez_compressed(em_result,vector=vec,word2id=word2id)


def unitvec(vec):
    return (1.0 / LA.norm(vec, ord=2)) * vec


def main(em_file, em_result,encoding='utf-8',desired_vocab=None,):
    with open(em_file, 'rb') as fin:
        header = fin.readlines()
        # vocab_size, vector_size = list(map(int, header.split()))

        vocab_size, vector_size = len(header),len(header[0])-1

        vocab = np.empty(vocab_size, dtype='<U%s' % 78)
        vectors = np.empty((vocab_size, vector_size), dtype=np.float)
        for i, line in enumerate(fin):
            line = line.decode(encoding).strip()
            parts = line.split(' ')
            word = parts[0]
            include = desired_vocab is None or word in desired_vocab
            if include:
                vector = np.array(parts[1:], dtype=np.float)
                vocab[i] = word
                vectors[i] = unitvec(vector)# todo unitvec
                # vectors[i] = vector

        if desired_vocab is not None:
            vectors = vectors[vocab != '', :]
            vocab = vocab[vocab != '']

        vocab_hash = {}
        for i, word in enumerate(vocab):
            vocab_hash[word] = i

    np.savez_compressed(em_result,vector=vectors,word2id=vocab_hash)


if __name__ == '__main__':
    # import fire
    # fire.Fire()
    # main('../../origindata/char_embed.txt', 'char_embed')
    main('../../origindata/word_embed.txt', 'word_embed')


# python scripts/data_process/embedding2matrix.py main origindata/char_embed.txt origindata/char_embed.npz
# python scripts/data_process/embedding2matrix.py main origindata/word_embed.txt origindata/word_embed.npz
