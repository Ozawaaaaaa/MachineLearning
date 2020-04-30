import numpy as np
import sys
import time

if len(sys.argv) == 1:
    train_input = open('data/trainwords.txt', 'r')
    index_to_word = open('data/index_to_word.txt', 'r')
    index_to_tag = open('data/index_to_tag.txt', 'r')
    hmmprior = open('output/hmmprior.txt', 'w')
    hmmemit = open('output/hmmemit.txt', 'w')
    hmmtrans = open('output/hmmtrans.txt', 'w')
else:
    train_input = open(sys.argv[1], 'r')
    index_to_word = open(sys.argv[2], 'r')
    index_to_tag = open(sys.argv[3], 'r')
    hmmprior = open(sys.argv[4], 'w')
    hmmemit = open(sys.argv[5], 'w')
    hmmtrans = open(sys.argv[6], 'w')


def get_index(file):
    index_dic = {}
    count = 0;
    for line in file:
        ele = line.strip()
        index_dic[ele] = count
        count += 1
    return index_dic


def get_data(file):
    data = []
    for line in file:
        temp = line.strip().split(" ")
        data.append(temp)
    return data


def update_prior(p, data, tag_index):
    for i in range(len(data)):
        ele = data[i][0].split("_")
        tag = ele[1]
        tag_i = tag_index[tag]
        p[tag_i] += 1
    return p


def update_emit(p, data, word_index, tag_index):
    for line in data:
        for segment in line:
            temp = segment.split("_")
            word = temp[0]
            tag = temp[1]
            word_i = word_index[word]
            tag_i = tag_index[tag]
            p[tag_i][word_i] += 1
    return p


def update_trans(p, data, tag_index):
    for i in range(len(data)):
        for j in range(len(data[i]) - 1):
            temp = data[i][j].split("_")
            temp2 = data[i][j + 1].split("_")
            tag = temp[1]
            tag2 = temp2[1]
            tag_i = tag_index[tag]
            tag_i2 = tag_index[tag2]
            p[tag_i][tag_i2] += 1
    return p


def update_pi(pi, p):
    c = np.sum(p)
    for i in range(len(p)):
        pi[i] = p[i] / c
    return pi


def update(B, p):
    for i in range(len(p)):
        c = np.sum(p[i])
        for j in range(len(p[i])):
            B[i][j] = p[i][j] / c
    return B


def wirte_file(data, file, flag):
    if flag == 0:
        s = ""
        for i in range(len(data)):
            t = "{:.18e}".format(float(data[i]))
            s += str(t)
            if i != len(data) - 1:
                s += "\n"
        file.write(s)
    else:
        s = ""
        for i in range(len(data)):
            for j in range(len(data[i])):
                t = "{:.18e}".format(float(data[i][j]))
                s += str(t)
                if j != len(data[i]) - 1:
                    s += " "
                else:
                    s += "\n"
        file.write(s)
    file.close()


if __name__ == '__main__':
    start_time = time.time()
    # create 2 dictionary to run the
    word_index = get_index(index_to_word)
    tag_index = get_index(index_to_tag)

    # read in training data
    train_data = get_data(train_input)

    # initial probabilities and passout
    p_hmmprior = np.zeros(len(tag_index))
    pi = np.zeros(len(tag_index))
    p_hmmemit = np.ones((len(tag_index), len(word_index)))
    B = np.ones((len(tag_index), len(word_index)))
    p_hmmtrans = np.ones((len(tag_index), len(tag_index)))
    A = np.ones((len(tag_index), len(tag_index)))

    # update prior
    p_hmmprior = update_prior(p_hmmprior+1, train_data, tag_index)
    # update emit
    p_hmmemit = update_emit(p_hmmemit, train_data, word_index, tag_index)
    # update trans
    p_hmmtrans = update_trans(p_hmmtrans, train_data, tag_index)

    # update probabilities
    pi = update_pi(pi, p_hmmprior)
    B = update(B, p_hmmemit)
    A = update(A, p_hmmtrans)

    # write into output file
    wirte_file(pi, hmmprior, 0)
    wirte_file(B, hmmemit, 1)
    wirte_file(A, hmmtrans, 1)

    # timer
    end_time = time.time()
    print()
    print("running time:", end_time - start_time)
