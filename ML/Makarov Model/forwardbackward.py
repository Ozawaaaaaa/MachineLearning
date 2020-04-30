import numpy as np
import sys
import time


if len(sys.argv) == 1:
    test_input = open('data/testwords.txt', 'r')
    index_to_word = open('data/index_to_word.txt', 'r')
    index_to_tag = open('data/index_to_tag.txt', 'r')
    hmmprior = open('data/hmmprior.txt', 'r')
    hmmemit = open('data/hmmemit.txt', 'r')
    hmmtrans = open('data/hmmtrans.txt', 'r')
    # test_input = open('data/toy_data/toytest.txt', 'r')
    # index_to_word = open('data/toy_data/toy_index_to_word.txt', 'r')
    # index_to_tag = open('data/toy_data/toy_index_to_tag.txt', 'r')
    # hmmprior = open('data/toy_data/toy_hmmprior.txt', 'r')
    # hmmemit = open('data/toy_data/toy_hmmemit.txt', 'r')
    # hmmtrans = open('data/toy_data/toy_hmmtrans.txt', 'r')
    predicted_file = open('output/predicttest.txt', 'w')
    metric_file = open('output/metrics.txt', 'w')
else:
    test_input = open(sys.argv[1], 'r')
    index_to_word = open(sys.argv[2], 'r')
    index_to_tag = open(sys.argv[3], 'r')
    hmmprior = open(sys.argv[4], 'r')
    hmmemit = open(sys.argv[5], 'r')
    hmmtrans = open(sys.argv[6], 'r')
    predicted_file = open(sys.argv[7], 'w')
    metric_file = open(sys.argv[8], 'w')


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


def read_prob(p, file, a):
    if a == 0:
        i = 0
        for line in file:
            # temp = line.split("")
            line.strip()
            # for j in range(len(temp)):
            #     p[j] = float(temp[j])
            p[i] = float(line)
            i += 1
        return p
    else:
        i = 0
        for line in file:
            temp = line.split(" ")
            for j in range(len(temp)):
                p[i][j] = float(temp[j])
            i += 1
        return p


def interpret(data):
    test_words = []
    test_tags = []

    for i in range(len(data)):
        temp_word = []
        temp_tag = []
        for j in range(len(data[i])):
            segment = data[i][j].split("_")
            word = segment[0]
            tag = segment[1]
            temp_word.append(word)
            temp_tag.append(tag)
        test_words.append(temp_word)
        test_tags.append(temp_tag)

    return test_words, test_tags


def to_index(data, dic):
    res = []
    for i in range(len(data)):
        temp = []
        for ele in data[i]:
            tag = dic[ele]
            temp.append(tag)
        res.append(temp)
    return res


def forwardbackward(line, word_index, tag_index, pi, A, B):
    prediction = []
    log_likelihood = 0

    # forward
    alpha = np.zeros((len(tag_index), len(line)))

    for i in range(len(alpha)):
        alpha[i][0] = pi[i] * B[i][line[0]]
    for i in range(1, len(line)):
        for j in range(len(tag_index)):
            sum = 0.0
            for k in range(len(tag_index)):
                sum += alpha[k][i - 1] * A[k][j]
            alpha[j][i] = B[j][line[i]] * sum

    log_likelihood = np.log(np.sum(alpha[:, -1]))


    # backward
    beta = np.zeros((len(tag_index), len(line)))

    beta[:, -1] = 1

    for i in range((len(line) - 2), -1, -1):
        for j in range(len(tag_index)):
            sum = 0.0
            for k in range(len(tag_index)):
                sum += B[k][line[i + 1]] * beta[k][i + 1] * A[j][k]
            beta[j][i] = sum
    # print(beta)

    for i in range(len(line)):
        prediction.append(np.argmax((alpha * beta)[:, i]))

    print(beta)
    return prediction, log_likelihood


def write_metrics(acc, l):
    s = "Average Log-Likelihood: " + str(l) + "\n" + "Accuracy: " + str(acc)
    metric_file.write(s)
    metric_file.close()


def write_pred(tag_index, words, tags):
    s = ""

    inv_index = {v: k for k, v in tag_index.items()}
    for i in range(len(words)):
        for j in range(len(words[i])):
            s += words[i][j]
            s += "_"
            s += inv_index[tags[i][j]]
            if j == len(words[i]) - 1:
                s += '\n'
            else:
                s += ' '

    predicted_file.write(s)
    predicted_file.close()


if __name__ == '__main__':
    start_time = time.time()
    print("--- Start ---")
    # create 2 dictionary to run the
    word_index = get_index(index_to_word)
    tag_index = get_index(index_to_tag)
    print("--- Index in ---")
    # read in testning data
    test_data = get_data(test_input)

    # interpret test data
    test_words, test_tags = interpret(test_data)
    # print(test_words)
    test_words_index = to_index(test_words, word_index)
    test_tags_index = to_index(test_tags, tag_index)
    print("--- Data in ---")
    # read in the probability
    pi = np.zeros(len(tag_index))
    B = np.ones((len(tag_index), len(word_index)))
    A = np.ones((len(tag_index), len(tag_index)))

    pi = read_prob(pi, hmmprior, 0)
    B = read_prob(B, hmmemit, 1)
    A = read_prob(A, hmmtrans, 1)
    # print('pi shape:', pi)
    # print('B  shape:', B)
    # print('A  shape:', A)
    # forward backward
    predictions = []
    log_liklihoods = []
    print("--- FB start ---")
    c0 = 0
    for line in test_words_index:
        p, l = forwardbackward(line, word_index, tag_index, pi, A, B)
        predictions.append(p)
        log_liklihoods.append(l)
        c0 += 1
        if c0 % 100 == 0:
            print("Finished", int(c0 / 100))


    # ll_mean = np.array(log_liklihoods).mean()
    var1 = np.sum(log_liklihoods)
    var2 = len(log_liklihoods)
    ll_mean = var1 / var2

    e = 0
    s = 0
    for i in range(len(test_tags_index)):
        for j in range(len(test_tags_index[i])):
            if test_tags_index[i][j] == predictions[i][j]:
                e += 1
            s += 1

    print("----------------------------------")
    acc = e / s
    print('Average Log-Likelihood:', ll_mean)
    print('Accuracy:', acc)
    print("----------------------------------")

    write_metrics(acc, ll_mean)
    write_pred(tag_index, test_words, predictions)

    ##
    end_time = time.time()
    print()
    print("running time:", end_time - start_time)
