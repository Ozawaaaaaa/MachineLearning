import sys

# # # input
train_input = "smalldata/train_data.tsv"
valid_input = 'smalldata/valid_data.tsv'
test_input = 'smalldata/test_data.tsv'
dict_input = 'dict.txt'
# output
train_output = 'model_train.tsv'
valid_output = 'model_valid.tsv'
test_output = 'model_test.tsv'
feature_flag = 1

# # input
# train_input = sys.argv[1]
# valid_input = sys.argv[2]
# test_input = sys.argv[3]
# dict_input = sys.argv[4]
# # output
# train_output = sys.argv[5]
# valid_output = sys.argv[6]
# test_output = sys.argv[7]
# feature_flag = sys.argv[8]

trimming_threshold = 4


def getDict(fileName):
    dict = {}
    file = open(fileName)
    for l in file:
        temp = l.strip().split(" ")
        dict[temp[0]] = temp[1]
    return dict


def model(file_in_path, file_out_path, dict, flag):
    file_in = open(file_in_path)
    file_out = open(file_out_path, 'w')
    for line in file_in:
        temp = line.split("\t")
        temp2 = temp[1].split(" ")
        d = {}
        for word in temp2:
            try:
                d.setdefault(dict[word], 0)
                d[dict[word]] += 1
            except:
                continue
        out = temp[0]
        for key, value in d.items():
            if flag == 1:
                if value < trimming_threshold:
                    out += ('\t' + key + ':' + '1')
            elif flag == 2:
                out += ('\t' + key + ':' + '1')
        out += "\n"
        file_out.write(out)


if __name__ == '__main__':
    dict = getDict(dict_input)  # dictionary
    model(train_input, train_output, dict, int(feature_flag))
    model(valid_input, valid_output, dict, int(feature_flag))
    model(test_input, test_output, dict, int(feature_flag))
