import random
import copy
import sys

import numpy as np


def ReadFile(fileName):
    data = []
    readIn = open(fileName)
    for i in readIn:
        data.append(i.strip('\r\n').split('\t'))
    # print(len(data))
    return data


#  input: voteCount[][]
#         gi --> number
def giniGain(voteCount, gi):
    voteNumber0 = voteCount[0][0] + voteCount[1][0]
    voteNumber1 = voteCount[0][1] + voteCount[1][1]

    # voteNumber0 = voteCount[0][0] + voteCount[0][1]
    # voteNumber1 = voteCount[1][0] + voteCount[1][1]

    voteTotal = voteNumber0 + voteNumber1
    prob = (voteNumber0 / voteTotal) * (voteNumber1 / voteTotal) * 2
    gg = prob - gi
    return gg


#  input: voteCount[][]
def giniInpurity(condition, vote, voteCount):
    voteNumber0 = voteCount[0][0] + voteCount[0][1]
    voteNumber1 = voteCount[1][0] + voteCount[1][1]

    # voteNumber0 = voteCount[0][0] + voteCount[1][0]
    # voteNumber1 = voteCount[0][1] + voteCount[1][1]

    voteTotal = voteNumber0 + voteNumber1
    # probability calcualtion
    ratio0 = voteNumber0 / voteTotal
    ratio1 = voteNumber1 / voteTotal
    if voteNumber0 == 0:
        voteNumber0 = 1
    if voteNumber1 == 0:
        voteNumber1 = 1
    prob00 = voteCount[0][0] / voteNumber0
    prob01 = voteCount[0][1] / voteNumber0
    prob10 = voteCount[1][0] / voteNumber1
    prob11 = voteCount[1][1] / voteNumber1
    prob0 = prob00 * prob01 * 2 * ratio0
    prob1 = prob10 * prob11 * 2 * ratio1
    gi = prob0 + prob1
    return gi


def majorityVote(condition, vote, voteArray):

    result = []
    voteCount = []

    # print("varray______",len(voteArray))
    for i in range(len(voteArray)):
        voteCount.append([])
        count0 = 0
        count1 = 0
        for j in range(len(voteArray[i])):
            if voteArray[i][j] == vote[0]:
                count0 += 1
            elif voteArray[i][j] == vote[1]:
                count1 += 1
        if count0 > count1:
            result.append(vote[0])
        elif count0 < count1:
            result.append(vote[1])
        elif count0 == count1:
            temp_vote = copy.copy(vote)
            temp_vote.sort()
            result.append(temp_vote[-1])
        voteCount[i].append(count0)
        voteCount[i].append(count1)
    return result, voteCount

# ----------------------------------------------------------------------------------

class Node:
    def __init__(self, input_data, depth, parent, signed_label, signed_attri_name, max_depth):
        self.left = None            # left node
        self.right = None           # right node
        self.parent = parent        # parent node

        self.depth = depth          # current depth
        self.data = input_data      # data working on
        self.label = signed_label           # y/n
        self.leftLabel = None       # y/n
        self.rightLabel = None      # y/n
        self.attri_name = signed_attri_name        # conlum name
        self.self_atti_name = ''
        self.attri_index = -1       # conlum index
        self.answer = ''            # TODO: FINAL ANSWER WHEN REACH THE END OF THE TREE

        self.new_data_left = []     # left node new data
        self.new_data_right = []    # right node new data
        self.vote_names = []
        self.vote_numbers = []
        self.vote_array = []  # vote number and name   ex:[[83, 'democrat'], [67, 'republican']]

        self.rightFlag = True
        self.leftFlag = True

        vote0, vote1, count0, count1 = nodeMajo(self)

        if count0 == 0 or count1 ==0:
            # print("FLAG")
            self.rightFlag = False

        if vote1 == None:
            # print(self.depth, vote1)
            if vote0 == self.parent.vote_names[0]:
                vote1 = self.parent.vote_names[1]
                # print(vote0, vote1)
            elif vote0 == self.parent.vote_names[1]:
                vote1 = self.parent.vote_names[0]
                # print("no")

        self.vote_names.append(vote0)
        self.vote_names.append(vote1)
        # print("vote name:  ", self.vote_names)
        self.vote_numbers.append(count0)
        self.vote_numbers.append(count1)
        # print("vote number:", self.vote_numbers)
        self.vote_array.append([count0, vote0])
        self.vote_array.append([count1, vote1])

        if count0 > count1:
            self.answer = vote0
        elif count0 < count1:
            self.answer = vote1
        elif count0 == count1:
            tmp = copy.copy(self.vote_names)
            tmp.sort()
            self.answer = tmp[-1]




        # print("vote array: ", self.vote_array)

        condition = []  # array. exp:        ['y', 'n']
        vote = []       # array. exp:        ['democrat', 'republican']
        majority = []   # array. exp:        ['republican', 'democrat']
        voteCount = []  # double array. exp: [[28, 64], [55, 2]]
        self.condition = []
        # Find the right stump to split on
        condition, vote, majority, voteCount, attri_index, self.gg = findStp(self)
        # print(condition, majority, voteCount, self.attri_index)
        self.attri_index = attri_index


        self.attri_name = self.data[0][self.attri_index]


        if len(condition) ==0:
            self.rightFlag = False
            self.leftFlag = False
        elif len(condition) ==1:
            self.rightFlag = False

        # if self.parent is not None:
        #     self.condition = parent.condition
        # else:
        #     self.condition = condition

        # print(condition)

        if len(condition) == 0:
            print(self.attri_name)
        # if vote[0] == vote1:
        #     print("att")
        #     t_c0 = copy.copy(condition[0])
        #     t_c1 = copy.copy(condition[1])
        #     condition.clear()
        #     condition.append(t_c1)
        #     condition.append(t_c0)

        # Make the left and right Lable
        if len(condition) == 1:
            self.leftLabel = condition[0]
        elif len(condition) ==2:
            self.leftLabel = condition[0]
            self.rightLabel = condition[1]
        # else:
        #     self.leftLabel = self.condition[1]
        #     self.rightLabel = self.condition[0]
            # self.leftLabel, self.rightLabel = condition[0], condition[1]
        # print("labels:",self.leftLabel, self.rightLable)

        # create new left or right node
        makeNode(self, max_depth)


def splitData(input_data, index, lable):
    output = []

    tmp_header = copy.copy(input_data[0])
    del tmp_header[index]
    output.append(tmp_header)

    for i in input_data:
        tmp = copy.copy(i)
        if tmp[index] == lable:
            del tmp[index]
            output.append(tmp)
    # print(output)
    # print("lenght:", len(output))
    return output


def makeNode(node, max_depth):
    data = node.data
    depth = node.depth

    if depth < max_depth and len(data[0]) >= 2 and node.leftFlag and node.attri_index != -1:
        left_data = splitData(data, node.attri_index, node.leftLabel)
        node.left = Node(left_data, node.depth+1, node, node.leftLabel, node.data[0][node.attri_index], max_depth)
        # if 0 not in node.vote_numbers and None not in node.vote_names and node.rightLable != None:
        if node.rightFlag:
            right_data = splitData(data, node.attri_index, node.rightLabel)
            node.right = Node(right_data, node.depth+1, node, node.rightLabel, node.data[0][node.attri_index], max_depth)


def nodeMajo(node):
    data = node.data
    index_vote_col = len(data[0]) - 1
    vote = []
    # print("data lenght:", len(data))
    for i in range(1, len(data)):
        if data[i][index_vote_col] not in vote:
            vote.append(data[i][index_vote_col])
    if len(vote) ==1:
        vote0 = vote[0]
        vote1 = None
    elif len(vote) == 2:
        vote0 = vote[0]
        vote1 = vote[1]

    count0 = 0
    count1 = 0

    for i in range(1, len(data)):
        if data[i][index_vote_col] == vote0:
            count0 += 1
        elif data[i][index_vote_col] == vote1 and vote1 is not None:
            count1 += 1

    return vote0, vote1, count0, count1


def findStp(node):
    data = node.data

    condition = []
    vote = []
    majority = []
    voteCount = []
    ginigain = 0
    index = -1
    for i in range(len(data[0]) - 1):
        stp_condition, stp_vote, stp_gg, stp_majority, stp_voteCount = dtStp(data, i)
        # make sure all the Gini Gain is larger than 0
        if stp_gg >= ginigain:
            condition = stp_condition
            vote = stp_vote
            majority = stp_majority
            voteCount = stp_voteCount
            ginigain = stp_gg
            index = i
        # elif stp_gg == 0:
        #     index = i

    if index == -1:
        print(data)

    return condition, vote, majority, voteCount, index, ginigain


def printInorder(root, max_depth):
    if root:

        if root.depth == 0:
            # print("[%d %s /%d %s]" % (dict_label[key_list[0]], key_list[0], dict_label[key_list[1]], key_list[1]))
            print("[%d %s/%d %s]" % (root.vote_array[0][0], root.vote_array[0][1], root.vote_array[1][0], root.vote_array[1][1]))
        elif 0 < root.depth <= max_depth:
            print("| " * root.depth, root.parent.attri_name, ' = ', end=root.label)
            print(" [%d %s/%d %s]" % (root.vote_array[0][0], root.vote_array[0][1], root.vote_array[1][0], root.vote_array[1][1]), root.answer)
        printInorder(root.left,max_depth)
        printInorder(root.right,max_depth)


# ---------------------------------------------------------------------------------------------------------

def dtStp(data, index):
    header = data[0]
    numOfAttributes = len(data[0]) - 1
    condition = []  # all possible index condition
    vote = []  # all possible vote
    voteArray = []
    voteArray.append([])
    voteArray.append([])

    for i in range(1, len(data)):
        temp1 = data[i][index]
        temp2 = data[i][numOfAttributes]

        if temp1 not in condition:
            condition.append(temp1)
            # voteArray.append([])
        if temp2 not in vote:
            vote.append(temp2)

        if len(condition) == 1:
            voteArray[0].append(temp2)
        else:
            for j in range(2):
                if condition[j] == temp1:
                    voteArray[j].append(temp2)

    majority, voteCount = majorityVote(condition, vote, voteArray)
    gi = giniInpurity(condition, vote, voteCount)
    gg = giniGain(voteCount, gi)

    return condition, vote, gg, majority, voteCount

ccc = []
shitcount = 0
def nodeResult(node, header, data):
    if node:
        tmp_index = -1
        for i in range(0, len(header)-1):
            x1 = node.attri_name
            x2 = header[i]
            if node.attri_name == header[i]:
                tmp_index = i

        # if tmp_index ==-1:
        #     print(tmp_index)
        data_label = copy.copy(data[tmp_index])
        ll = copy.copy(node.leftLabel)
        rl = copy.copy(node.rightLabel)
        str(data_label)
        str(ll)
        str(rl)
        if data_label == node.leftLabel and node.left is not None:
            nodeResult(node.left, header, data)
        elif data_label == node.rightLabel and node.right is not None:
            nodeResult(node.right, header, data)
        # elif data_lable == node.leftLabel and node.left is None:
        #     c = copy.copy(node.answer)
        #     ccc.append(c)
        # elif data_lable == node.rightLabel and node.right is None:
        #     # print(node.answer)
        #     cc = copy.copy(node.answer)
        #     ccc.append(cc)
        # elif node.rightLabel == None and node.leftLabel == None:
        #     # print(data_label, node.rightLabel, node.leftLabel)
        #     xx = copy.copy(node.answer)
        #     ccc.append(xx)
        else:
            a = node.attri_name
            al = node.leftLabel
            ar = node.rightLabel
            x = copy.copy(node.answer)
            ccc.append(x)

def write(input_data, output_direction):
    a =1

def test(node, data):
    header = data[0]
    data_length = len(data)-1

    ans = []
    score = 0

    for i in range(1,len(data)):
        tmp = data[i]
        nodeResult(node, header, tmp)
        ans.append(data[i][-1])
    # print(len(ans))
    # print(ccc)

    for i in range(0,len(ans)):
        if ans[i] != ccc[i]:
            score+=1


    # print(score/data_length)
    return score/data_length


def writefile(fileName):
    f = open(fileName, 'w')
    return f

# for local test
if __name__ == '__main__':
    # train_input_path = 'small_train.tsv'
    # test_input_path  = 'small_test.tsv'
    # train_input_path = 'education_train.tsv'
    # test_input_path  = 'education_test.tsv'
    #
    # train_input_path = 'politicians_train.tsv'
    # test_input_path = 'politicians_test.tsv'
    # max_depth = 3
    # train_output_path = 'politicians_3_train.labels'
    # test_output_path = 'politicians_3_test.labels'
    # metrics_output_path = 'politicians_3_metrics.txt'

    train_input_path = sys.argv[1]
    test_input_path = sys.argv[2]
    index = int(sys.argv[3])
    max_depth = int(sys.argv[3])
    train_output_path = sys.argv[4]
    test_output_path = sys.argv[5]
    metrics_output_path = sys.argv[6]



    train_data = ReadFile(train_input_path)
    test_data = ReadFile(test_input_path)

    #train
    n = Node(train_data, 0, None, None, "", max_depth)
    print("=================RESULT=====================")
    printInorder(n, max_depth)

    err_train = test(n, train_data)
    train_output_file = writefile(train_output_path)
    for i in range(len(ccc)):
        if i == len(ccc)-1:
            train_output_file.write(ccc[i])
        else:
            train_output_file.write(ccc[i]+"\n")
    ccc.clear()

    err_test = test(n, test_data)
    test_output_file = writefile(test_output_path)
    for i in range(len(ccc)):
        if i == len(ccc) - 1:
            test_output_file.write(ccc[i])
        else:
            test_output_file.write(ccc[i] + "\n")

    metrics_output_file = writefile(metrics_output_path)
    str1 = "error(train): " + str(err_train) + "\n"
    str2 = "error(test): " + str(err_test)
    metrics_output_file.write(str1)
    metrics_output_file.write(str2)





