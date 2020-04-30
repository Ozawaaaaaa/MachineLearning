import numpy as np
import time
import sys
import matplotlib.pyplot as plt

# inputs

case = 0

if case == 0:
    train_input = 'data/largeTrain.csv'
    test_input = 'data/largeTest.csv'
    train_out = 'trainOut.label'
    test_out = 'testOut.label'
    metrics_out = 'abc.txt'
    num_epoch = int(1)
    hidden_units = int(50)
    init_flag = int(1)  # 1 or 2
    learning_rate = float(0.01)
else:
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])  # 1 or 2
    learning_rate = float(sys.argv[9])


# ------------------------------------------------------------------------------------

def read_file(filename):
    data = np.genfromtxt(filename, delimiter=',').astype(np.int)
    return data


def read_lables(data):
    # deal with y
    y_raw = data[:, 0]
    y = np.zeros((data.shape[0], 10))
    y[np.arange(data.shape[0]), y_raw] = 1
    # deal with x
    # x_raw = data[:, 1:]
    x = np.copy(data)
    x[:, 0] = 1
    x.astype(float)
    return y, x, y_raw


def init_pram(x, y):
    if init_flag == 1:
        alpha_ = np.random.uniform(-0.1, 0.1, (hidden_units, x.shape[1] - 1))
        alpha = np.hstack((np.zeros((hidden_units, 1)), alpha_))
        beta_ = np.random.uniform(-0.1, 0.1, (y.shape[1], hidden_units))
        beta = np.hstack((np.zeros((y.shape[1], 1)), beta_))
    elif init_flag == 2:
        alpha = np.zeros((hidden_units, x.shape[1]))
        beta = np.zeros((y.shape[1], hidden_units + 1))
    return alpha, beta


# ------------------------------------------------------------------------------------
class obj:

    def __init__(self, xi, a, z, b, y_hat, J, s):
        self.xi = xi
        self.a = a
        self.z = z
        self.b = b
        self.y_hat = y_hat
        self.J = J
        self.s = s


# ------------------------------------------------------------------------------------

def LinearForward(x, alpha):
    a = alpha.dot(x)
    return a


def SigmoidForward(a):
    s = 1 / (1 + np.exp(-a))
    return s


def LinearForward(z, beta):
    b = beta.dot(z)
    return b


def SoftmaxForward(b):
    y_hat = np.exp(b) / np.sum(np.exp(b))
    return y_hat


def CrossEntropyForward(y, y_hat):
    J = -np.sum(np.multiply(y, np.log(y_hat)))
    return J


def NNForward(xi, yi, alpha, beta):
    a = LinearForward(xi, alpha)
    s = SigmoidForward(a)
    z = np.vstack((np.array([1]), s))
    b = LinearForward(z, beta)
    # print("b",b)
    y_hat = SoftmaxForward(b)
    # print("yhat",y_hat)
    J = CrossEntropyForward(yi, y_hat)
    # print('j',J)
    o = obj(xi, a, z, b, y_hat, J, s)
    return o


# ------------------------------------------------------------------------------------
def CrossEntropyBackward(y, y_hat, J, gj):
    return None


def SoftmaxBackward(b, y_hat, gy_hat):
    gb = y_hat - gy_hat
    return gb


def LinearBackward(a, alpha, b, g):
    GG = g.dot(a.T)
    gg = g.T.dot(alpha[:, 1:])
    return GG, gg


def SigmoidBackward(a, z, gz):
    ga = gz * (z * (1 - z))
    return ga


def NNBackward(xi, yi, alpha, beta, o):
    gj = 1
    # gy_hat = CrossEntropyBackward(yi, o.y_hat, o.J, gj)
    gb = SoftmaxBackward(o.b, o.y_hat, yi)
    gbeta, gz = LinearBackward(o.z, beta, o.b, gb)
    ga = SigmoidBackward(o.a, o.s, gz.T)
    # print(ga)
    galpha, gx = LinearBackward(xi, alpha, o.b, ga)
    return galpha, gbeta


# ------------------------------------------------------------------------------------

def SGD_unit(x, y, alpha, beta):
    o = NNForward(x, y, alpha, beta)
    g_alpha, g_beta = NNBackward(x, y, alpha, beta, o)
    # update para
    alpha -= learning_rate * g_alpha
    beta -= learning_rate * g_beta
    return alpha, beta


def cross_entropy(x, y, alpha, beta):
    loss = []
    y_predict = []
    for i in range(x.shape[0]):
        xi = x[i].reshape(-1, 1)
        yi = y[i].reshape(-1, 1)
        o = NNForward(xi, yi, alpha, beta)
        loss.append(o.J)
        l = np.argmax(o.y_hat)
        y_predict.append(l)
    ce = np.mean(loss)
    return ce, y_predict


def error(y_predict, y):
    e = 0
    for i in range(len(y_predict)):
        if y_predict[i] != y[i]:
            e += 1
    er = e / len(y_predict)
    return er


def write_out_lable(y_predict, file_path):
    file = open(file_path, 'w')
    for y in y_predict:
        file.write(str(y) + '\n')


def write_out_metrics(ce_train, ce_test, epoch):
    str1 = "epoch=" + str(epoch) + " crossentropy(train): " + str(ce_train) + "\n"
    str2 = "epoch=" + str(epoch) + " crossentropy(test): " + str(ce_test) + "\n"
    print(str1, str2)
    # metrics_out.write(str1)
    # metrics_out.write(str2)
    return str1,str2


# ------------------------------------------------------------------------------------
def plot(train, test):
    epoch_range = range(0,100)
    plt.plot(epoch_range, train)
    plt.plot(epoch_range, test)
    plt.title('Leaning rate 0.001')
    plt.ylabel("Cross Entropy")
    plt.xlabel('Epochs')
    plt.legend(['Train','Test'], loc='upper right')
    plt.show()

def plot2(train, test):
    epoch_range = [5,20,50,100,200]
    plt.plot(epoch_range, train)
    plt.plot(epoch_range, test)
    # plt.title('Leaning rate 0.001')
    plt.ylabel("Cross Entropy")
    plt.xlabel('Epochs')
    plt.legend(['Train','Test'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    start_time = time.time()


    train_data = read_file(train_input)
    test_data = read_file(test_input)

    y_train, x_train, y_raw_train = read_lables(train_data)
    y_test, x_test, y_raw_test = read_lables(test_data)

    alpha, beta = init_pram(x_train, y_train)

    # file_metrics = open(metrics_out, 'w')
    tmp = []
    ce_array_train = []
    ce_array_test = []

    tr = [0.6614156399404323,
          0.4269603979117648,
          0.27456832800016245,
          0.17629721630527093,
          0.10642240248110303]
    te = [0.7395264863023121,
          0.5564284370459656,
          0.466105246938685,
          0.4357564724294334,
          0.4507741772150679]
    # procedure SGD:
    for epoch in range(num_epoch):
        # Train
        for i in range(x_train.shape[0]):
            # print(x_train)
            xi = x_train[i].reshape(-1, 1)
            yi = y_train[i].reshape(-1, 1)
            alpha, beta = SGD_unit(xi, yi, alpha, beta)

        print(alpha)
        ce_train, y_predict_train = cross_entropy(x_train, y_train, alpha, beta)
        ce_test, y_predict_test = cross_entropy(x_test, y_test, alpha, beta)

        #plt
        ce_array_train.append(ce_train)
        ce_array_test.append(ce_test)


        str1, str2 = write_out_metrics(ce_train, ce_test, epoch + 1)
        tmp.append(str1)
        tmp.append(str2)


    print(y_predict_train)
    print(y_predict_test)
    error_train = error(y_predict_train, y_raw_train)
    error_test = error(y_predict_test, y_raw_test)
    print(error_train, error_test)

    write_out_lable(y_predict_train, train_out)
    write_out_lable(y_predict_test, test_out)

    str3 = "error(train): " + str(error_train) + "\n"
    str4 = "error(test): " + str(error_test) + "\n"

    tmp.append(str3)
    tmp.append(str4)
    string = ""
    for sad in tmp:
        string+=sad
    print(string)

    open(metrics_out,'w').write(string)

    plot2(tr, te)
    print("tr",np.mean(ce_array_train),"te",np.mean(ce_array_test))

    end_time = time.time()
    print("Time used: ", (end_time-start_time))
    # SGD_unit(x_train,y_train,alpha,beta)

