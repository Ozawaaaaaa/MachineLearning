from environment import MountainCar
import sys
import numpy as np
import time

# input
if len(sys.argv) == 1:
    mode = 'tile'
    weight_out = open('weight.out', 'w')
    returns_out = open('returns.out', 'w')
    episodes = 25
    max_iterations = 200
    epsilon = 0.0
    gamma = 0.99
    learning_rate = 0.005
else:
    mode = sys.argv[1]
    weight_out = open(sys.argv[2], 'w')
    returns_out = open(sys.argv[3], 'w')
    episodes = int(sys.argv[4])
    max_iterations = int(sys.argv[5])
    epsilon = float(sys.argv[6])
    gamma = float(sys.argv[7])
    learning_rate = float(sys.argv[8])


# ------------------------------------------------------------------------------------

class Agent:
    def __init__(self):
        self.mc = MountainCar(mode)
        self.w = np.zeros((self.mc.state_space, self.mc.action_space))  # 2x3 or 2048x3
        self.b = 0
        self.a = None
        self.done = False
        self.r = []
        self.s = self.mc.reset()


    def train(self):
        for i in range(episodes):
            print('EP' + str(i))
            self.s = self.mc.reset()  # for each episode, we need to reset the state in the envionment

            r_sum = 0.0  # hold the sum of rewards in this episode
            for j in range(max_iterations):
                r = self.one_round()  # return the reward in this iteration
                r_sum += r
                if self.done:  # if the car get to the flag, then we are done with this episode
                    break
                # self.mc.render()
            print(self.s)
            self.r.append(r_sum)


    # each iteration
    def one_round(self):
        q = self.calc_q(self.s)  # calculate the Q of this step
        self.a = self.greedy_action(q)  # find out the action using greedy method
        s_star, r, self.done = self.mc.step(self.a)  # take the step in the environment
        q_star = self.calc_q(s_star)  # calulate the new Q of the next step
        TD_target = self.get_target(r, q_star)  # find the TD target
        TD_error = self.get_error(q, TD_target)  # find the TD error
        self.update(TD_error, s_star)  # update the params
        return r

    # update method
    def update(self, error, s_star):
        w_new = np.zeros((self.mc.state_space, self.mc.action_space))  # create a new weight matrix
        for key, value in self.s.items():
            w_new[key][self.a] = value  # put this step's value in the new weight matrix
        t_w = self.w - learning_rate * error * w_new
        t_b = self.b - learning_rate * error * 1
        self.w = t_w
        self.b = t_b
        # self.w -= learning_rate * error * w_new  # set the weight matrix
        # self.b -= learning_rate * error * 1  # set the bias term
        self.s = s_star  # update the state

    # calculate TD target method
    def get_target(self, r, q):
        max_q = np.max(q)  # find the max in a list of Q
        t = gamma * max_q + r  # calc TD target
        return t

    # calculate TD error method
    def get_error(self, q, t):
        q_ = q[self.a]  # the Q of taking the action
        e = q_ - t  # different of Q and TD target
        return e

    # epsilon-greedy action selection method
    def greedy_action(self, q):
        best_action = np.argmax(q)  # best action we can take according to Q
        p = 1 - epsilon  # probability
        rand = np.random.uniform(0, 1)  # random a probability between 0 to 1
        if rand < p:  # if the random probability is less than p
            a = best_action  # take the best action
        else:
            a = np.random.randint(0, 3)  # take a random action
        return a

    # calculate Q method
    def calc_q(self, s):
        Q = []  # list holder of Q
        for i in range(self.w.shape[1]):
            temp = 0.0  # temp holder
            for key, value in s.items():
                temp += value * self.w[key][i]  # each value x the weight with given key
            temp += self.b
            Q.append(temp)
        return Q


# ------------------------------------------------------------------------------------

# write returns method
def write_returns(r):
    print('---------- Returns ----------')
    s = ''
    for i in r:
        s += str(i)
        print(i)
        s += '\n'
    returns_out.write(s)
    returns_out.close()


# write weight method
def write_weight(w, b):
    print('---------- Weight ----------')
    s = str(b)
    print(b)
    s += '\n'
    for i in w:
        for j in i:
            s += str(j)
            print(j)
            s += '\n'
    weight_out.write(s)
    weight_out.close()


# run tha agent
def run():
    a = Agent()
    a.train()
    write_returns(a.r)
    write_weight(a.w, a.b)


# main
if __name__ == "__main__":
    start_time = time.time()
    print('--------- Start ---------')
    run()
    end_time = time.time()
    print('--------- Finish ---------')
    print('Time used:', end_time - start_time)
