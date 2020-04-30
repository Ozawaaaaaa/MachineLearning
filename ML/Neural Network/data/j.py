#!/usr/bin/env python
# coding: utf-8

# In[297]:


import numpy as np


# In[198]:


def read_X(x):
    fin = open(x,'rt',encoding = 'utf-8')
    output=[]
    for line in fin:
        temp=line.split(',')
        temp_2=[]
        temp_2.append(1)
        for a in range(1,len(temp)):
            temp_2.append(int(temp[a]))
        output.append(np.asarray(temp_2))
    return np.asarray(output)


# In[436]:


def read_y(x):
    fin = open(x,'rt',encoding = 'utf-8')
    output=[]
    counter=0
    for line in fin:
        output.append([])
        temp=line.split(',')
        for a in range(10):
            if int(temp[0])==a:
                output[counter].append(1)
            else:
                output[counter].append(0)
        counter+=1
    return np.asmatrix(output)


# In[729]:


def read_ylist(x):
    fin = open(x,'rt',encoding = 'utf-8')
    output=[]
    for line in fin:
        output.append(int(line.split(',')[0]))
    return output


# In[437]:


def initial_alpha(X,init_flag,hidden_units):
    if init_flag == 1:
        temp = np.random.uniform (-0.1, 0.1, size = (hidden_units,X.shape[1]-1))
        alpha=[]
        for a in temp:
            b=np.insert(a, 0, 0)
            alpha.append(b)
        alpha=np.asarray(alpha)
    else:
        alpha = np.zeros((hidden_units, X.shape[1]))
    return alpha


# In[438]:


def initial_beta(init_flag,hidden_units):
    if init_flag == 1:
        temp = np.random.uniform (-0.1, 0.1, size = (10,hidden_units))
        beta=[]
        for a in temp:
            b=np.insert(a, 0, 0)
            beta.append(b)
        beta=np.asarray(beta)
    else:
        beta = np.zeros((10, hidden_units+1))
    return beta


# In[ ]:





# In[439]:


def LINEARFORWARD(a,b):
    return np.matmul(b,a.transpose()).transpose()


# In[489]:


def LINEARBACKWARD(a,b,c,d):
    par_de=np.matmul(d.transpose(),a)
    x_de=np.matmul(d,b)
    return par_de,x_de


# In[490]:


def SIGMOIDFORWARD(a):
    temp=1/(1+np.exp(-a))
    return temp


# In[491]:


def SIGMOIDBACKWARD(a,b,c):
    return np.multiply(np.exp(-a)/np.square(1+np.exp(-a)),c)


# In[492]:


def SOFTMAXFORWARD(a):
    return np.exp(a)/np.sum(np.exp(a))


# In[493]:


def SOFTMAXBACKWARD(a,b):
    return a-b


# In[494]:


def CROSSENTROPYFORWARD(a,b):
    return -np.sum(np.multiply(b,np.log(a)))


# In[699]:


def NNFORWARD(x,y,alpha,beta):
    a=LINEARFORWARD(x,alpha)
    z=SIGMOIDFORWARD(a)
    z=np.insert(z, 0, 1)
    b=LINEARFORWARD(z,beta)
    y_hat=SOFTMAXFORWARD(b)
    j=CROSSENTROPYFORWARD(y_hat,y)
    return [x,a,z,b,y_hat,j]


# In[700]:


def NNBACKWARD(x,y,alpha,beta,temp):
    g_b=SOFTMAXBACKWARD(temp[4],y)
    g_beta,g_z=LINEARBACKWARD(temp[2],beta,temp[3],g_b)
    g_z=np.delete(g_z, 0)
    g_a=SIGMOIDBACKWARD(temp[1],temp[2],g_z)
    g_alpha,g_x=LINEARBACKWARD(x,alpha,temp[1],g_a)
    return g_alpha,g_beta


# In[724]:


def croosentropy(alpha,beta,x,y):
    croosentropy=0
    pre_y=[]
    for a in range(len(x)):
        train_list=NNFORWARD(x[a],y[a],alpha,beta)
        croosentropy+=train_list[-1]
        pre_y.append(np.argmax(train_list[-2]))
    croosentropy=croosentropy/len(x)
    return croosentropy,pre_y


# In[735]:


def cal_error(pre_y,y):
    error=0
    for a in range(len(pre_y)):
        if pre_y[a]==y[a]:
            continue
        else:
            error+=1
    return error/len(y)


# In[748]:


def writeoutput(x,y):
    fout = open(x,'wt',encoding = 'utf-8')
    for i in y:
        fout.write(str(i))
        fout.write('\n')


# In[755]:


def writemetrix(x,y_train,y_test,error):
    fout = open(x,'wt',encoding = 'utf-8')
    for a in range(len(y_train)):
        fout.write(y_train[a])
        fout.write('\n')
        fout.write(y_test[a])
        fout.write('\n')
    fout.write('error(train): '+str(error[0]))
    fout.write('\n')
    fout.write('error(test): '+str(error[1]))


# In[758]:

import sys
if __name__ == '__main__':
    train_X=np.asmatrix(read_X(sys.argv[1]))
    train_y=np.asmatrix(read_y(sys.argv[1]))
    train_ylist=read_ylist(sys.argv[1])
    test_X=np.asmatrix(read_X(sys.argv[2]))
    test_y=np.asmatrix(read_y(sys.argv[2]))
    test_ylist=read_ylist(sys.argv[2])
    alpha=np.asmatrix(initial_alpha(train_X,int(sys.argv[8]),int(sys.argv[7])))
    beta=np.asmatrix(initial_beta(int(sys.argv[8]),int(sys.argv[7])))
    cross_train_list=[]
    cross_test_list=[]
    for e in range(int(sys.argv[6])):
        for a in range(len(train_X)):
            list_o=NNFORWARD(train_X[a],train_y[a],alpha,beta)
            g_alpha,g_beta=NNBACKWARD(train_X[a],train_y[a],alpha,beta,list_o)
            alpha=alpha-float(sys.argv[9])*g_alpha
            beta=beta-float(sys.argv[9])*g_beta
        cross_train,pre_train=croosentropy(alpha,beta,train_X,train_y)
        cross_train_list.append('epoch='+str(e+1)+' crossentropy(train): '+str(cross_train))
        cross_test,pre_test=croosentropy(alpha,beta,test_X,test_y)
        cross_test_list.append('epoch='+str(e+1)+' crossentropy(test): '+str(cross_test))
    writeoutput(sys.argv[3],pre_train)
    writeoutput(sys.argv[4],pre_test)
    error=[cal_error(pre_train,train_ylist),cal_error(pre_test,test_ylist)]
    writemetrix(sys.argv[5],cross_train_list,cross_test_list,error)


# In[ ]:




