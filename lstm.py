#-*-coding:utf-8-*
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


#set FLAG and get data
class BaseConfig(object):
    seq_lenght=20      #seq lenght
    batch_size=64      #batch_size
    feature_num=1      #dim of a seq
    lstm_size=64   #hidden layer units
    lstm_layers=6
    keep_prob=0.5
    lr=0.00001        #learn rate
    sep=0.9         #train and test sep
    epoch_size=10000 #train number
    
config=BaseConfig()

sp500=pd.read_csv('sp500.csv')
data=sp500['Close']
data_shift1=data.shift(1)
data_diff1=(data-data_shift1)[1:]
#以折线图展示data
#plt.figure()
#plt.plot(data)
#plt.show()



def division_data(data,config,regular=True):
    #data:np.array([1,2,3,4,5,6,7,8])
    #all_data
    if regular:
        data=(np.array(data)-np.array(data).mean())/np.array(data).std()
        
    X,y=[],[]
    for i in range(len(data) - config.seq_lenght-1):
        X.append(data[i:i+config.seq_lenght])
        y.append(data[i+config.seq_lenght]+1)
    X=np.array(X)[:,:,np.newaxis]
    y=np.array(y)[:,np.newaxis]
    #index
    train_size=int(config.sep*len(X))
    split_index=[1]*train_size
    split_index.extend([0] * (len(X) - train_size))
    np.random.shuffle(split_index)

    #division all_data into train and test data
    train_X,train_y,test_X,test_y=[],[],[],[]
    for i,v in enumerate(split_index):
        if v==0:
            test_X.append(X[i])
            test_y.append(y[i])
        else:
            train_X.append(X[i])
            train_y.append(y[i])
    train_X=np.array(train_X).astype('float32')
    train_y=np.array(train_y).astype('float32')
    test_X=np.array(test_X).astype('float32')
    test_y=np.array(test_y).astype('float32')
    return train_X,train_y,test_X,test_y
train_X,train_y,test_X,test_y=division_data(data_diff1,config)



#general W
def W_var(in_dim,out_dim):
    return tf.Variable(tf.random_normal([in_dim,out_dim]),tf.float32)

#general b
def b_var(out_dim):
    return tf.Variable(tf.random_normal([out_dim,]),tf.float32)

#lstm : 64 lstm_size, 2 lstm_layer

def lstm_cell(config,keep_prob):
    temp=tf.contrib.rnn.BasicLSTMCell(config.lstm_size)
    drop = tf.nn.rnn_cell.DropoutWrapper(temp, output_keep_prob=keep_prob)
    return drop

def lstm_layers(config,X,keep_prod):
    #input
    
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
    [lstm_cell(config,keep_prod) for _ in range(config.lstm_layers)])
    initial_state = stacked_lstm.zero_state(config.batch_size, tf.float32)
    
    outputs, final_state = tf.nn.dynamic_rnn(stacked_lstm, X, 
          initial_state=initial_state)
    return outputs,final_state
        
def output_layers(config,output_lstm):
    in_size=output_lstm.get_shape()[-1].value
    output_lstm=output_lstm[:,-1,:]
    output_lstm=tf.reshape(output_lstm,[-1,in_size])
    W=W_var(in_size,config.feature_num)
    b=b_var(config.feature_num)
    output_final=tf.add(tf.matmul(output_lstm,W),b)
    return output_final

def loss_function(output_final,Y):
    print(output_final.shape,Y.shape)
    loss=tf.reduce_mean(tf.square(output_final-Y))
    return loss

def optimizer_function(config,loss):
    opt=tf.train.AdamOptimizer(config.lr).minimize(loss)
    return opt


class train_body:

    def __init__(self):
        self.X_placehold=tf.placeholder(tf.float32, [None,config.seq_lenght,config.feature_num])
        self.Y_placehold=tf.placeholder(tf.float32, [None,1])
        self.keep_prod=tf.placeholder(tf.float32)
        self.output_lstm,_=lstm_layers(config,self.X_placehold,self.keep_prod)

        self.output_final=output_layers(config,self.output_lstm)

        self.loss=loss_function(self.output_final,self.Y_placehold)

        self.opt=optimizer_function(config,self.loss)

        



def myrun():
    
    tb=train_body()

    #save model

    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        
        tf.global_variables_initializer().run()
        
        range(int(len(train_X)/config.batch_size))

        for e in range(config.epoch_size):
            loss_values1=np.array([])
            for i in range(int(len(train_X)/config.batch_size)):
                
                tempx=train_X[i*config.batch_size:i*config.batch_size+config.batch_size]
                tempy=train_y[i*config.batch_size:i*config.batch_size+config.batch_size]
                #print(tempx)
                tmp_loss_value,_=sess.run([tb.loss,tb.opt],feed_dict={tb.X_placehold:tempx,tb.Y_placehold:tempy,tb.keep_prod:0.5})
                loss_values1=np.append(loss_values1,tmp_loss_value)
                
            if e%10==0:
                loss_values2=np.array([])
                for i in range(int(len(test_X)/config.batch_size)):

                    tempx=test_X[i*config.batch_size:i*config.batch_size+config.batch_size]
                    tempy=test_y[i*config.batch_size:i*config.batch_size+config.batch_size]
                    #print(tempx)
                    tmp_loss_value=sess.run([tb.loss],feed_dict={tb.X_placehold:tempx,tb.Y_placehold:tempy,tb.keep_prod:1})
                    loss_values2=np.append(loss_values2,tmp_loss_value)
               
                print('std is: ',train_y.std())
                print('ephoch: '+ str(e)+'\ntrain loss is: '+str(loss_values1.mean())
                      +'; test loss is: ' + str(loss_values2.mean()))
                #print('ephoch: '+ str(e)+'\ntrain loss is: '+str(loss_values1.mean()))
                print ("save model:",saver.save(sess,'./sp500_model/sp500.model\n'))

myrun()



