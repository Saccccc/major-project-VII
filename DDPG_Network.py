import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class CriticNetwork(keras.Model):
    def __init__(self, net, name='critic'):
        super(CriticNetwork, self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu'))
        self.q = Dense(1, activation=None)

    def call (self, state, action):       
        action_value = self.fc[0](tf.concat([state, action], axis=1))
        for i in range (1, len(self.net)):
            action_value = self.fc[i](action_value)
        q = self.q(action_value)
        return q

class ActorNetwork(keras.Model):
    def __init__(self, net, n_actions, name='actor'):
        super(ActorNetwork,self).__init__()
        self.net = net
        self.fc=[]
        for i in range (len(self.net)):
            self.fc.append(Dense(self.net[i], activation='relu'))
        self.n_actions = n_actions
        self.mu = Dense(self.n_actions, activation='sigmoid') 

    def call (self, state):
        prob = self.fc[0](state)
        for i in range (1, len(self.net)):
            prob = self.fc[i](prob)
        mu = self.mu(prob)
        return mu