import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from DDPG_Network import ActorNetwork, CriticNetwork
from ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(self, input_dims, n_actions,nn_actor, nn_critic, env=None, alpha=1e-3, beta=2e-3, 
                 gamma=0.9, max_size=10000, tau=0.0001, batch_size=32, noise=1e-3):
        
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.noise = noise
        self.env=env
        self.max_action = env.max_action
        self.min_action = env.min_action
        
        self.actor = ActorNetwork(nn_actor, n_actions=self.n_actions, name='actor')
        self.critic = CriticNetwork(nn_critic, name='critic')
        self.target_actor = ActorNetwork(nn_actor, n_actions=self.n_actions, name='target_actor')
        self.target_critic = CriticNetwork(nn_critic, name='target_critic')

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))
        
        self.update_network_parameters(tau=tau) #dr phill memberi tau=1
        
        

    def update_network_parameters(self, tau=None):
        
        if tau is None:
            tau = self.tau
        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)
        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def remember(self, state, action, reward, new_state):
        self.memory.store_transition(state, action, reward, new_state)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        actions += (tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise)) #add noise
        # actions += (tf.random.normal(shape=[self.n_actions], mean=0.0, stddev=self.noise))
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)
        return actions[0].numpy()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state = self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(states_, target_actions), 1)
            critic_value = tf.squeeze(self.critic(states, actions), 1) #Q
            target = rewards + self.gamma*critic_value_ #r+gama*Q_next
            critic_loss = keras.losses.MSE(target, critic_value)
        critic_network_gradient = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_network_gradient, self.critic.trainable_variables))


        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic(states, new_policy_actions) 
            actor_loss = tf.math.reduce_mean(actor_loss)
        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))
        
        self.update_network_parameters()
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

