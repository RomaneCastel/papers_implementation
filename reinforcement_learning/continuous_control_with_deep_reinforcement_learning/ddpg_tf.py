import os
import numpy as np
import tensorflow as tf


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self,sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # noise = OUActionNoise()
        # ourNoise = noise
        x = self.x_prev + self.theta*(self.mu-self.x_prev)*self.dt + self.sigma*np.sqrt(self.dt)*np.reandom.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        # give value for x_prev
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# different ways to implement replay buffer (some use dqueue)
# we gonna use set of arrays with numpy, cause more convenient for data type manipulation
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        # store current transition
        # index where we want to store
        # not to overflow -> modulo
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = action
        # done is True or False, don't count the rewards after the episod has ended
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        # sample the memories from 0 to mem_size but only if all the memory is filled
        max_mem = min(self.mem_cntr, self.mem_size)
        # take random choice 
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

# actor network, two actor networks
class Actor(object):
    # (learing rate, number of actions, name to distinguish regular actor network from target neural network, input dimension, session: tensorflow has the construct of the session
    # we have all the graphs and variables parameters, can have each class having its own session but tidy, first fully connected dimensions, second fc dimensions,
    # action bound, batch size, directory to save the models because can take a long time to run)
    def __init__(self, lr, n_actions, name, input_dims, sess, fc1_dims, fc2_dims, action_bound, batch_size=64, chkpt_dir='tmp/ddpg'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        # the purpose of action_bound is to accomodate environments where the action is either greater than plus or minus negative one,
        # if goes from -2 to 2, the tanh would sample only half of the range, so multiplicative factor for that
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir
        self.build_network()
        # have to do soft update rule, have to find a way to keep tracks of parameters in each nets
        # regular and target nets to be independant, by the scope
        self.params = tf.trainable_variables(scope=self.name)
        self.saver - tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.ckpt')
        # going to compute the gradients by hand, write some functions that will facilitate that
        # mu the actual actions of the agent, param: net parameters and action_gradient
        # last part of policy gradient in pseudo-code
        # the two following line because we want to compute the gradients manually of the critic w.r.t the actions taken
        self.unormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self,actor_gradients, self.params))
    
    def build_network(self):
        # every net gets its own scope
        with tf.variables_scope(self.name):
            # name for debugging
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            # gradient of Q w.r.t to each action so number of dimensions = number of actions
            self.action_gradient = tf.placeholder(tf.float32, shape=[None, self.n_actions])
            # construct the actual net
            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1), bias_initializer=tf.random_uniform_initializer(-f1, f1))
            # batch normalization
            batch1 = tf.layers.batch_normalization(dense1)
            # debate about wether or not we should do the activation before or after the batch normalization
            # here decide to do it after because relu might troncate to much for the needed statistics
            layer1_activation = tf.nn.relu(batch1)
            
            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2), bias_initializer=tf.random_uniform_initializer(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.relu(batch2)

            # output layer, the actual policy of our region (deterministic)
            f3 = 0.003
            mu = tf.layers.dens(layer2_activation, units=self.n_actions, activation='tanh', kernel_initializer=tf.random_uniform_initializer(-f3, f3), bias_initializer=tf.random_uniform_initializer(-f3, f3))
            # take into account that our environment may very well require actions that have values greater than +- 1
            self.mu = tf.multiply(mu, self.action_bound)

    # way of getting the actual actions out of the networks
    def predict(self):
        return self.sess.run(self.mu, feed_dict={self.input: inputs})
    
    def train(self, inputs, gradients):
        self.sess.run(self.optimize, feed_dict={self.input : inputs, self.action_gradient: gradients})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # save the current session to the checkpoint file
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

class Critic(object):
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims, fc2_dims, batch_size=64, checkpoint_file='tmp/ddpf'):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        # the purpose of action_bound is to accomodate environments where the action is either greater than plus or minus negative one,
        # if goes from -2 to 2, the tanh would sample only half of the range, so multiplicative factor for that
        self.chkpt_dir = chkpt_dir
        self.build_network()
        # have to do soft update rule, have to find a way to keep tracks of parameters in each nets
        # regular and target nets to be independant, by the scope
        self.params = tf.trainable_variables(scope=self.name)
        self.saver - tf.train.Saver()
        self.checkpoint_file = os.path.join(chkpt_dir, name+'_ddpg.ckpt')

        self.optimize = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # calculate the gradients of Q w.r.t a
        self.action_gradients = tf.gradients(self.q, self.actions)

    def build_network(self):
        with tf.variables_scope(self.name):
            # None for shape because do not know in advance the batch of inputs beforehand
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            # only take into account the actions in the second hidden layer of the critic neural net
            self.actions = tf.placeholder(tf.float32, shape=[None, *self,n_actions], name='actions')
            # woth Q learning, wa have a target value, quantity yi in the pseudo-code in the paper
            # batch size 1 because scaler
            self.q_target = tf.placeholder(tf.float32, shape=[None, 1], name='targets')

            f1 = 1 / np.sqrt(self.fc1_dims)
            dense1 = tf.layers.dense(self.input, units=self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1), bias_initializer=tf.random_uniform_initializer(-f1, f1))
            # batch normalization
            batch1 = tf.layers.batch_normalization(dense1)
            # debate about wether or not we should do the activation before or after the batch normalization
            # here decide to do it after because relu might troncate to much for the needed statistics
            layer1_activation = tf.nn.relu(batch1)

            f2 = 1 / np.sqrt(self.fc2_dims)
            dense2 = tf.layers.dense(layer1_activation, units=self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2), bias_initializer=tf.random_uniform_initializer(-f2, f2))
            batch2 = tf.layers.batch_normalization(dense2)
            # get rid of the activation when comparing to Actor network, because take into consideration the actions

            action_in = tf.layers.dense(self.actions, units=self.fc2_dims, activation='relu')

            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.relu(state_actions)

            # calculate the actual output of the layer
            f3 = 0.003
            self.q = tf.layers.dens(state_actions, units=1, kernel_initializer=tf.random_uniform_initializer(-f3, f3), bias_initializer=tf.random_uniform_initializer(-f3, f3), kernel_regularizer=tf.keras.regularizers.l2(0.01))

            # self.q is the output of the deep neural net
            self.loss = tf.losses.mean_squared_error(self,q_target, self.q)

    def predict(self, inputs, actions):
        return self.sess.run(self.q, feed_dict={self.input: inputs, self.actions: actions})

    def train(self, inputs, qctions, q_target):
        return self.sess.run(self.optimize, feed_dict={self.input: inputs, self.actions: actions, self.q_target: q_target})

    # get the action gradients operation up above
    def get_action_gradients(self, inputs, actions):
        return self.sess.run(self.action_gradients, feed_dict={self.input: inputs, self.actions: actions})

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # save the current session to the checkpoint file
        self.saver.save(self.sess, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.saver.restore(self.sess, self.checkpoint_file)

            
# Now all we need is our agent, it gathers everything 
class Agent(object):
    # (lr for the actor 0.001, lr for the critic 0.001)
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, n_actions=2, max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        # one single session for all 4 networks
        self.sess = tf.Session()
        self.actor = Actor(alpha, n_actions, 'Actor', input_dims, self.sess, layer1_size, layer2_size, env.action_space.high)
        self.critic = Critic(beta, n_actions, 'Critic', input_dims, self.sess, layer1_size, layer2_size)

        self.target_actor = Actor(alpha, n_actions, 'TargetActor', input_dims, self.sess, layer1_size, layer2_size, env.action_space.high)
        self.target_critic = Critic(beta, n_actions, 'TargetCritic', input_dims, self.sess, layer1_size, layer2_size)

        # now we need noise
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        # need operation to perform the soft updates
        self.update_critic = [self.target_critic.params[i].assign(tf.multiply(self.critic_params[i], self.tau) + tf.multiply(self.target_critic_params[i], 1 - self.tau)) for i in range(len(self.target_critic_params))]
        self.update_actor = [self.target_actor.params[i].assign(tf.multiply(self.actor_params[i], self.tau) + tf.multiply(self.target_actor_params[i], 1 - self.tau)) for i in range(len(self.target_actor_params))]

        # init the variables
        self.sess.run(tf.global_variables_initializer())
        
        # according to the pseudo-code in the paper
        # at the beginning the target gets updated with the 4 value of the evaluations networks -> first=True
        self.update_network_parameters(first=True)

    def update_network_parameters(self, first=False):
        if first:
            # save the old value of tau to set it again
            old_tau = self.tau
            self.tau = 1.0
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)
            self.tau = old_tau
        else:
            self.target_critic.sess.run(self.update_critic)
            self.target_actor.sess.run(self.update_actor)

    # need a way of storing transitions
    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    # the input variable to be shaped batch_size by n_actions, want to reshape the state to be 1 by the observation space
    def choose_action(self, state):
        # because self.input as shape None by input_dims
        state = state[np.newaxis, :]
        mu = self.actor.predict(state)
        noise = self.noise
        mu_prime = mu + noise

        # mu_prime is a tuple so we want the 0-th element
        return mu_prime[0]

    def learn(self):
        # if not filled out the memory
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        # now need to do the update from the paper
        # compute yi, need q' the target critic network and the output of the target actor network (mu_prime)
        # need the output from the critic as well as the output of the actor networks
        # pass states and actions through all the 4 nets to get the learning function
        critic_value_ = self.target_critic.predict(new_state, self.target_actor.predict(new_state))
        # compute yi
        target = []
        for j in range(self.batch_size):
            # get no reward after the terminal state, that's why the 'done' appears
            target.append(reward[j]+self.gamma*critic_value_[j]*done[j])
        # reshape to be consistent with the placeholders
        target = np.reshape(target, (self.batch_size,1))

        # call critic train function
        _ = self.critic.train(state, action, target)

        # actor updates
        a_outs = self.actor.predict(state)
        # get the remember to do feedforward and get the gradient of the critic w.r.t to the actions taken
        grads = seld.critic.get_action_gradients(state, a_outs)
        # train the actor
        self.actor.train(state, grads[0])
        # update net params
        self.update_network_parameters()

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()