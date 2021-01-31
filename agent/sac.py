import tensorflow as tf
import copy
import datetime
import random
import numpy as np
from collections import deque

from utils import gaussian_likelihood

class Actor(tf.keras.Model):
    def __init__(self, action_size, use_visual):
        super(Actor, self).__init__()
        self.use_visual = use_visual
        if use_visual:
            self.conv1 = tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu', padding="same")
            self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu', padding="same")
            self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu', padding="same")
            self.flat  = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.mu = tf.keras.layers.Dense(action_size, activation='sigmoid')
        self.log_std = tf.keras.layers.Dense(action_size, activation='tanh')

    def call(self, x):
        x = tf.convert_to_tensor(x, tf.float32)
        if self.use_visual:
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.mu(x), self.log_std(x)

class Critic(tf.keras.Model):
    def __init__(self, action_size, use_visual):
        super(Critic, self).__init__()
        self.use_visual = use_visual
        if use_visual:
            self.conv1 = tf.keras.layers.Conv2D(32, 4, strides=2, activation='relu', padding="same")
            self.conv2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu', padding="same")
            self.conv3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu', padding="same")
            self.flat  = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(512, activation='relu')
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.q   = tf.keras.layers.Dense(action_size)
    
    def call(self, x1, x2):
        if self.use_visual:
            x1 = self.conv1(x1)
            x1 = self.conv2(x1)
            x1 = self.conv3(x1)
            x1 = self.flat(x1)
        x = tf.concat([x1, x2], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.q(x)

class SACAgent:
    def __init__(self,
                 action_size,
                 use_visual,
                 actor_lr,
                 critic_lr,
                 alpha_lr,
                 batch_size,
                 maxlen,
                 use_dynamic_alpha,
                 static_log_alpha,
                 tau,
                 gamma,
                 ):
        self.actor = Actor(action_size, use_visual)
        self.critic1 = Critic(action_size, use_visual)
        self.critic2 = Critic(action_size, use_visual)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)
        self.critic1_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.critic2_optimizer = tf.keras.optimizers.Adam(critic_lr)

        self.use_dynamic_alpha = use_dynamic_alpha
        if use_dynamic_alpha:
            self.log_alpha = tf.Variable(initial_value=0., trainable=True)
            self.alpha_optimizer = tf.keras.optimizers.Adam(alpha_lr)
        else:
            self.log_alpha = static_log_alpha
            self.alpha_optimizer = None
        self.alpha = tf.math.exp(self.log_alpha)

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size    
        self.memory = deque(maxlen=maxlen)

        self.save_path = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = tf.summary.create_file_writer(f"./summary/{self.save_path}")

    def act(self, state, training=True):
        mu, log_std = self.actor(state, training=training)
        action = tf.sigmoid(tf.random.normal(tf.shape(mu), mu, tf.math.exp(log_std)))
        return action.numpy()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def update_target(self):
        for t_c, c in zip(self.target_critic1.trainable_weights, self.critic1.trainable_weights):
            t_c.assign((1-self.tau)*t_c + self.tau*c)
        for t_c, c in zip(self.target_critic2.trainable_weights, self.critic2.trainable_weights):
            t_c.assign((1-self.tau)*t_c + self.tau*c)

    def learn(self):
        batch = random.sample(self.memory, self.batch_size)
        state      = np.stack([b[0] for b in batch], axis=0)
        action     = np.stack([b[1] for b in batch], axis=0)
        reward     = np.stack([b[2] for b in batch], axis=0).astype(np.float32)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done       = np.stack([b[4] for b in batch], axis=0).astype(np.float32)

        actor_loss, critic1_loss, critic2_loss, alpha_loss = \
            self.train(state, action, reward, next_state, done)

        self.update_target()
        self.alpha = tf.math.exp(self.log_alpha)

        return actor_loss, critic1_loss, critic2_loss, alpha_loss

    @tf.function
    def train(self, state, action, reward, next_state, done):
        next_mu, next_log_std = self.actor(next_state)
        next_action = tf.random.normal(tf.shape(next_mu), next_mu, tf.math.exp(next_log_std))
        next_log_prob = gaussian_likelihood(next_action, next_mu, next_log_std)
        next_target_q1 = self.target_critic1(next_state, next_action)
        next_target_q2 = self.target_critic2(next_state, next_action)
        min_next_target_q = tf.math.minimum(next_target_q1, next_target_q2) - self.alpha * next_log_prob
        target_q = reward + (1.-done)*self.gamma*min_next_target_q
        with tf.GradientTape(persistent=True) as tape:
            # Critic
            q1 = self.critic1(state, action)
            q2 = self.critic2(state, action)

            critic1_loss = tf.reduce_mean(tf.keras.losses.mse(target_q, q1))
            critic2_loss = tf.reduce_mean(tf.keras.losses.mse(target_q, q2))

            # Actor
            min_q = tf.math.minimum(q1, q2)
            mu, log_std = self.actor(state)
            log_prob = gaussian_likelihood(action, mu, log_std)
            actor_loss = tf.reduce_mean(self.alpha*log_prob - min_q)

            # Alpha
            alpha_loss = -tf.reduce_mean(self.log_alpha * log_prob)


        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_weights)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_weights))
        
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_weights)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_weights))

        actor_grads = tape.gradient(actor_loss, self.actor.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_weights))

        if self.use_dynamic_alpha:
            alpha_grad = tape.gradient(alpha_loss, self.log_alpha)
            self.alpha_optimizer.apply_gradients([(alpha_grad, self.log_alpha)])

        return actor_loss, critic1_loss, critic2_loss, alpha_loss

    def load(self, path):
        print("... Load Model ...")
        self.actor.load_weights(path+"/actor")
        self.critic1.load_weights(path+"/critic1")
        self.critic2.load_weights(path+"/critic2")
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        if self.use_dynamic_alpha:
            self.log_alpha.load_weights(self.save_path+"/alpha")

    def save(self):
        print("... Save Model ...")
        self.actor.save_weights(self.save_path+"/actor")
        self.critic1.save_weights(self.save_path+"/critic1")
        self.critic2.save_weights(self.save_path+"/critic2")
        # if self.use_dynamic_alpha:
        #     self.log_alpha.save_weights(self.save_path+"/alpha")
        #     self.log_alpha

    def write(self, score, actor_loss, critic1_loss, critic2_loss, alpha_loss, episode):
        with self.writer.as_default():
            tf.summary.scalar("sac/score", score, step=episode)
            tf.summary.scalar("sac/actor_loss", actor_loss, step=episode)
            tf.summary.scalar("sac/critic1_loss", critic1_loss, step=episode)
            tf.summary.scalar("sac/critic2_loss", critic2_loss, step=episode)
            # tf.summary.scalar("sac/alpha_loss", alpha_loss, step=episode)
    