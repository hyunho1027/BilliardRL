import numpy as np
import tensorflow as tf
from collections import deque
import random
from mlagents.envs import UnityEnvironment
import datetime
from utils import logit, clip_b4_exp, vec_data_augmentation, vis_data_augmentation

np.set_printoptions(precision=3)
np.random.seed(1)
tf.set_random_seed(1)

vis_mode = True
s_size = [128, 64, 3] if vis_mode else [8]
a_size = 2

batch_size = 256

tau = 1e-3
actor_lr = 5e-5
critic_lr = 3e-4
discount_factor = 0.9

train_start_step = 1000
mem_maxlen = 50000

mu = 0
theta = 0.01
sigma = 0.02

load_model = False
train_mode = True

load_path = ""
save_path = "./summary/ddpg/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_interval = 1000

class OU_noise:
    def __init__(self):
        self.reset()

    def reset(self):
        self.X = np.ones(a_size) * mu
    
    def sample(self):
        dx = theta * (mu - self.X)
        dx += sigma * np.random.randn(len(self.X))
        self.X += dx
        return self.X

class Actor():
    def __init__(self, name):
        with tf.variable_scope(name):
            self.s = tf.placeholder(tf.float32, [None, *s_size])

            if vis_mode:
                self.s_normalize = (self.s - (255.0 / 2)) / (255.0 / 2)

                self.conv1 = tf.layers.conv2d(self.s_normalize, filters=32,
                                            activation=tf.nn.leaky_relu, kernel_size=[8,8],
                                            strides=[4,4], padding="SAME")
                self.conv2 = tf.layers.conv2d(self.conv1, filters=64,
                                            activation=tf.nn.leaky_relu, kernel_size=[4,4],
                                            strides=[2,2], padding="SAME")
                self.conv3 = tf.layers.conv2d(self.conv2, filters=64,
                                            activation=tf.nn.leaky_relu, kernel_size=[3,3],
                                            strides=[1,1], padding="SAME")

                self.flat = tf.layers.flatten(self.conv3)
                self.fc1 = tf.layers.dense(self.flat, 128, tf.nn.leaky_relu)
            else:
                self.fc1 = tf.layers.dense(self.s, 128, tf.nn.leaky_relu)
                
            self.fc2 = tf.layers.dense(self.fc1, 128, tf.nn.leaky_relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, tf.nn.leaky_relu)
            self.a = tf.sigmoid(clip_b4_exp(tf.layers.dense(self.fc3, a_size)))

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class Critic():
    def __init__(self, name):
        with tf.variable_scope(name):
            self.s = tf.placeholder(tf.float32, [None, *s_size])
            
            if vis_mode:
                self.s_normalize = (self.s - (255.0 / 2)) / (255.0 / 2)

                self.conv1 = tf.layers.conv2d(self.s_normalize, filters=32,
                                            activation=tf.nn.leaky_relu, kernel_size=[8,8],
                                            strides=[4,4], padding="SAME")
                self.conv2 = tf.layers.conv2d(self.conv1, filters=64,
                                            activation=tf.nn.leaky_relu, kernel_size=[4,4],
                                            strides=[2,2], padding="SAME")
                self.conv3 = tf.layers.conv2d(self.conv2, filters=64,
                                            activation=tf.nn.leaky_relu, kernel_size=[3,3],
                                            strides=[1,1], padding="SAME")

                self.flat = tf.layers.flatten(self.conv3)
                self.fc1 = tf.layers.dense(self.flat, 128, tf.nn.leaky_relu)
            else:
                self.fc1 = tf.layers.dense(self.s, 128, tf.nn.leaky_relu)

            self.a = tf.placeholder(tf.float32, [None, a_size])
            self.concat = tf.concat([self.fc1, self.a],axis=-1)
            self.fc2 = tf.layers.dense(self.concat, 128, tf.nn.leaky_relu)
            self.fc3 = tf.layers.dense(self.fc2, 128, tf.nn.leaky_relu)
            self.q = tf.squeeze(tf.tanh(clip_b4_exp(tf.layers.dense(self.fc3, 1))))

        self.trainable_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name)

class DDPGAgent:
    def __init__(self):
        self.actor = Actor("actor")
        self.critic = Critic("critic")
        self.target_actor = Actor("target_actor")
        self.target_critic = Critic("target_critic")
        
        self.target_q = tf.placeholder(tf.float32, [None])
        critic_loss = tf.losses.mean_squared_error(self.target_q, self.critic.q)
        with tf.control_dependencies(self.critic.trainable_var):
            self.train_critic = tf.train.AdamOptimizer(critic_lr).minimize(critic_loss)

        action_grad = tf.clip_by_value(tf.gradients(self.critic.q, self.critic.a),-10,10)
        policy_grad = tf.gradients(self.actor.a, self.actor.trainable_var, action_grad)
        for idx, grads in enumerate(policy_grad):
            policy_grad[idx] = -grads/batch_size
        with tf.control_dependencies(self.actor.trainable_var):
            self.train_actor = tf.train.AdamOptimizer(actor_lr).apply_gradients(zip(policy_grad, self.actor.trainable_var))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary, self.merge = self.make_summary()
        self.OU = OU_noise()
        self.memory = deque(maxlen=mem_maxlen)

        self.soft_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            self.soft_update_target.append(self.target_actor.trainable_var[idx].assign(
                ((1 - tau) * self.target_actor.trainable_var[idx].value()) + (tau * self.actor.trainable_var[idx].value())))
        for idx in range(len(self.critic.trainable_var)):
            self.soft_update_target.append(self.target_critic.trainable_var[idx].assign(
                ((1 - tau) * self.target_critic.trainable_var[idx].value()) + (tau * self.critic.trainable_var[idx].value())))
        
        init_update_target = []
        for idx in range(len(self.actor.trainable_var)):
            init_update_target.append(self.target_actor.trainable_var[idx].assign(self.actor.trainable_var[idx]))
        for idx in range(len(self.critic.trainable_var)):
            init_update_target.append(self.target_critic.trainable_var[idx].assign(self.critic.trainable_var[idx]))
        self.sess.run(init_update_target)

    def train_model(self):
        mini_batch = random.sample(self.memory, batch_size)
        states = np.asarray([sample[0] for sample in mini_batch])
        actions = np.asarray([sample[1] for sample in mini_batch])
        rewards = np.asarray([sample[2] for sample in mini_batch])
        next_states = np.asarray([sample[3] for sample in mini_batch])
        dones = np.asarray([sample[4] for sample in mini_batch])

        target_actor_as= self.sess.run(self.target_actor.a, feed_dict={self.target_actor.s: next_states})
        target_critic_qs = self.sess.run(self.target_critic.q, feed_dict={self.target_critic.s: next_states, self.target_critic.a: target_actor_as})
        target_qs = np.asarray([r + discount_factor * (1 - d) * target_critic_q for r, target_critic_q, d in zip(rewards, target_critic_qs, dones)])
        self.sess.run(self.train_critic, feed_dict={self.critic.s: states, self.critic.a: actions, self.target_q:target_qs})

        actions_for_train = self.sess.run(self.actor.a, feed_dict={self.actor.s: states})
        self.sess.run(self.train_actor, feed_dict={self.actor.s: states, self.critic.s: states, self.critic.a: actions_for_train})

        self.sess.run(self.soft_update_target)

    def get_action(self, s, train_mode):
        a = self.sess.run(self.actor.a, feed_dict={self.actor.s: s})
        noise = self.OU.sample()
        return a if train_mode else a + noise
    
    def append_sample(self, s, a, r, next_s, d):
        self.memory.append((s, a, r, next_s, d))
   
    def save_model(self):
        self.saver.save(self.sess, save_path + "/model.ckpt")
    
    def make_summary(self):
        self.summary_reward = tf.placeholder(tf.float32)
        tf.summary.scalar("reward", self.summary_reward)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def write_summary(self, reward, episode):
            self.summary.add_summary(self.sess.run(self.merge,  feed_dict={self.summary_reward: reward}), episode)

if __name__ == '__main__' :
    env_name = "./env/Billiard"
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    agent = DDPGAgent()

    episode = 0
    recent_rs = deque(maxlen=100)
    while True:
        episode += 1
        env_info = env.reset(train_mode=train_mode)[default_brain]
        s = np.array(env_info.visual_observations[0])[0,:,:,:] if vis_mode else env_info.vector_observations[0]
        d = False

        while not d:
            a = agent.get_action([s], train_mode)[0]
            env_info = env.step(a)[default_brain]
            next_s = np.array(env_info.visual_observations[0])[0,:,:,:] if vis_mode else env_info.vector_observations[0]
            r = env_info.rewards[0]
            d = env_info.local_done[0]
            recent_rs.append(r)

            if train_mode:
                s_augs, a_augs, next_s_augs = vis_data_augmentation(s, a, next_s) if vis_mode else vec_data_augmentation(s, a, next_s)
                for s_aug, a_aug, next_s_aug in zip(s_augs, a_augs, next_s_augs):
                    agent.append_sample(s_aug, a_aug, r, next_s_aug, d)

                if len(agent.memory) >= train_start_step:
                    agent.train_model()

        print(f"{episode+1} Episode / Recent Reward: {np.mean(recent_rs):.3f} / Mem Len: {len(agent.memory)}")
        agent.write_summary(np.mean(recent_rs), episode)
 
        if train_mode and (episode+1) % save_interval==0:
            print("model saved")
            agent.save_model()

    env.close()