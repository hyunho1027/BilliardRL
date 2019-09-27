import tensorflow as tf
import numpy as np
from collections import deque
from mlagents.envs import UnityEnvironment
import datetime
from utils import gaussian_likelihood, get_gaes, logit, clip_b4_exp

np.set_printoptions(precision=3)
np.random.seed(1)
tf.set_random_seed(1)

vis_mode = True
s_size = [128, 64, 3] if vis_mode else [8]
a_size = 2

gamma = 0.9
_lambda = 0.95
epoch = 10
lr = 5e-5
ppo_eps = 0.2
n_step = 64

load_model = False
train_mode = True

load_path = ""
save_path = "./summary/ppo/" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_interval = 100


class Net:
    def __init__(self, name):
        with tf.variable_scope(name):

            self.s = tf.placeholder(tf.float32, [None, *s_size])
            self.a = tf.placeholder(tf.float32, [None, a_size])
            
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
                self.fc2 = tf.layers.dense(self.fc1, 128, tf.nn.leaky_relu)
                self.fc3 = tf.layers.dense(self.fc2, 128, tf.nn.leaky_relu)

            else:
                self.fc1 = tf.layers.dense(self.s, 128, tf.nn.leaky_relu)
                self.fc2 = tf.layers.dense(self.fc1, 128, tf.nn.leaky_relu)
                self.fc3 = tf.layers.dense(self.fc2, 128, tf.nn.leaky_relu)

            ## Actor
            self.mu = tf.layers.dense(self.fc3, a_size)
            self.log_std = tf.get_variable("log_std", initializer=-0.5*np.ones(a_size, np.float32))
            self.std = tf.exp(clip_b4_exp(self.log_std))
            self.pi = tf.sigmoid(clip_b4_exp(tf.random_normal(tf.shape(self.mu), self.mu, self.std)))
            self.test_pi = tf.sigmoid(clip_b4_exp(self.mu))
            self.logp = gaussian_likelihood(logit(self.a), self.mu, self.log_std)
            self.logp_pi = gaussian_likelihood(self.pi, self.mu, self.log_std)

            ## Critic
            self.v = tf.squeeze(tf.tanh(clip_b4_exp(tf.layers.dense(self.fc3, 1))), axis=1)

class PPOAgent:
    def __init__(self):
        self.net = Net("Net")

        self.adv = tf.placeholder(tf.float32, [None])
        self.ret = tf.placeholder(tf.float32, [None])
        self.logp_old = tf.placeholder(tf.float32, [None])

        self.ratio = tf.exp(self.net.logp - self.logp_old)
        self.min_adv = tf.where(self.adv > 0, (1.0 + ppo_eps)*self.adv, (1.0 - ppo_eps)*self.adv)
        self.pi_loss = - tf.reduce_mean(tf.minimum(self.ratio*self.adv, self.min_adv))
        self.v_loss = tf.reduce_mean((self.ret - self.net.v)**2)

        self.total_loss = self.pi_loss + self.v_loss
        self.train_model = tf.train.AdamOptimizer(lr).minimize(self.total_loss)

        self.approx_kl = tf.reduce_mean(self.logp_old-self.net.logp)
        self.approx_ent = tf.reduce_mean(- self.net.logp)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.summary, self.merge = self.make_summary()

    def update(self, s, a, target, adv, logp_old):
        v_loss, kl, ent = 0, 0, 0
        for i in range(epoch):
            _, sub_v_loss, approx_kl, approx_ent = \
                 self.sess.run([self.train_model, self.v_loss, self.approx_kl, self.approx_ent],
                                feed_dict={self.net.s: s, self.net.s: s, self.net.a: a, self.ret: target, self.adv: adv, self.logp_old: logp_old})
            v_loss += sub_v_loss
            kl += approx_kl
            ent += approx_ent
        return v_loss, kl, ent

    def get_action(self, s, train_mode):
        if train_mode:
            a, v, logp_pi = self.sess.run([self.net.pi, self.net.v, self.net.logp_pi], feed_dict={self.net.s: s, self.net.s: s})
        else:
            a, v, logp_pi = self.sess.run([self.net.test_pi, self.net.v, self.net.logp_pi], feed_dict={self.net.s: s, self.net.s: s})

        return a, v, logp_pi

    def save_model(self):
        self.saver.save(self.sess, save_path + "/model.ckpt")
    
    def make_summary(self):
        self.summary_v_loss = tf.placeholder(tf.float32)
        self.summary_kl = tf.placeholder(tf.float32)
        self.summary_ent = tf.placeholder(tf.float32)
        self.summary_r = tf.placeholder(tf.float32)
        tf.summary.scalar("value_loss", self.summary_v_loss)
        tf.summary.scalar("kl_divergence", self.summary_kl)
        tf.summary.scalar("entropy", self.summary_ent)
        tf.summary.scalar("reward", self.summary_r)
        return tf.summary.FileWriter(logdir=save_path, graph=self.sess.graph), tf.summary.merge_all()

    def write_summary(self, v_loss, kl, ent, r, rollout):
            self.summary.add_summary(self.sess.run(self.merge,  feed_dict={self.summary_v_loss: v_loss,
                                                                            self.summary_kl: kl,
                                                                            self.summary_ent: ent,
                                                                            self.summary_r: r}), rollout)

if __name__ == '__main__' :
    env_name = "./env/Billiard"
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    agent = PPOAgent()

    rollout = 0
    recent_rs = deque(maxlen=100)
    env_info = env.reset(train_mode=train_mode)[default_brain]
    
    s = np.array(env_info.visual_observations[0])[0,:,:,:] if vis_mode else env_info.vector_observations[0]
    while True:
        rollout += 1
        for step in range(n_step):
            a, v_t, logp_t = agent.get_action([s], train_mode)
            a, v_t, logp_t  =  a[0], v_t[0], logp_t[0]
            env_info = env.step(a)[default_brain]
            r = env_info.rewards[0]
            d = env_info.local_done[0]
            # print(rollout, "Rollout", step+1, "Step:", "\nState:", s, "\nAction:", a, "Reward:", r, "Done:", d)
            recent_rs.append(r)
            if step == 0:
                s_list, a_list, v_list, d_list, r_list, logp_t_list = [s], [a], [v_t], [d], [r], [logp_t]
            else:
                s_list.append(s)
                a_list.append(a)
                v_list.append(v_t)
                d_list.append(d)
                r_list.append(r)
                logp_t_list.append(logp_t)

            s = np.array(env_info.visual_observations[0])[0,:,:,:] if vis_mode else env_info.vector_observations[0]

        a, v_t, logp_t = agent.get_action([s], train_mode)
        v_t = v_t[0]
        v_list.append(v_t)
        next_v_list = np.copy(v_list[1:])
        v_list = v_list[:-1]
        
        adv, target = get_gaes(r_list, d_list, v_list, next_v_list, gamma, _lambda, False)

        v_loss, kl, ent = agent.update(s_list, a_list, target, adv, logp_t_list)

        agent.write_summary(v_loss, kl, ent, np.mean(recent_rs), rollout)
        print(f"{rollout} Rollout / Value Loss : {v_loss:.3f} / KL div : {kl:.3f} / Entropy : {ent:.3f} / Recent Reward : {np.mean(recent_rs):.3f}")

        if train_mode and rollout % save_interval == 0:
            print("Model saved.")
            agent.save_model()

    env.close()
