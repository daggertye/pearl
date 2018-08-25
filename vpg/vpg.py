#package imports
import tensorflow as tf
import numpy as np
import gym
import time

#local imports
import agent

class VPG(agent.Agent):
    @staticmethod
    def pathlengh(path):
        return len(path["reward"])

    @staticmethod
    def norm(values, mean, std):
        std_away = (values - np.mean(values))/(np.std(values) + 1e-8)
        return mean + std * std_away

    def __init__(self, env, network):
        self.env = env
        self.network = network

    def train(self, 
              n_iter=100, 
              gamma=1.0, 
              _lambda=1.0,
              min_timesteps_per_batch=1000,
              max_path_length=None,
              learning_rate=5e-3,
              reward_to_go=True,
              logdir=None,
              normalize_advantages=True,
              nn_baseline=False,
              seed=0):

        #Set random seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        #discrete/continuous space
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)

        # max path len for episodes
        max_path_length = max_path_length or self.env.spec.max_episode_steps

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]

        # placeholders for observations and actions
        sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
        if discrete:
            sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32)
        else:
            sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)

        # advantage placeholder
        sy_adv_n = tf.placeholder(shape=[None], name="adv", dtype=tf.float32)

        # model
        if discrete:
            sy_logits_na = self.network(sy_ob_no, ac_dim, "policy")
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=[1])
            sy_logprob_n = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)

        else:
            sy_mean = self.network(sy_ob_no, ac_dim, "policy")
            sy_logstd = tf.Variable(tf.zeros([1, ac_dim], name='logstd'))
            sy_std = tf.exp(sy_logstd)
            sy_z_sampled = tf.random_normal(tf.shape(sy_mean))
            sy_sampled_ac = sy_mean + sy_std * sy_z_sampled

            sy_z = (sy_ac_na - sy_mean)/sy_std
            sy_logprob_n = - 0.5 * tf.reduce_sum(tf.square(sy_z), axis=1)

        # loss and update op
        loss = -tf.reduce_mean(tf.multiply(sy_logprob_n, sy_adv_n))
        update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        # baseline nn creation
        if nn_baseline:
            baseline_prediction = tf.squeeze(self.network(sy_ob_no, 1, "nn_baseline"))
            bl_n = tf.placeholder(shape=[None], name='bl_n', dtype=tf.float32)
            bl_loss = tf.nn.l2_loss(baseline_prediction - bl_n)
            baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(bl_loss)

        # session
        sess = tf.Session()
        sess.__enter__()
        tf.global_variables_initializer().run()

        # training
        total_timesteps = 0

        for itr in range(n_iter):
            print("********** Iteration %i ************"%itr)

            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            paths = []
            while True:
                ob = self.env.reset()
                obs, acs, rewards = [], [], []
                steps = 0
                while True:
                    obs.append(ob)
                    ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                    ac = ac[0]
                    acs.append(ac)
                    ob, rew, done, _ = self.env.step(ac)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > max_path_length:
                        break
                path={"observation" : np.array(obs), 
                        "reward" : np.array(rewards),
                        "action" : np.array(acs)}
                paths.append(path)
                timesteps_this_batch += VPG.pathlengh(path)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch

            # observation and action arrays
            ob_no = np.concatenate([path["observation"] for path in paths])
            ac_na = np.concatenate([path["action"] for path in paths])

            #reward to go
            if reward_to_go:
                q_n = []
                for path in paths:
                    q = np.zeros(VPG.pathlengh(path))
                    q[-1] = path['reward'][-1]
                    for i in reversed(range(VPG.pathlengh(path) - 1)):
                        q[i] = path['reward'][i] + gamma * q[i+1]
                    q_n.extend(q)
            else:
                q_n = []
                for path in paths:
                    ret_tau = 0
                    for i in range(VPG.pathlengh(path)):
                        ret_tau += (gamma ** i) * path['reward'][i]
                    q = np.ones(shape=[VPG.pathlengh(path)]) * ret_tau
                    q_n.extend(q)

            if nn_baseline:
                # nn baseline
                b_n = VPG.norm(sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no}), np.mean(q_n), np.std(q_n))

                # Implementation of GAE
                adv_n = []
                for path in paths:
                    adv = np.zeros(VPG.pathlengh(path))
                    adv[-1] = path['reward'][-1] - b_n[-1]
                    for i in reversed(range(VPG.pathlengh(path) - 1)):
                        delta = path['reward'][i] + gamma * b_n[i + 1] - b_n[i]
                        adv[i] = delta + gamma * _lambda * adv[i+1]
                    if not reward_to_go:
                        adv = np.ones(size=[VPG.pathlengh(path)]) * adv[0]
                    adv_n.extend(adv)
                q_n = adv_n + b_n
            else:
                adv_n = q_n.copy()

            # normalize advantages (heuristic that helps)
            if normalize_advantages:

                adv_n = VPG.norm(adv_n, 0, 1)

            # nn baseline training
            if nn_baseline:
                bl_true = VPG.norm(q_n, 0, 1)
                _ = sess.run(baseline_update_op, feed_dict={bl_n : bl_true, sy_ob_no : ob_no})

    def run(self):
        raise NotImplementedError
