#package imports
import tensorflow as tf
import numpy as np
import gym

#local imports
import pearl.agents.agent as agent

class VPG(agent.Agent):
    """
    A VPG agent. The intuition is to backpropogate onto the policy, resulting in the optimal policy
    where we seek to maximize J = sum(rewards). Paper can be found here
    https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf

    The following heuristics are implemented:

    -reward discount
    -generalized advantage estimate
    -normalized advantage
    -baseline neural network for comparison
    -reward to go
    """
    @staticmethod
    def pathlengh(path):
        return len(path["reward"])

    @staticmethod
    def norm(values, mean, std):
        std_away = (values - np.mean(values))/(np.std(values) + 1e-8)
        return mean + std * std_away

    def __init__(self, env, network_func, sess=None):
        """
        Initializes a vpg agent. 

        Args
        ----
            env (gym.Env) :
                environment to run the agent on, can be discrete or continuous

            network_func (tf.Tensor -> tf.Tensor) :
                function to take an input tensor to the output tensor. Used to build
                the policy's neural network function. Input and output should be built
                for batches (start with [None]).

            sess (tf.Session -- None) :
                session to run the agent on. If None, then just a plain old session
        """
        self.env = env
        self.sess = tf.Session() if sess is None else sess
        self.network_func = network_func
        
        self.input = None
        self.output = None

    def train(self, 
              n_iter=100, 
              gamma=1.0, 
              _lambda=1.0,
              min_timesteps_per_batch=1000,
              max_path_length=None,
              learning_rate=5e-3,
              reward_to_go=True,
              normalize_advantages=True,
              nn_baseline=False,
              seed=0,
              logdir=None):
        """
        Initializes the neural network and trains it.

        Params
        ------
            n_iter (int -- 100) : 
                number of iterations to train the network

            gamma (float -- 1.0) :
                value of gamma (reward discount)

            _lambda (float -- 1.0) :
                value of lambda (for gae)

            min_timesteps_per_batch (int -- 1000) :
                minimum number of timesteps required to batch train

            max_path_length (int -- None) :
                maximum length of path, an integer

            reward_to_go (bool -- True) :
                train the agent on the future rewards

            normalize_advantage (bool -- True) :
                normalize the agent to N(0, 1)

            nn_baseline (bool -- False) :
                use a baseline neural network

            seed (int -- 0) :
                random seed

            logdir (~path (optional) -- None) :
                directory to log reward
        """

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
            sy_logits_na = self.network_func(sy_ob_no, ac_dim, "policy")
            sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1), axis=[1])
            sy_logprob_n = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=sy_ac_na, logits=sy_logits_na)

        else:
            sy_mean = self.network_func(sy_ob_no, ac_dim, "policy")
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
            baseline_prediction = tf.squeeze(self.network_func(sy_ob_no, 1, "nn_baseline"))
            bl_n = tf.placeholder(shape=[None], name='bl_n', dtype=tf.float32)
            bl_loss = tf.nn.l2_loss(baseline_prediction - bl_n)
            baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(bl_loss)

        # writer
        if logdir:
            writer = tf.summary.FileWriter(logdir)
            reward_ph = tf.placeholder(tf.float32, shape=())
            reward_summ = tf.summary.scalar('avg/reward', reward_ph)

        # session
        self.sess.__enter__()
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
                    ac = self.sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
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

            if logdir:
                reward_sum = self.sess.run(reward_summ, feed_dict={reward_ph : np.mean(q_n)})
                writer.add_summary(reward_sum, global_step=itr)

            if nn_baseline:
                # nn baseline
                b_n = VPG.norm(self.sess.run(baseline_prediction, feed_dict={sy_ob_no: ob_no}), np.mean(q_n), np.std(q_n))

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
                _ = self.sess.run(baseline_update_op, feed_dict={bl_n : bl_true, sy_ob_no : ob_no})

            _, after_loss = self.sess.run([update_op, loss],feed_dict = {sy_ob_no : ob_no, sy_ac_na : ac_na, sy_adv_n : adv_n})

            self.input = sy_ob_no
            self.output = sy_sampled_ac

    def run(self):
        """
        Runs the agent in the environment. Renders the graphics. If model is not trained,
        nothing happens.
        """
        if self.output is None or self.input is None:
            return

        ob = self.env.reset()
        steps = 0
        while True:
            self.env.render()
            ac = self.sess.run(self.output, feed_dict={self.input : ob[None]})
            ac = ac[0]
            ob, _ , done, _ = self.env.step(ac)
            if done:
                break

    def reset(self):
        """
        Reset the agent. The agent must be retrained.
        """
        self.input = None
        self.output = None

    def update_network(self, new_network_func):
        """
        Change the agent's network.

        Args
        ----
            new_network_func (tf.Tensor -> tf.Tensor) :
                function to take an input tensor to the output tensor. Used to build
                the policy's neural network function. Input and output should be built
                for batches (start with [None]).
        """
        self.reset()
        self.network_func = new_network_func
