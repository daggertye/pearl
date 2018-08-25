import agent

class VPG(agent.Agent):
    def __init__(self, env, network):
        self.env = env
        self.network = network

    def train(self, 
            n_iter=100, 
            gamme=1.0, 
            _lambda=1.0,
            min_timsteps_per_batch=1000
            max_path_length,
            learning_rate=5e-3, 
            reward_to_go=True,
            logdir=None, 
            normalize_advantages=True,
            nn_baseline=False):
        raise NotImplementedError

    def run(self):
        raise NotImplementedError