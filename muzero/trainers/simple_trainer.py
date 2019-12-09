from muzero.loggers import SimpleLogger


class SimpleTrainer:
    """Orchestrates training of an RL algorithm.

    This trainer is "simple" in that it doesn't manager distributed sampling.
    """

    def __init__(self, sampler, agent, logger=SimpleLogger()):
        """
        Args:
            sampler: A class that samples trajectories from an environment.
            agent: The agent that interacts with the environment and learns from experience.
            logger: Class for logging information about the training.
        """
        self.sampler = sampler
        self.agent = agent
        self.logger = logger

    def train(self, max_steps):
        for step in range(max_steps):
            trajs = self.sampler.sample(self.agent)
            info = self.agent.learn(trajs)
            self.logger.log(step, max_steps, trajs, info, self.sampler.env)
