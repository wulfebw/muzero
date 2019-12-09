from muzero.utils import categorical_sample


class TabularMuZero:
    """A tabular implementation of the MuZero algorithm.

    Nothing about this implementation assumes a tabular representation,
    but the version that uses function approximation will have a different
    form because (a) parameter updates will be performed through backprop
    by exposing the parameters of the model and (b) sampling in the environment
    should be performed in a parallel manner, and that's not accounted for here.
    """

    def __init__(self, model, planner, dataset):
        """
        Args:
            model: A MuZero model, which is a class implementing the three functions
                described in the paper (represent, transition, predict), and in the
                tabular case implementing an `update` function for learning.
            planner: The planning algorithm to run on the (learned) model. This
                is an implementation of MCTS in the paper, but could be any online
                planning algorithm in theory.
            dataset: This is used to convert the trajectories of the agent in the
                environment into a useful format for learning.
        """
        self.model = model
        self.planner = planner
        self.dataset = dataset

    def step(self, x):
        pi, v = self.planner.plan(x)
        return categorical_sample(pi), dict(v=v)

    def learn(self, trajs):
        samples, samples_info = self.dataset.trajectories_to_samples(trajs)
        model_info = self.model.update(samples)
        return dict(samples_info=samples_info, model_info=model_info, planner_info=self.planner.info())
