import abc


class OnlinePlanningAlgorithm(abc.ABC):
    @abc.abstractmethod
    def plan(self, state):
        """Perform planning starting from `state` using the provided model.

        Args:
            state: The initial state from which planning is performed.

        Returns:
            A tuple of the policy to employ from `state` (a distribution over
                the available actions), and the (estimated) value of the state.
        """
        return
