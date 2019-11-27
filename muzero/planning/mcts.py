import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import numpy as np

from muzero.planning.online_planning_algorithm import OnlinePlanningAlgorithm


class Edge:
    def __init__(self, prior):
        self.p = prior
        self.child = None
        self.n = 0
        self.r = 0
        self.q = 0

    def __repr__(self):
        return f"p: {self.p}\nn: {self.n}\nr: {self.r}\nq: {self.q}"


class Node:
    def __init__(self, state, prior, terminal=False):
        self.state = state
        self.edges = dict()
        for action, prob in prior.items():
            self.edges[action] = Edge(prob)
        self.terminal = terminal

    def add_to_graph(self, g):
        edge_labels = dict()
        for a, edge in self.edges.items():
            if edge.child is not None:
                g.add_edge(self.state, edge.child.state)
                edge_labels[(self.state, edge.child.state)] = str(edge)
                new_labels = edge.child.add_to_graph(g)
                for k, v in new_labels.items():
                    if k not in edge_labels:
                        edge_labels[k] = v
        return edge_labels


class MCTS(OnlinePlanningAlgorithm):
    """An implementation of Monte Carlo Tree Search specific to the MuZero case."""
    def __init__(self, model, num_simulations, c1=1.25, c2=20000, temp=1.0):
        self.model = model
        self.num_simulations = num_simulations
        self.c1 = c1
        self.c2 = c2
        self.temp = temp

    def plan(self, state, visualize=False):
        """See base class documentation."""
        prior, _ = self.model.predict(state)
        root = Node(state, prior)
        for sim in range(self.num_simulations):
            self._simulate(root)
        if visualize:
            self._visualize(root)
        return self._get_pi_v(root)

    def _uct_action(self, node):
        n_total = sum(edge.n for edge in node.edges.values())
        best_score, best_a = -np.inf, None
        for a, edge in node.edges.items():
            n_ratio = np.sqrt(n_total) / (1 + edge.n)
            exploration = self.c1 + np.log((n_total + self.c2 + 1) / self.c2)
            score = edge.q + edge.p * n_ratio * exploration
            if score > best_score:
                best_score = score
                best_a = a
        assert best_a is not None
        return best_a, best_score

    def _simulate(self, node):
        # If we've reached a terminal node, return a value of 0.
        if node.terminal:
            return 0

        a, _ = self._uct_action(node)
        if node.edges[a].child is None:
            # First selection of action; execute the model and expand the tree.
            sp, r, t = self.model.transition(node.state, a)
            pi, v = self.model.predict(sp)
            node.edges[a].r = r
            node.edges[a].child = Node(sp, pi, t)
            g = r + self.model.discount * v * (0 if t else 1)
        else:
            # Action previously selected; recurse simulation deterministically.
            g = node.edges[a].r + self.model.discount * self._simulate(node.edges[a].child)

        node.edges[a].q = (node.edges[a].n * node.edges[a].q + g) / (node.edges[a].n + 1)
        node.edges[a].n += 1
        return g

    def _softmax(self, scores):
        max_score = max(v for v in scores.values())
        total = sum(np.exp((v - max_score) / self.temp) for v in scores.values())
        result = dict()
        for k, v in scores.items():
            result[k] = np.exp((v - max_score) / self.temp) / total
        return result

    def _get_pi_v(self, root):
        pi = self._softmax({a: edge.q for (a, edge) in root.edges.items()})
        v = sum(prob * root.edges[a].q for (a, prob) in pi.items())
        return pi, v

    def _visualize(self, root, filepath="/tmp/graph{}.png".format(np.random.randint(1e8))):
        g = nx.Graph()
        edge_labels = root.add_to_graph(g)
        a = to_agraph(g)

        for k, edge_label in edge_labels.items():
            e = a.get_edge(*k)
            e.attr["label"] = edge_label
        a.layout('dot')
        a.draw(filepath)
