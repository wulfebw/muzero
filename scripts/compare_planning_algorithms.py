from muzero.envs.maze import Maze
from muzero.planning.value_iteration import value_iteration


def run_value_iteration(env):
    _, _, v, pi = value_iteration(env, max_iter=100)
    return v, pi


def main():
    env = Maze()
    v, pi = run_value_iteration(env)
    env.render_value_function(v)
    print(pi)


if __name__ == "__main__":
    main()
