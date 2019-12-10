from muzero.envs.maze import Maze
from muzero.datasets.tabular_dataset import TabularDataset
from muzero.loggers import SimpleLogger
from muzero.models.tabular_model import TabularModel
from muzero.planning.mcts import MCTS
from muzero.planning.value_iteration import value_iteration
from muzero.rl.tabular_muzero import TabularMuZero
from muzero.samplers.serial_sampler import SerialSampler
from muzero.trainers.simple_trainer import SimpleTrainer


def run_value_iteration(env):
    _, _, v, pi = value_iteration(env, max_iter=100)
    return v, pi


def run_muzero(env, log_dir, total_steps=50000, steps_per_update=100, num_mcts_simulations=50, return_n=4):
    max_steps = total_steps // steps_per_update
    model = TabularModel(env.observation_space, env.action_space)
    planner = MCTS(model,
                   num_simulations=num_mcts_simulations,
                   discount=env.discount,
                   num_temp_steps=total_steps)
    dataset = TabularDataset(return_n=return_n, discount=env.discount)
    agent = TabularMuZero(model, planner, dataset)
    sampler = SerialSampler(env, max_steps=steps_per_update, max_rollout_steps=50)
    logger = SimpleLogger(log_dir, max_steps, print_env_info_every=20)
    trainer = SimpleTrainer(sampler, agent, logger=logger)
    trainer.train(max_steps)
    pi, v = model.env_pi_v()
    return v, pi


def main():
    env = Maze()
    vi_v, _ = run_value_iteration(env)
    num_experiments = 10
    for i in range(num_experiments):
        log_dir = "../data/muzero_{}".format(i)
        mu_v, _ = run_muzero(env, log_dir)
        env.render_value_function(vi_v, mode="text")
        env.render_value_function(mu_v, mode="text")


if __name__ == "__main__":
    main()
