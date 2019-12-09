import csv
import os


class SimpleLogger:
    def __init__(self, log_dir, max_steps, print_env_info_every=100, visualize=True):
        os.makedirs(log_dir, exist_ok=True)
        assert os.path.exists(log_dir)
        self.log_dir = log_dir
        self.max_steps = max_steps

        self.print_env_info_every = print_env_info_every
        self.visualize = visualize

        self.viz_dir = os.path.join(self.log_dir, "viz")
        os.makedirs(self.viz_dir, exist_ok=True)
        self.stats = ["avg_discounted_return"]
        self.stats_filepath = os.path.join(self.log_dir, "stats.csv")
        self.writer = self._get_stats_writer()
        self.step = 0

    def log(self, step, trajs, info, env):
        self._print_info(step, trajs, info, env)
        self._log_env_info(step, trajs, info, env)
        self.writer.writerow(dict(avg_discounted_return=info["samples_info"]["avg_discounted_return"]))

    def _get_stats_writer(self):
        outfile = open(self.stats_filepath, "w", encoding="utf-8")
        writer = csv.DictWriter(outfile, fieldnames=self.stats)
        writer.writeheader()
        return writer

    def _print_info(self, step, trajs, info, env):
        print("\nstep: {} \ {}".format(step + 1, self.max_steps))
        print("avg discounted return: {}".format(info["samples_info"]["avg_discounted_return"]))
        print("planning temp: {}".format(info["planner_info"]["temp"]))
        if (step + 1) % self.print_env_info_every == 0:
            self.step += 1
            env.render_value_function(info["model_info"]["v"], mode="text")

    def _log_env_info(self, step, trajs, info, env):
        if self.visualize and (step + 1) % self.print_env_info_every == 0:
            filepath = os.path.join(self.viz_dir, "v_{}.png".format(self.step))
            env.render_value_function(info["model_info"]["v"], mode="image", filepath=filepath)
