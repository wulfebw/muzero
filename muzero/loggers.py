class SimpleLogger:
    def __init__(self, print_env_info_every=100, visualize=False):
        self.print_env_info_every = print_env_info_every
        self.visualize = visualize

    def log(self, step, max_steps, trajs, info, env):
        print("\nstep: {} \ {}".format(step + 1, max_steps))
        print("avg discounted return: {}".format(info["samples_info"]["avg_discounted_return"]))
        print("planning temp: {}".format(info["planner_info"]["temp"]))
        if (step + 1) % self.print_env_info_every == 0:
            env.render_value_function(info["model_info"]["v"], mode="text")
            if self.visualize:
                env.render_value_function(info["model_info"]["v"], mode="image")
