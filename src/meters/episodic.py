import numpy as np

class TrackEpisodes:
    """Class to track episdodic stats.

    Note: Currently only tracking episodic returns
    """
    def __init__(self, num_envs, writer):
        self.returns = np.zeros(num_envs)
        self.writer = writer

    def step(self, global_step, rewards, dones):
        self.returns += rewards
        for i, done in enumerate(dones):
            if done:
                # Print only 1st environment's returns
                if i == 0:
                    print(
                        f"global_step={global_step}, episodic_return={self.returns[i]}"
                    )

                    if self.writer:
                        self.writer.add_scalar(
                            "train/episodic_return",
                            self.returns[i],
                            global_step,
                        )
                    # self.writer.add_scalar(
                    #     "train/episodic_length",
                    #     infos[done_idx]["episode"]["l"].item(),
                    #     global_step,
                    # )

                # Reset the return for the completed
                self.returns[i] = 0