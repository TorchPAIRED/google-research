class AbstractDiayn:
    def __init__(self, train_on_trajectory=None):
        assert train_on_trajectory is not None, "must pass argparse args"

        self.train_on_trajectory = train_on_trajectory
        pass

    def augment(self, timestep):
        # TODO add z to timestep or wtv
        pass

    def score_and_augment(self, timestep, as_one_hot=True):
        raise NotImplemented()
        return augmented_scored_timestep

    def train_on_trajectory(self, traj):
        if self.train_on_trajectory:
            self._train_on_trajectory(self, traj)

    def train_on_single_event(self, traj):
        if not self.train_on_trajectory:
            self._train_on_trajectory(self, traj)

    def _train_on_trajectory(self, traj):
        raise NotImplemented()

    def reset(self):
        raise NotImplemented()

    def reset_agent(self):
        raise NotImplemented()