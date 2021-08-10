from social_rl.diayn.abstract_diayn import AbstractDiayn


class NoDiayn(AbstractDiayn):
    def __init__(self):
        super().__init__()

    def score_and_augment(self, timestep, as_one_hot=True):
        return timestep

    def _train_on_trajectory(self, traj):
        return

    def reset(self):
        return

    def reset_agent(self):
        return