from social_rl.diayn.abstract_diayn import AbstractDiayn


class NoDiayn(AbstractDiayn):
    def __init__(self, n_skills, n_envs, train_on_trajectory=None):
        super().__init__(n_skills, n_envs, train_on_trajectory=train_on_trajectory)

    def score_and_augment(self, timestep, as_one_hot=True):
        return timestep

    def _train_on_trajectory(self, traj):
        return

    # override
    def reset(self, obs):
        return obs

    def reset_agent(self):
        return