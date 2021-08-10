import numpy as np
import tensorflow as tf

class AbstractDiayn:
    def __init__(self, n_skills, n_envs, train_on_trajectory=None):
        assert train_on_trajectory is not None, "must pass argparse args"

        self.train_on_trajectory = train_on_trajectory
        self.n_skills = n_skills
        self.n_envs = n_envs

        pass

    def augment(self, timestep, score=None):
        # TODO add z to timestep or wtv
        obs = timestep.observation
        image = obs["image"]
        direction = obs["direction"]

        augmented_image = image.numpy() # tensorflow is an atrocity of a framework
        for i in range(self.n_envs):
            augmented_image[i,:,:,2] = self.skills[i]
        from tensorflow.python.framework.ops import EagerTensor
        augmented_image = tf.convert_to_tensor(augmented_image)

        from collections import OrderedDict
        augmented_obs = OrderedDict([("direction", direction), ("image",augmented_image)])

        if score is not None:
            score = tf.convert_to_tensor(score)
        else:
            score = timestep.reward

        import tf_agents
        augmented_timestep = tf_agents.trajectories.time_step.TimeStep(timestep.step_type, score, timestep.discount, augmented_obs)
        return augmented_timestep

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

    def _reset_skills(self):
        self.skills = np.arange(self.n_skills)
        np.random.shuffle(self.skills)
        self.skills = self.skills[:self.n_envs]

    def reset(self, obs):
        self._reset_skills()
        return self.augment(obs)    # NOTE: we don't score here!

    def reset_agent(self):
        raise NotImplemented()