import numpy as np
import tf_agents.specs

from social_rl.diayn.abstract_diayn import AbstractDiayn
import keras.losses
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

class CEDiayn(AbstractDiayn):
    def  __init__(self, n_skills, n_envs, obs_spec, actor_fc_layers, conv_filters=8, conv_kernel=3, scalar_fc=5, train_on_trajectory=None):

        action_spec = tf_agents.specs.BoundedTensorSpec(shape=[n_skills,], dtype=tf.float32, name="SkillPredictionActionSpec", minimum=-1, maximum=1)

        super().__init__(train_on_trajectory=train_on_trajectory, n_skills=n_skills, n_envs=n_envs)

        from social_rl.multiagent_tfagents import multigrid_networks
        (self.actor_net,
         self.value_net) = multigrid_networks.construct_multigrid_networks(
            obs_spec, action_spec, use_rnns=False,
            actor_fc_layers=actor_fc_layers, value_fc_layers=actor_fc_layers,
            lstm_size=None, conv_filters=conv_filters,
            conv_kernel=conv_kernel, scalar_fc=scalar_fc)

        self.forward_loss = keras.losses.SparseCategoricalCrossentropy(reduction="none", from_logits=False)
        self.backward_loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.03) # todo

        self.z = np.arange(4)   # todo

        self.n_skills = n_skills

    def score_and_augment(self, timestep, as_one_hot=True):
        vals = self.actor_net.predict(timestep)

        score = self.forward_loss(self.z, timestep.observation)
        return timestep


    def loss(self, y_pred, y_true):
        y_pred += 1 # tanh to prob distr
        y_pred /= 2
        loss = self.backward_loss(y_true, y_pred)
        return loss

    def grad(self, obss):
        with tf.GradientTape() as t:
            loss_value = self.loss(self.actor_net(obss), self.z)
        return loss_value, t.gradient(loss_value, self.actor_net.trainable_variables)

    def _train_on_trajectory(self, traj):
        loss_value, grads = self.grad(traj)
        self.optimizer.apply_gradients(zip(grads, self.actor_net.trainable_variables))
        print("Step: {}, Initial Loss: {}".format(self.optimizer.iterations.numpy(), loss_value.numpy()))


    def reset_agent(self):
        return