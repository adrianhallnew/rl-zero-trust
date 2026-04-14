"""Neural network architecture definitions for DQN and PPO agents.

All architectures are designed for CPU training on a 16 GB RAM system.
Default hidden layers: [128, 128, 64] with ReLU activations.

DQN:
    Input  -> Dense(128) -> Dense(128) -> Dense(64) -> Dense(num_actions)
    Output: Q-values for each discrete action (linear activation).

PPO (Actor-Critic with shared feature extraction):
    Shared -> Dense(128)
    Actor  -> Dense(128) -> Dense(128) -> Dense(64) -> Gaussian means + log_std
    Critic -> Dense(128) -> Dense(128) -> Dense(64) -> Scalar V(s)
"""

import logging
from typing import List, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger(__name__)


def build_dqn_network(
    state_dim: int,
    num_actions: int,
    hidden_layers: List[int],
    activation: str = "relu",
    learning_rate: float = 0.001,
) -> keras.Model:
    """Build the DQN Q-network.

    Args:
        state_dim: Dimensionality of the observation vector.
        num_actions: Number of discrete actions.
        hidden_layers: List of hidden layer sizes (e.g. [128, 128, 64]).
        activation: Activation function for hidden layers.
        learning_rate: Adam optimiser learning rate.

    Returns:
        Compiled Keras model that maps state -> Q-values.
    """
    inputs = keras.Input(shape=(state_dim,), name="state_input")

    x = inputs
    for i, units in enumerate(hidden_layers):
        x = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_uniform",
            name=f"hidden_{i}",
        )(x)

    outputs = keras.layers.Dense(
        num_actions,
        activation="linear",
        kernel_initializer="he_uniform",
        name="q_values",
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="dqn_network")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    total_params = model.count_params()
    logger.info(
        "DQN network built: %s -> %d actions (%d params)",
        hidden_layers, num_actions, total_params,
    )

    return model


def build_dqn_target_network(online_model: keras.Model) -> keras.Model:
    """Create a target network by cloning the online model's architecture.

    The target network is initialised with the same weights as the online
    model and is updated periodically via soft updates (tau blending).

    Args:
        online_model: The online DQN network to clone.

    Returns:
        A new Keras model with identical architecture and weights.
    """
    target = keras.models.clone_model(online_model)
    target.set_weights(online_model.get_weights())
    target._name = "dqn_target_network"
    logger.info("Target network created (cloned from online)")
    return target


def soft_update_target(
    online_model: keras.Model,
    target_model: keras.Model,
    tau: float = 0.005,
) -> None:
    """Polyak-average (soft) update of target network weights.

    theta_target = tau * theta_online + (1 - tau) * theta_target

    Args:
        online_model: Source (online) network.
        target_model: Destination (target) network.
        tau: Interpolation factor in [0, 1].
    """
    new_weights = []
    for online_w, target_w in zip(
        online_model.get_weights(), target_model.get_weights()
    ):
        new_weights.append(tau * online_w + (1.0 - tau) * target_w)
    target_model.set_weights(new_weights)


# ======================================================================
# PPO Network Builders
# ======================================================================

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def build_ppo_actor_network(
    state_dim: int,
    action_dim: int,
    shared_layers: List[int],
    actor_layers: List[int],
    activation: str = "relu",
) -> keras.Model:
    """Build the PPO actor network that outputs Gaussian distribution params.

    Architecture:
        Input -> Shared layers -> Actor layers -> (means, log_stds)

    The actor outputs a mean and log-standard-deviation for each continuous
    action dimension, parameterising a diagonal Gaussian policy.

    Args:
        state_dim: Observation vector dimensionality.
        action_dim: Number of continuous action dimensions.
        shared_layers: Shared feature extraction layer sizes.
        actor_layers: Actor-specific hidden layer sizes.
        activation: Activation function for hidden layers.

    Returns:
        Keras model mapping state -> (action_means, action_log_stds).
    """
    inputs = keras.Input(shape=(state_dim,), name="state_input")

    # Shared feature extraction
    x = inputs
    for i, units in enumerate(shared_layers):
        x = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_uniform",
            name=f"shared_{i}",
        )(x)

    # Actor-specific layers
    actor = x
    for i, units in enumerate(actor_layers):
        actor = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_uniform",
            name=f"actor_{i}",
        )(actor)

    # Output: action means (tanh to bound [-1, 1]) and log standard deviations
    action_means = keras.layers.Dense(
        action_dim,
        activation="tanh",
        kernel_initializer=keras.initializers.RandomUniform(-0.03, 0.03),
        name="action_means",
    )(actor)

    action_log_stds = keras.layers.Dense(
        action_dim,
        activation="linear",
        kernel_initializer=keras.initializers.Zeros(),
        name="action_log_stds",
    )(actor)

    model = keras.Model(
        inputs=inputs,
        outputs=[action_means, action_log_stds],
        name="ppo_actor",
    )

    total_params = model.count_params()
    logger.info(
        "PPO actor built: shared=%s, actor=%s -> %d actions (%d params)",
        shared_layers, actor_layers, action_dim, total_params,
    )
    return model


def build_ppo_critic_network(
    state_dim: int,
    shared_layers: List[int],
    critic_layers: List[int],
    activation: str = "relu",
    learning_rate: float = 0.0003,
) -> keras.Model:
    """Build the PPO critic (value function) network.

    Architecture:
        Input -> Shared layers -> Critic layers -> Scalar V(s)

    Args:
        state_dim: Observation vector dimensionality.
        shared_layers: Shared feature extraction layer sizes.
        critic_layers: Critic-specific hidden layer sizes.
        activation: Activation function for hidden layers.
        learning_rate: Adam optimiser learning rate.

    Returns:
        Compiled Keras model mapping state -> scalar value V(s).
    """
    inputs = keras.Input(shape=(state_dim,), name="state_input")

    # Shared feature extraction
    x = inputs
    for i, units in enumerate(shared_layers):
        x = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_uniform",
            name=f"shared_{i}",
        )(x)

    # Critic-specific layers
    critic = x
    for i, units in enumerate(critic_layers):
        critic = keras.layers.Dense(
            units,
            activation=activation,
            kernel_initializer="he_uniform",
            name=f"critic_{i}",
        )(critic)

    value = keras.layers.Dense(
        1,
        activation="linear",
        kernel_initializer=keras.initializers.RandomUniform(-0.03, 0.03),
        name="value",
    )(critic)

    model = keras.Model(inputs=inputs, outputs=value, name="ppo_critic")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )

    total_params = model.count_params()
    logger.info(
        "PPO critic built: shared=%s, critic=%s (%d params)",
        shared_layers, critic_layers, total_params,
    )
    return model
