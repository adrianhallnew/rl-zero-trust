"""Neural network architecture definitions for DQN and PPO agents.

All architectures are designed for CPU training on a 16 GB RAM system.
Default hidden layers: [128, 128, 64] with ReLU activations.

DQN:
    Input  -> Dense(128) -> Dense(128) -> Dense(64) -> Dense(num_actions)
    Output: Q-values for each discrete action (linear activation).

PPO (Sprint 5 — stubs provided for forward compatibility):
    Shared -> Dense(128)
    Actor  -> Dense(128) -> Dense(128) -> Dense(64) -> Gaussian params
    Critic -> Dense(128) -> Dense(128) -> Dense(64) -> Scalar V(s)
"""

import logging
from typing import List

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
    for online_w, target_w in zip(
        online_model.get_weights(), target_model.get_weights()
    ):
        target_w[:] = tau * online_w + (1.0 - tau) * target_w

    # set_weights expects a list, but we modified in-place above
    # so we need to explicitly push the updated arrays
    new_weights = []
    for online_w, target_w in zip(
        online_model.get_weights(), target_model.get_weights()
    ):
        new_weights.append(tau * online_w + (1.0 - tau) * target_w)
    target_model.set_weights(new_weights)
