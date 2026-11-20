"""Student AI core — Project 09 (Delivery Agent).

Implement the two functions below. This is the graded AI core: the
environment (`environment/gridworld.py`) only provides deterministic
transitions and rewards; it does not choose actions or learn anything.

Do not:
- Ship a pre-trained/serialized Q-table (no ``qtable.json``, ``.pkl``, etc.).
- Call a hosted LLM to pick actions or to "learn" the policy.
- Hard-code the optimal action sequence for a specific map.

The starter Streamlit app (`starter/app.py`) already demonstrates a random
policy walking the grid. Your job is epsilon-greedy action selection and
the tabular Q-learning update loop.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Tuple

State = Tuple[int, int, bool]
QTable = Dict[Tuple[State, str], float]


def select_action(
    state: State,
    q_table: QTable,
    actions: Iterable[str],
    epsilon: float
) -> str:
    """Choose an action for ``state`` using an epsilon-greedy policy.

    With probability ``epsilon``, return a uniformly random action from
    ``actions`` (exploration). Otherwise, return the action with the
    highest ``q_table[(state, action)]`` value, breaking ties consistently
    (exploitation). Unseen (state, action) pairs should be treated as
    Q-value 0.0.

    Args:
        state: current ``(row, col, has_package)`` state.
        q_table: mapping from ``(state, action)`` to a learned Q-value.
        actions: iterable of legal action names, e.g. ``GridWorld.actions``.
        epsilon: exploration probability in ``[0, 1]``.

    Returns:
        The chosen action name (must be one of ``actions``).
    """
    # Convert actions into a tuple
    action_list = tuple(actions)

    # Validate inputs
    if not action_list:
        raise ValueError("actions must contain at least one action")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1 inclusive")

    # Exploration: choose a random action
    if random.random() < epsilon:
        return random.choice(action_list)

    # Exploitation: choose the action with the highest Q-value
    return max(
        action_list,
        key=lambda action: q_table.get((state, action), 0.0),
    )


def q_learning(
    env,
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
) -> QTable:
    """Train a tabular Q-learning agent on ``env``.

    Run ``episodes`` episodes. Each episode should call ``env.reset()``,
    then repeatedly call ``select_action`` and ``env.step`` until the
    episode is done, applying the standard Q-learning update:

        Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

    Args:
        env: a ``environment.gridworld.GridWorld`` instance.
        episodes: number of training episodes to run.
        alpha: learning rate.
        gamma: discount factor.
        epsilon: exploration probability for the behavior policy.

    Returns:
        The learned Q-table mapping ``(state, action) -> value``. An empty
        or all-zero table is not a valid "trained" result; this function
        must actually run the update loop above.
    """
    # Validate training parameters
    if episodes <= 0:
        raise ValueError("episodes must be greater than 0")

    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be greater than 0 and at most 1")

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between 0 and 1 inclusive")

    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must be between 0 and 1 inclusive")

    # Get available actions from the environment
    actions = tuple(env.actions)

    if not actions:
        raise ValueError("env.actions must contain at least one action")

    # Initialize an empty Q-table
    q_table: QTable = {}

    # Training loop
    for _ in range(episodes):

        # Reset the environment for a new episode
        state = env.reset()
        done = False

        # Run until the episode finishes
        while not done:

            # Step 1: Select an action
            action = select_action(
                state, q_table, actions, epsilon
            )

            # Step 2: Execute the action
            result = env.step(action)

            # Step 3: Get the transition result
            next_state = result.state
            reward = float(result.reward)
            done = bool(result.done)

            # Step 4: Get the old Q-value
            old_value = q_table.get((state, action), 0.0)

            # Step 5: Calculate the target Q-value
            if done:
                target = reward
            else:
                best_next_value = max(
                    q_table.get((next_state, next_action), 0.0)
                    for next_action in actions
                )

                target = reward + gamma * best_next_value

            # Step 6: Update the Q-table
            q_table[(state, action)] = (
                    old_value + alpha * (target - old_value)
            )

            # Step 7: Move to the next state
            state = next_state

    # Return the learned Q-table
    return q_table

