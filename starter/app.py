
"""Autonomous Campus Delivery Agent - Final Project (Phase 1).

Streamlit application:
- Train a tabular Q-learning agent.
- Save the trained Q-table in session state.
- Run the trained agent on the delivery map.
- Display delivery results and action history.

Run:
    python -m streamlit run starter/app.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Allow importing modules from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from environment.gridworld import (
    GridWorld,
    list_maps,
    list_reward_presets,
)

from starter import student_core
from starter.visualization import plot_grid


# ============================================================
# 1. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Campus Delivery Agent",
    layout="wide",
)

st.title("Autonomous Campus Delivery Agent")

st.caption(
    "Final Project - Tabular Q-learning Delivery Simulation"
)


# ============================================================
# 2. RESET ENVIRONMENT
# ============================================================

def reset_episode(map_name: str, preset: str) -> GridWorld:
    """Reset the environment without deleting the trained Q-table."""

    env = GridWorld(
        map_name=map_name,
        reward_preset=preset,
    )

    st.session_state["env"] = env

    st.session_state["_env_key"] = (
        map_name,
        preset,
    )

    st.session_state["episode_reward"] = 0.0

    st.session_state["path"] = [
        env.position
    ]

    st.session_state["log"] = []

    st.session_state["last_outcome"] = "Ready"

    return env


# ============================================================
# 3. GET CURRENT ENVIRONMENT
# ============================================================

def get_env(map_name: str, preset: str) -> GridWorld:
    """Get the current environment for the selected settings."""

    selected_key = (
        map_name,
        preset,
    )

    if st.session_state.get("_env_key") != selected_key:

        # A Q-table from another map or reward configuration
        # must not be reused.
        st.session_state.pop(
            "trained_q_table",
            None,
        )

        st.session_state.pop(
            "training_details",
            None,
        )

        return reset_episode(
            map_name,
            preset,
        )

    return st.session_state["env"]


# ============================================================
# 4. EXECUTE ONE ACTION
# ============================================================

def take_step(
    env: GridWorld,
    action: str,
    policy_name: str,
) -> None:
    """Execute one action and update the simulation information."""

    if env.is_terminal():
        return

    result = env.step(action)

    # Update total reward.
    st.session_state["episode_reward"] += float(
        result.reward
    )

    # Save the robot's position.
    st.session_state["path"].append(
        env.position
    )

    # Save the action history.
    st.session_state["log"].insert(
        0,
        f"Step {env.steps_taken} | "
        f"Policy: {policy_name} | "
        f"Action: {action} | "
        f"State: {result.state} | "
        f"Reward: {result.reward:+.1f} | "
        f"Info: {result.info}"
    )

    # Update episode status.
    if result.info.get("delivered"):

        st.session_state["last_outcome"] = (
            "Delivery successful"
        )

    elif result.info.get("timed_out"):

        st.session_state["last_outcome"] = (
            "Time limit reached"
        )

    elif result.done:

        st.session_state["last_outcome"] = (
            "Episode ended"
        )

    else:

        st.session_state["last_outcome"] = (
            "In progress"
        )


# ============================================================
# 5. SIDEBAR - ENVIRONMENT SETTINGS
# ============================================================

with st.sidebar:

    st.header("Environment Settings")

    map_names = list_maps()

    reward_presets = list_reward_presets()

    map_name = st.selectbox(
        "Select Map",
        map_names,
        index=(
            map_names.index("easy")
            if "easy" in map_names
            else 0
        ),
    )

    preset = st.selectbox(
        "Reward Preset",
        reward_presets,
        index=(
            reward_presets.index("default")
            if "default" in reward_presets
            else 0
        ),
    )

    st.caption(
        "Reward configurations are loaded from "
        "environment/config.json."
    )

    st.divider()

    # --------------------------------------------------------
    # Q-learning hyperparameters
    # --------------------------------------------------------

    st.header("Q-learning Training")

    episodes = st.number_input(
        "Training Episodes",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100,
    )

    alpha = st.slider(
        "Alpha - Learning Rate",
        min_value=0.01,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )

    gamma = st.slider(
        "Gamma - Discount Factor",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.01,
    )

    epsilon = st.slider(
        "Epsilon - Exploration Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.20,
        step=0.01,
    )

    seed = st.number_input(
        "Random Seed",
        min_value=0,
        max_value=999999,
        value=42,
        step=1,
    )

    train_clicked = st.button(
        "Train Q-learning",
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 6. INITIALIZE ENVIRONMENT
# ============================================================

env = get_env(
    map_name,
    preset,
)


# ============================================================
# 7. TRAIN Q-LEARNING
# ============================================================

if train_clicked:

    # Create a separate environment for training.
    training_env = GridWorld(
        map_name=map_name,
        reward_preset=preset,
    )

    # Save the current random state.
    previous_random_state = random.getstate()

    try:

        random.seed(
            int(seed)
        )

        with st.spinner(
            "Training Q-learning agent..."
        ):

            # Call the original student Q-learning function.
            q_table = student_core.q_learning(
                training_env,
                episodes=int(episodes),
                alpha=float(alpha),
                gamma=float(gamma),
                epsilon=float(epsilon),
            )

    finally:

        # Restore the random state after training.
        random.setstate(
            previous_random_state
        )

    # --------------------------------------------------------
    # Save the trained Q-table
    # --------------------------------------------------------

    st.session_state["trained_q_table"] = q_table

    st.session_state["training_details"] = (
        f"Map: {map_name} | "
        f"Reward: {preset} | "
        f"Episodes: {int(episodes)} | "
        f"Alpha: {alpha:.2f} | "
        f"Gamma: {gamma:.2f} | "
        f"Epsilon: {epsilon:.2f} | "
        f"Seed: {int(seed)}"
    )

    # Reset the robot to START after training.
    env = reset_episode(
        map_name,
        preset,
    )


# ============================================================
# 8. CHECK TRAINED POLICY
# ============================================================

q_table = st.session_state.get(
    "trained_q_table"
)

trained = q_table is not None

if trained:

    st.success(
        f"Q-learning training completed! "
        f"Q-table entries: {len(q_table)}"
    )

    st.caption(
        st.session_state["training_details"]
    )

    st.info(
        "The trained policy is ready. "
        "Click 'Run Trained Agent' to start delivery."
    )

else:

    st.info(
        "Train Q-learning first to enable "
        "the trained agent."
    )


# ============================================================
# 9. AGENT CONTROL BUTTONS
# ============================================================

st.subheader("Agent Controls")

col_reset, col_random, col_step, col_run = st.columns(4)

with col_reset:

    reset_clicked = st.button(
        "Reset Episode",
        use_container_width=True,
    )

with col_random:

    random_clicked = st.button(
        "Step Random",
        disabled=env.is_terminal(),
        use_container_width=True,
    )

with col_step:

    trained_step_clicked = st.button(
        "Step Trained",
        disabled=(
            not trained
            or env.is_terminal()
        ),
        use_container_width=True,
    )

with col_run:

    trained_run_clicked = st.button(
        "Run Trained Agent",
        disabled=(
            not trained
            or env.is_terminal()
        ),
        type="primary",
        use_container_width=True,
    )


# ============================================================
# 10. HANDLE USER ACTIONS
# ============================================================

if reset_clicked:

    # Reset the episode but keep the trained Q-table.
    env = reset_episode(
        map_name,
        preset,
    )


elif random_clicked:

    # Random baseline action.
    action = random.choice(
        tuple(env.actions)
    )

    take_step(
        env,
        action,
        "Random",
    )


elif trained_step_clicked:

    # Use the learned Q-table.
    # epsilon=0 means no random exploration.
    action = student_core.select_action(
        env.state,
        q_table,
        env.actions,
        epsilon=0.0,
    )

    take_step(
        env,
        action,
        "Trained",
    )


elif trained_run_clicked:

    # Execute the learned policy until the episode ends.
    # GridWorld stops the episode at its configured step limit.
    while not env.is_terminal():

        action = student_core.select_action(
            env.state,
            q_table,
            env.actions,
            epsilon=0.0,
        )

        take_step(
            env,
            action,
            "Trained",
        )


# ============================================================
# 11. DISPLAY DELIVERY MAP
# ============================================================

col_grid, col_info = st.columns(
    [2, 1]
)

with col_grid:

    st.subheader("Campus Delivery Map")

    fig = plot_grid(env)

    st.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    st.caption(
        "S = START | "
        "P = PICKUP | "
        "G = GOAL | "
        "D = DANGER | "
        "@ = ROBOT | "
        "Dark cells = WALL"
    )


# ============================================================
# 12. DISPLAY AGENT STATUS
# ============================================================

with col_info:

    st.subheader("Agent Status")

    row, col, has_package = env.state

    st.write(
        f"**Position:** ({row}, {col})"
    )

    st.write(
        "**Package Picked Up:** "
        + ("Yes" if has_package else "No")
    )

    st.metric(
        "Total Steps",
        f"{env.steps_taken} / {env.max_steps}",
    )

    st.metric(
        "Total Reward",
        f"{st.session_state['episode_reward']:.1f}",
    )

    st.write(
        "**Episode Status:** "
        + st.session_state["last_outcome"]
    )

    # Display the final episode result.
    if env.is_terminal():

        if (
            st.session_state["last_outcome"]
            == "Delivery successful"
        ):

            st.success(
                "SUCCESS: The agent delivered the package!"
            )

        else:

            st.warning(
                "The episode finished without "
                "a successful delivery."
            )


# ============================================================
# 13. ACTION HISTORY
# ============================================================

st.subheader("Action History")

if st.session_state["log"]:

    with st.expander(
        "Show Recent Actions",
        expanded=True,
    ):

        for line in st.session_state["log"][:20]:

            st.text(line)

        if len(st.session_state["log"]) > 20:

            st.caption(
                "Showing the 20 most recent actions."
            )

else:

    st.caption(
        "No actions yet. Train and run the agent "
        "to see its actions."
    )


# ============================================================
# 14. ROBOT PATH
# ============================================================

with st.expander("Robot Delivery Path"):

    path = st.session_state["path"]

    path_text = " -> ".join(
        str(position)
        for position in path[:80]
    )

    st.code(
        path_text
    )

    if len(path) > 80:

        st.caption(
            "Path display is limited to 80 positions."
        )
