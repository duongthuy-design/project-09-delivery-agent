
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
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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
from starter.visualization import (
    plot_grid,
    plot_q_heatmap,
    plot_policy_arrows,
)
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
# TRACK TRAINING REWARDS FOR LEARNING CURVE
# ============================================================

class TrackingGridWorld(GridWorld):
    """Record the total reward of each training episode."""

    def __init__(self, *args, **kwargs):
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        super().__init__(*args, **kwargs)

    def reset(self):
        self.current_episode_reward = 0.0
        return super().reset()

    def step(self, action):
        result = super().step(action)

        self.current_episode_reward += float(result.reward)

        if result.done:
            self.episode_rewards.append(
                self.current_episode_reward
            )

        return result


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
        st.session_state.pop("comparison_results", None)

        # A Q-table from another map or reward configuration
        # must not be reused.
        st.session_state.pop(
            "trained_q_table",
            None,
        )
        st.session_state.pop(
            "training_rewards",
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
    st.divider()

    st.header("Simulation Settings")

    animation_delay = st.slider(
        "Robot Speed (seconds per step)",
        min_value=0.05,
        max_value=1.0,
        value=0.30,
        step=0.05,
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
    st.session_state.pop("comparison_results", None)
    # Create a separate environment for training.
    training_env = TrackingGridWorld(
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
    st.session_state["training_rewards"] = (
        training_env.episode_rewards
    )

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
    # The animated run will be handled
    # in the map visualization section.
    pass


# ============================================================
# 11. DISPLAY DELIVERY MAP
# ============================================================

col_grid, col_info = st.columns(
    [2, 1]
)


with col_grid:

    st.subheader("Campus Delivery Map")

    # Placeholder for updating the map.
    map_placeholder = st.empty()

    # Display live animation information.
    progress_placeholder = st.empty()

    # Show the initial map before moving.
    fig = plot_grid(
        env,
        st.session_state["path"],
    )

    map_placeholder.pyplot(
        fig,
        clear_figure=True,
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Animated Q-learning execution
    # --------------------------------------------------------

    if trained_run_clicked:

        while not env.is_terminal():

            # Select the best action from the trained Q-table.
            action = student_core.select_action(
                env.state,
                q_table,
                env.actions,
                epsilon=0.0,
            )

            # Move the robot one step.
            take_step(
                env,
                action,
                "Trained",
            )

            # Draw the updated map.
            fig = plot_grid(
                env,
                st.session_state["path"],
            )
            map_placeholder.pyplot(
                fig,
                clear_figure=True,
            )

            plt.close(fig)

            # Update the live status.
            progress_placeholder.info(
                f"Step: {env.steps_taken}/{env.max_steps} "
                f"| Position: {env.position} "
                f"| Action: {action} "
                f"| Reward: "
                f"{st.session_state['episode_reward']:.1f}"
            )

            # Wait before the next movement.
            time.sleep(animation_delay)

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

    st.code(path_text)

    if len(path) > 80:
        st.caption(
            "Path display is limited to 80 positions."
        )


# ============================================================
# 15. LEARNING CURVE
# ============================================================

st.divider()
st.subheader("Q-learning Training Curve")

training_rewards = st.session_state.get(
    "training_rewards", []
)

if training_rewards:

    reward_df = pd.DataFrame({
        "Episode": range(
            1,
            len(training_rewards) + 1
        ),
        "Episode Reward": training_rewards,
    })

    # Calculate moving average over 50 episodes.
    reward_df["Moving Average (50)"] = (
        reward_df["Episode Reward"]
        .rolling(window=50, min_periods=1)
        .mean()
    )


    # Draw Learning Curve using Matplotlib.
    fig, ax = plt.subplots(figsize=(10, 4))

    # Episode reward.
    ax.plot(
        reward_df["Episode"],
        reward_df["Episode Reward"],
        color="#93C5FD",
        linewidth=1,
        alpha=0.45,
        label="Episode Reward",
    )

    # Moving average.
    ax.plot(
        reward_df["Episode"],
        reward_df["Moving Average (50)"],
        color="#F97316",
        linewidth=2.5,
        label="Moving Average (50)",
    )

    ax.set_title("Q-learning Training Progress")
    ax.set_xlabel("Training Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


    st.caption(
        "Episode Reward: total reward per training episode. "
        "Moving Average (50): smoothed reward trend."
    )

    avg_last_100 = sum(
        training_rewards[-100:]
    ) / min(100, len(training_rewards))

    st.metric(
        "Average Reward (Last 100 Episodes)",
        f"{avg_last_100:.2f}",
    )

    # Download training results as CSV.
    csv_data = reward_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Training Results (CSV)",
        data=csv_data,
        file_name="q_learning_training_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


else:

    st.info(
        "Train the Q-learning agent to display "
        "its learning curve."
    )



# ============================================================
# 16. Q-VALUE HEATMAP
# ============================================================

st.divider()
st.subheader("Q-value Heatmap")

st.caption(
    "Visualize the highest learned Q-value "
    "at each grid position."
)

if trained and q_table:

    heatmap_mode = st.radio(
        "Select Agent State",
        options=[
            "Before Pickup",
            "After Pickup",
        ],
        horizontal=True,
    )

    has_package = (
        heatmap_mode == "After Pickup"
    )

    fig = plot_q_heatmap(
        env=env,
        q_table=q_table,
        has_package=has_package,
    )

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Brighter colors represent higher learned Q-values. "
        "Gray cells have no recorded Q-values. "
        "Dark cells are walls."
    )

else:

    st.info(
        "Train the Q-learning agent first "
        "to display the Q-value Heatmap."
    )

# ============================================================
# 17. POLICY ARROWS VISUALIZATION
# ============================================================

st.divider()
st.subheader("Learned Policy Arrows")

st.caption(
    "Visualize the best learned action "
    "at each grid position."
)

if trained and q_table:

    policy_mode = st.radio(
        "Select Policy State",
        options=[
            "Before Pickup",
            "After Pickup",
        ],
        horizontal=True,
        key="policy_mode",
    )

    has_package_policy = (
        policy_mode == "After Pickup"
    )

    fig = plot_policy_arrows(
        env=env,
        q_table=q_table,
        has_package=has_package_policy,
    )

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "↑ UP | ↓ DOWN | ← LEFT | → RIGHT. "
        "Arrows show the highest Q-value action. "
        "Cells without a learned action preference "
        "have no arrow."
    )

else:

    st.info(
        "Train the Q-learning agent first "
        "to display the learned policy."
    )


# ============================================================
# 18. RANDOM AGENT VS Q-LEARNING COMPARISON
# ============================================================

st.divider()
st.subheader("Random Agent vs. Q-learning Agent")

st.caption(
    "Compare a random agent with the trained Q-learning "
    "agent using the same map and reward configuration."
)


def evaluate_comparison_episode(policy, rng=None):
    """Evaluate one episode without further training."""

    test_env = GridWorld(
        map_name=map_name,
        reward_preset=preset,
    )

    state = test_env.reset()
    total_reward = 0.0
    delivered = False
    danger_hits = 0


    while not test_env.is_terminal():

        if policy == "Random":
            # Use a separate random generator.
            action = rng.choice(tuple(test_env.actions))

        else:
            # Evaluate Q-learning without exploration.
            action = student_core.select_action(
                state,
                q_table,
                test_env.actions,
                epsilon=0.0,
            )

        result = test_env.step(action)

        # Count encounters with dangerous cells.
        if result.info.get("hit_danger", False):
            danger_hits += 1


        total_reward += float(result.reward)

        delivered = (
            delivered
            or bool(result.info.get("delivered", False))
        )

        state = result.state


    return {
        "delivered": delivered,
        "steps": test_env.steps_taken,
        "reward": total_reward,
        "danger_hits": danger_hits,
    }



if trained:

    random_runs = st.number_input(
        "Number of Random Agent evaluation runs",
        min_value=10,
        max_value=100,
        value=30,
        step=10,
    )

    compare_clicked = st.button(
        "Compare Random vs. Q-learning",
        use_container_width=True,
    )

    if compare_clicked:

        # The environment is deterministic.
        # One greedy run represents the current trained policy.
        trained_result = evaluate_comparison_episode(
            "Q-learning"
        )

        # Run multiple independent random episodes.
        random_results = []

        for run_index in range(int(random_runs)):

            rng = random.Random(
                int(seed) + run_index
            )

            run_result = evaluate_comparison_episode(
                "Random",
                rng=rng,
            )

            random_results.append(run_result)

        # Calculate random baseline statistics.
        n = len(random_results)

        random_success = (
            100.0
            * sum(r["delivered"] for r in random_results)
            / n
        )

        random_steps = (
            sum(r["steps"] for r in random_results)
            / n
        )

        random_reward = (
            sum(r["reward"] for r in random_results)
            / n
        )

        # Average danger encounters for Random Agent.
        random_danger_hits = (
            sum(r["danger_hits"] for r in random_results)
            / n
        )


        # Calculate trained policy statistics.
        trained_success = (
            100.0 if trained_result["delivered"] else 0.0
        )

        trained_steps = float(
            trained_result["steps"]
        )

        trained_reward = float(
            trained_result["reward"]
        )

        # Danger encounters for the trained agent.
        trained_danger_hits = float(
            trained_result["danger_hits"]
        )


        # Save results so they remain visible after reruns.
        st.session_state["comparison_results"] = {
            "random_success": random_success,
            "random_steps": random_steps,
            "random_reward": random_reward,
            "trained_success": trained_success,
            "trained_steps": trained_steps,
            "trained_reward": trained_reward,
            "random_danger_hits": random_danger_hits,
            "trained_danger_hits": trained_danger_hits,
            "runs": n,
        }

    comparison = st.session_state.get(
        "comparison_results"
    )

    if comparison:

        st.success("Agent comparison completed!")

        st.caption(
            f"Random: {comparison['runs']} evaluation runs. "
            "Q-learning: one deterministic greedy run."
        )

        # ----------------------------------------------------
        # SUCCESS RATE
        # ----------------------------------------------------

        st.markdown("#### Delivery Success Rate")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Random Agent",
                f"{comparison['random_success']:.1f}%",
            )

        with col2:
            st.metric(
                "Q-learning Agent",
                f"{comparison['trained_success']:.1f}%",
            )

        # Draw a success rate comparison chart.
        fig, ax = plt.subplots(figsize=(7, 4))

        ax.bar(
            ["Random Agent", "Q-learning Agent"],
            [
                comparison["random_success"],
                comparison["trained_success"],
            ],
            color=["#94A3B8", "#2563EB"],
            width=0.5,
        )

        ax.set_ylim(0, 105)
        ax.set_ylabel("Delivery Success Rate (%)")
        ax.set_title("Delivery Success Comparison")
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        # ----------------------------------------------------
        # AVERAGE STEPS
        # ----------------------------------------------------

        st.markdown("#### Average Steps")

        col3, col4 = st.columns(2)

        with col3:
            st.metric(
                "Random Agent",
                f"{comparison['random_steps']:.1f}",
            )

        with col4:
            st.metric(
                "Q-learning Agent",
                f"{comparison['trained_steps']:.1f}",
            )

        # ----------------------------------------------------
        # AVERAGE REWARD
        # ----------------------------------------------------

        st.markdown("#### Average Reward")

        col5, col6 = st.columns(2)

        with col5:
            st.metric(
                "Random Agent",
                f"{comparison['random_reward']:.1f}",
            )

        with col6:
            st.metric(
                "Q-learning Agent",
                f"{comparison['trained_reward']:.1f}",
            )

        # ----------------------------------------------------
        # DANGER HITS COMPARISON
        # ----------------------------------------------------

        st.markdown("#### Average Danger Hits")

        col7, col8 = st.columns(2)

        with col7:
            st.metric(
                "Random Agent",
                f"{comparison['random_danger_hits']:.2f}",
            )

        with col8:
            st.metric(
                "Q-learning Agent",
                f"{comparison['trained_danger_hits']:.2f}",
            )

        st.caption(
            "Lower is better. Random results are averaged "
            "over multiple runs, while Q-learning is "
            "evaluated using one trained greedy policy."
        )


        st.caption(
            "Average Steps includes successful and failed "
            "episodes. Random results are averaged across "
            "multiple runs; Q-learning evaluates the current "
            "trained policy on a deterministic map. "
            "A multi-seed training experiment is needed "
            "to assess Q-learning stability."
        )

else:

    st.info(
        "Train the Q-learning agent first "
        "to enable agent comparison."
    )
