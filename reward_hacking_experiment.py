
"""Project 09 — Reward Hacking Experiment.

Compare F1 (bad_reward) and X1 (improved_reward)
on the medium map.

This experiment evaluates whether an agent maximizes
reward without completing the actual delivery task.
"""

import csv
import random
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt

from environment.gridworld import GridWorld
from starter.student_core import select_action


# =====================================================
# 1. EXPERIMENT SETTINGS
# =====================================================

MAP_NAME = "medium"

PRESETS = {
    "F1": "bad_reward",
    "X1": "improved_reward"
}

SEEDS = [11, 22, 33, 44, 55]

EPISODES = 1000
ALPHA = 0.2
GAMMA = 0.95
EPSILON = 0.2

LAST_N = 100
WINDOW = 50

OUTPUT_DIR = (
    Path(__file__).resolve().parent
    / "reward_hacking_results"
)


# =====================================================
# 2. RUN ONE EPISODE
# =====================================================

def run_episode(
    env,
    q_table,
    epsilon,
    alpha=None,
    gamma=None
):
    """Run one episode.

    If alpha and gamma are provided, update the Q-table.
    Otherwise, evaluate the learned policy without learning.
    """

    state = env.reset()

    total_reward = 0.0
    delivered = False

    no_package_goal_visits = 0
    danger_hits = 0

    while not env.is_terminal():

        # Step 1: Choose an action
        action = select_action(
            state=state,
            q_table=q_table,
            actions=env.actions,
            epsilon=epsilon
        )

        # Step 2: Execute the action
        result = env.step(action)

        next_state = result.state
        reward = float(result.reward)
        done = bool(result.done)

        # Step 3: Detect reward-hacking behavior

        # Count steps ending at GOAL without a package
        if (
            next_state[:2] in env.goal_cells
            and not next_state[2]
        ):
            no_package_goal_visits += 1

        # Count danger hits
        if result.info.get("hit_danger", False):
            danger_hits += 1

        # Check whether delivery succeeded
        if result.info.get("delivered", False):
            delivered = True

        # Step 4: Update Q-table during training
        if alpha is not None and gamma is not None:

            old_q = q_table.get(
                (state, action), 0.0
            )

            if done:
                target = reward

            else:
                best_next_q = max(
                    q_table.get(
                        (next_state, next_action), 0.0
                    )
                    for next_action in env.actions
                )

                target = (
                    reward + gamma * best_next_q
                )

            q_table[(state, action)] = (
                old_q + alpha * (target - old_q)
            )

        # Step 5: Accumulate reward
        total_reward += reward

        # Step 6: Move to the next state
        state = next_state

    return {
        "reward": total_reward,
        "delivered": int(delivered),
        "steps": env.steps_taken,
        "no_package_goal_visits": no_package_goal_visits,
        "danger_hits": danger_hits
    }


# =====================================================
# 3. TRAIN AND EVALUATE
# =====================================================

def train_and_evaluate(preset, seed):

    # Set random seed
    random.seed(seed)

    # Create training environment
    train_env = GridWorld(
        map_name=MAP_NAME,
        reward_preset=preset
    )

    # Initialize Q-table
    q_table = {}

    history = []

    # Train for multiple episodes
    for _ in range(EPISODES):

        result = run_episode(
            env=train_env,
            q_table=q_table,
            epsilon=EPSILON,
            alpha=ALPHA,
            gamma=GAMMA
        )

        history.append(result)

    # Create a separate evaluation environment
    eval_env = GridWorld(
        map_name=MAP_NAME,
        reward_preset=preset
    )

    # Evaluate with epsilon = 0
    evaluation = run_episode(
        env=eval_env,
        q_table=q_table,
        epsilon=0.0
    )

    return history, evaluation


# =====================================================
# 4. SAVE CSV FILES
# =====================================================

def save_csv(path, rows):

    if not rows:
        return

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# =====================================================
# 5. MOVING AVERAGE
# =====================================================

def moving_average(values, window):

    return [
        mean(values[i - window + 1:i + 1])
        for i in range(window - 1, len(values))
    ]


# =====================================================
# 6. MAIN EXPERIMENT
# =====================================================

def main():

    # Create output folder
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    seed_rows = []
    summary_rows = []

    training_success_curves = {}

    # Run F1 and X1
    for scenario_id, preset in PRESETS.items():

        histories = []
        evaluations = []

        print(
            f"\n=== {scenario_id}: {preset} ==="
        )

        # Run multiple random seeds
        for seed in SEEDS:

            history, evaluation = train_and_evaluate(
                preset=preset,
                seed=seed
            )

            histories.append(history)
            evaluations.append(evaluation)

            # Last 100 training episodes
            last = history[-LAST_N:]

            seed_rows.append({
                "scenario": scenario_id,
                "preset": preset,
                "seed": seed,

                "train_last100_success_percent":
                    100 * mean(
                        x["delivered"] for x in last
                    ),

                "train_last100_avg_goal_without_package_visits":
                    mean(
                        x["no_package_goal_visits"]
                        for x in last
                    ),

                "train_last100_avg_danger_hits":
                    mean(
                        x["danger_hits"] for x in last
                    ),

                "train_last100_mean_reward":
                    mean(
                        x["reward"] for x in last
                    ),

                "eval_delivered":
                    evaluation["delivered"],

                "eval_steps":
                    evaluation["steps"],

                "eval_reward":
                    evaluation["reward"],

                "eval_goal_without_package_visits":
                    evaluation["no_package_goal_visits"],

                "eval_danger_hits":
                    evaluation["danger_hits"]
            })

            print(
                f"seed={seed}: "
                f"eval delivered="
                f"{bool(evaluation['delivered'])}, "
                f"no-package GOAL visits="
                f"{evaluation['no_package_goal_visits']}, "
                f"danger hits="
                f"{evaluation['danger_hits']}, "
                f"steps={evaluation['steps']}, "
                f"reward={evaluation['reward']:.1f}"
            )

        # =================================================
        # 7. SUMMARIZE RESULTS
        # =================================================

        # Collect last 100 training episodes
        # from all 5 seeds
        final_train = [
            step
            for history in histories
            for step in history[-LAST_N:]
        ]

        # Greedy evaluation success rate
        mean_eval_success = (
            100 * mean(
                x["delivered"] for x in evaluations
            )
        )

        # Average GOAL steps without package
        avg_eval_goal_without_package = mean(
            x["no_package_goal_visits"]
            for x in evaluations
        )

        # Average danger hits
        avg_eval_danger = mean(
            x["danger_hits"]
            for x in evaluations
        )

        # Save summary
        summary_rows.append({

            "scenario": scenario_id,
            "preset": preset,
            "map": MAP_NAME,

            "episodes_per_seed": EPISODES,
            "seeds": len(SEEDS),

            "alpha": ALPHA,
            "gamma": GAMMA,
            "epsilon_train": EPSILON,

            "train_last100_success_percent": round(
                100 * mean(
                    x["delivered"] for x in final_train
                ),
                2
            ),

            "train_last100_avg_goal_without_package_visits":
                round(
                    mean(
                        x["no_package_goal_visits"]
                        for x in final_train
                    ),
                    2
                ),

            "train_last100_avg_danger_hits": round(
                mean(
                    x["danger_hits"] for x in final_train
                ),
                2
            ),

            "eval_success_percent": round(
                mean_eval_success,
                2
            ),

            "eval_avg_goal_without_package_visits": round(
                avg_eval_goal_without_package,
                2
            ),

            "eval_avg_danger_hits": round(
                avg_eval_danger,
                2
            ),

            "eval_mean_reward": round(
                mean(
                    x["reward"] for x in evaluations
                ),
                2
            ),

            "eval_mean_steps": round(
                mean(
                    x["steps"] for x in evaluations
                ),
                2
            )
        })

        # Average delivery success across seeds
        # for every training episode
        training_success_curves[scenario_id] = [
            100 * mean(
                history[i]["delivered"]
                for history in histories
            )
            for i in range(EPISODES)
        ]

        # Print scenario summary
        print(
            f"\n=> {scenario_id} Greedy evaluation:"
        )

        print(
            f"Delivery success: "
            f"{mean_eval_success:.1f}%"
        )

        print(
            f"GOAL without package: "
            f"{avg_eval_goal_without_package:.1f} "
            f"steps/run"
        )

        print(
            f"Danger hits: "
            f"{avg_eval_danger:.1f} hits/run"
        )

    # =====================================================
    # 8. EXPORT CSV RESULTS
    # =====================================================

    save_csv(
        OUTPUT_DIR / "reward_hacking_by_seed.csv",
        seed_rows
    )

    save_csv(
        OUTPUT_DIR / "reward_hacking_summary.csv",
        summary_rows
    )

    # =====================================================
    # 9. PLOT DELIVERY SUCCESS COMPARISON
    # =====================================================

    plt.figure(figsize=(10, 6))

    for scenario_id, curve in training_success_curves.items():

        smoothed = moving_average(
            curve,
            WINDOW
        )

        plt.plot(
            range(WINDOW, EPISODES + 1),
            smoothed,
            label=f"{scenario_id} ({PRESETS[scenario_id]})"
        )

    plt.xlabel("Training Episode")

    plt.ylabel(
        f"Delivery Success Rate (%) "
        f"- {WINDOW}-Episode Moving Average"
    )

    plt.title(
        "Reward Design Experiment: "
        "F1 vs X1 Delivery Success"
    )

    plt.ylim(0, 105)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_delivery_comparison.png",
        dpi=200
    )

    plt.close()

    # =====================================================
    # 10. PLOT REWARD HACKING COMPARISON
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.bar(
        [
            row["scenario"]
            for row in summary_rows
        ],
        [
            row["eval_avg_goal_without_package_visits"]
            for row in summary_rows
        ]
    )

    plt.xlabel("Scenario")

    plt.ylabel(
        "Average Steps at GOAL Without Package"
    )

    plt.title(
        "Reward Hacking Indicator: "
        "F1 vs X1"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "goal_without_package_comparison.png",
        dpi=200
    )

    plt.close()

    # =====================================================
    # 11. PRINT SAVED FILES
    # =====================================================

    print("\nExperiment completed!")

    print("\nSaved results in:")
    print(OUTPUT_DIR)

    print("\nGenerated files:")
    print("- reward_hacking_by_seed.csv")
    print("- reward_hacking_summary.csv")
    print("- training_delivery_comparison.png")
    print("- goal_without_package_comparison.png")

    print(
        "\nNote: Do not directly compare raw rewards "
        "between different reward presets."
    )


# =====================================================
# 12. RUN PROGRAM
# =====================================================

if __name__ == "__main__":
    main()
