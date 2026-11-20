
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
REWARD_PRESET = "default"

EPISODES = 1000
SEEDS = [11, 22, 33, 44, 55]

EPSILON = 0.2
WINDOW = 20
LAST_N = 100

CONFIGS = [
    {"alpha": 0.2, "gamma": 0.95},  # Baseline
    {"alpha": 0.1, "gamma": 0.95},
    {"alpha": 0.5, "gamma": 0.95},
    {"alpha": 0.2, "gamma": 0.80},
    {"alpha": 0.2, "gamma": 0.99},
]

OUTPUT_DIR = Path(__file__).resolve().parent / "experiment_results"


# =====================================================
# 2. TRAIN ONE RUN
# =====================================================

def train_one_run(alpha, gamma, seed):

    random.seed(seed)

    env = GridWorld(
        map_name=MAP_NAME,
        reward_preset=REWARD_PRESET
    )

    actions = tuple(env.actions)
    q_table = {}

    rewards_per_episode = []
    deliveries_per_episode = []

    for _ in range(EPISODES):

        state = env.reset()
        done = False

        total_reward = 0.0
        delivered = False

        while not done:

            # Step 1: Select action
            action = select_action(
                state=state,
                q_table=q_table,
                actions=actions,
                epsilon=EPSILON
            )

            # Step 2: Execute action
            result = env.step(action)

            next_state = result.state
            reward = float(result.reward)
            done = bool(result.done)

            delivered = delivered or bool(
                result.info.get("delivered", False)
            )

            # Step 3: Get old Q-value
            old_q = q_table.get((state, action), 0.0)

            # Step 4: Calculate target
            if done:
                target = reward
            else:
                best_next_q = max(
                    q_table.get(
                        (next_state, next_action), 0.0
                    )
                    for next_action in actions
                )

                target = reward + gamma * best_next_q

            # Step 5: Update Q-table
            q_table[(state, action)] = (
                old_q + alpha * (target - old_q)
            )

            # Step 6: Accumulate reward
            total_reward += reward
            state = next_state

        # Save results of this episode
        rewards_per_episode.append(total_reward)
        deliveries_per_episode.append(delivered)

    return (
        q_table,
        rewards_per_episode,
        deliveries_per_episode
    )


# =====================================================
# 3. EVALUATE THE TRAINED AGENT
# =====================================================

def evaluate_greedy(q_table):

    env = GridWorld(
        map_name=MAP_NAME,
        reward_preset=REWARD_PRESET
    )

    state = env.reset()

    total_reward = 0.0
    delivered = False

    while not env.is_terminal():

        # Evaluate without random exploration
        action = select_action(
            state=state,
            q_table=q_table,
            actions=env.actions,
            epsilon=0.0
        )

        result = env.step(action)

        total_reward += float(result.reward)

        delivered = delivered or bool(
            result.info.get("delivered", False)
        )

        state = result.state

    return {
        "delivered": delivered,
        "reward": total_reward,
        "steps": env.steps_taken
    }


# =====================================================
# 4. MOVING AVERAGE
# =====================================================

def moving_average(values, window):

    return [
        mean(values[i - window + 1:i + 1])
        for i in range(window - 1, len(values))
    ]


# =====================================================
# 5. RUN ALL EXPERIMENTS
# =====================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_curves = {}
    summary_rows = []

    for config in CONFIGS:

        alpha = config["alpha"]
        gamma = config["gamma"]

        label = f"alpha={alpha}, gamma={gamma}"

        seed_rewards = []
        seed_deliveries = []
        eval_results = []

        print(f"\nRunning {label} ...")

        for seed in SEEDS:

            q_table, rewards, deliveries = train_one_run(
                alpha, gamma, seed
            )

            seed_rewards.append(rewards)
            seed_deliveries.append(deliveries)

            evaluation = evaluate_greedy(q_table)
            eval_results.append(evaluation)

        # Average reward across 5 seeds
        average_curve = [
            mean(
                run[episode]
                for run in seed_rewards
            )
            for episode in range(EPISODES)
        ]

        all_curves[label] = average_curve

        # Results from last 100 training episodes
        last_rewards = [
            reward
            for run in seed_rewards
            for reward in run[-LAST_N:]
        ]

        last_deliveries = [
            delivered
            for run in seed_deliveries
            for delivered in run[-LAST_N:]
        ]

        # Greedy evaluation results
        greedy_successes = [
            result
            for result in eval_results
            if result["delivered"]
        ]

        row = {
            "alpha": alpha,
            "gamma": gamma,
            "epsilon_train": EPSILON,
            "map": MAP_NAME,
            "reward_preset": REWARD_PRESET,
            "seeds": len(SEEDS),
            "episodes_per_seed": EPISODES,

            "train_mean_reward_last_100": round(
                mean(last_rewards), 3
            ),

            "train_delivery_rate_last_100_percent": round(
                100 * sum(last_deliveries) / len(last_deliveries),
                2
            ),

            "greedy_eval_success_rate_percent": round(
                100 * len(greedy_successes) / len(SEEDS),
                2
            ),

            "greedy_eval_mean_reward": round(
                mean(r["reward"] for r in eval_results),
                3
            ),

            "greedy_eval_mean_steps_if_success": (
                round(
                    mean(r["steps"] for r in greedy_successes),
                    2
                )
                if greedy_successes
                else ""
            )
        }

        summary_rows.append(row)

        print(
            f"Training last {LAST_N}: "
            f"reward={row['train_mean_reward_last_100']:.2f}, "
            f"delivery="
            f"{row['train_delivery_rate_last_100_percent']:.1f}%"
        )

        print(
            f"Greedy evaluation: "
            f"success="
            f"{row['greedy_eval_success_rate_percent']:.1f}%, "
            f"reward={row['greedy_eval_mean_reward']:.2f}"
        )

    # =================================================
    # 6. SAVE CSV FILES
    # =================================================

    rewards_path = OUTPUT_DIR / "episode_rewards.csv"

    with rewards_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "episode",
            *all_curves.keys()
        ])

        for ep in range(EPISODES):

            writer.writerow([
                ep + 1,
                *(
                    curve[ep]
                    for curve in all_curves.values()
                )
            ])

    summary_path = OUTPUT_DIR / "experiment_summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(summary_rows[0])
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    # =================================================
    # 7. PLOT LEARNING CURVES
    # =================================================

    plt.figure(figsize=(11, 6))

    for label, curve in all_curves.items():

        smoothed = moving_average(curve, WINDOW)

        plt.plot(
            range(WINDOW, EPISODES + 1),
            smoothed,
            label=label
        )

    plt.xlabel("Episode")
    plt.ylabel(
        f"Mean training reward "
        f"({WINDOW}-episode moving average)"
    )

    plt.title(
        f"Q-learning on {MAP_NAME} | "
        f"Alpha/Gamma comparison ({len(SEEDS)} seeds)"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    chart_path = OUTPUT_DIR / "reward_comparison.png"

    plt.savefig(chart_path, dpi=200)
    plt.show()

    # =================================================
    # 8. PRINT OUTPUT LOCATIONS
    # =================================================

    print("\nSaved results:")
    print(f"- {rewards_path}")
    print(f"- {summary_path}")
    print(f"- {chart_path}")


if __name__ == "__main__":
    main()
