
import csv
import json
import random
import matplotlib.pyplot as plt
from pathlib import Path
from statistics import mean

from environment.gridworld import GridWorld
from starter.student_core import q_learning, select_action


# ============================================================
# 1. EXPERIMENT SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

with open(ROOT / "data" / "scenarios.json", encoding="utf-8") as f:
    SCENARIOS = json.load(f)["scenarios"]

SEEDS = [11, 22, 33, 44, 55]
RANDOM_RUNS = 30

ALPHA = 0.20
GAMMA = 0.95
EPSILON = 0.20

OUTPUT_DIR = ROOT / "scenario_results"


# ============================================================
# 2. EVALUATE ONE EPISODE
# ============================================================

def evaluate(map_name, preset, policy, q_table=None, rng=None):

    env = GridWorld(
        map_name=map_name,
        reward_preset=preset,
    )

    state = env.reset()

    total_reward = 0.0
    delivered = False
    danger_hits = 0
    goal_without_package = 0

    while not env.is_terminal():

        if policy == "Random":
            action = rng.choice(tuple(env.actions))

        else:
            action = select_action(
                state,
                q_table,
                env.actions,
                epsilon=0.0,
            )

        result = env.step(action)

        total_reward += float(result.reward)

        delivered = delivered or bool(
            result.info.get("delivered", False)
        )

        if result.info.get("hit_danger", False):
            danger_hits += 1

        if (
            result.state[:2] in env.goal_cells
            and not result.state[2]
        ):
            goal_without_package += 1

        state = result.state

    return {
        "success": int(delivered),
        "steps": env.steps_taken,
        "reward": round(total_reward, 2),
        "danger_hits": danger_hits,
        "goal_without_package": goal_without_package,
    }


# ============================================================
# 3. SAVE CSV
# ============================================================

def save_csv(path, rows):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 4. RUN ALL SCENARIOS
# ============================================================

def main():

    OUTPUT_DIR.mkdir(exist_ok=True)

    details = []
    summary = []

    for scenario in SCENARIOS:

        scenario_id = scenario["id"]
        map_name = scenario["map"]
        preset = scenario["reward_preset"]

        episodes = 2000 if map_name == "hard" else 1000

        results = {
            "Random": [],
            "Q-learning": [],
        }

        print(
            f"\n=== {scenario_id}: "
            f"{map_name} / {preset} ==="
        )

        # ----------------------------------------------------
        # Train Q-learning across five seeds
        # ----------------------------------------------------

        for seed in SEEDS:

            random.seed(seed)

            train_env = GridWorld(
                map_name=map_name,
                reward_preset=preset,
            )

            q_table = q_learning(
                train_env,
                episodes=episodes,
                alpha=ALPHA,
                gamma=GAMMA,
                epsilon=EPSILON,
            )

            result = evaluate(
                map_name,
                preset,
                "Q-learning",
                q_table=q_table,
            )

            results["Q-learning"].append(result)

            details.append({
                "scenario": scenario_id,
                "agent": "Q-learning",
                "seed": seed,
                **result,
            })

            print(
                f"seed={seed} | "
                f"delivered={bool(result['success'])} | "
                f"steps={result['steps']} | "
                f"danger_hits={result['danger_hits']}"
            )

        # ----------------------------------------------------
        # Random baseline - 30 evaluation runs
        # ----------------------------------------------------

        for i in range(RANDOM_RUNS):

            rng = random.Random(1000 + i)

            result = evaluate(
                map_name,
                preset,
                "Random",
                rng=rng,
            )

            results["Random"].append(result)

            details.append({
                "scenario": scenario_id,
                "agent": "Random",
                "seed": 1000 + i,
                **result,
            })

        # ----------------------------------------------------
        # Calculate summary statistics
        # ----------------------------------------------------

        for agent, agent_results in results.items():

            summary_row = {
                "scenario": scenario_id,
                "map": map_name,
                "reward_preset": preset,
                "agent": agent,
                "runs": len(agent_results),
                "training_episodes": (
                    episodes if agent == "Q-learning" else 0
                ),
                "success_rate_percent": round(
                    100 * mean(
                        r["success"] for r in agent_results
                    ),
                    2,
                ),
                "avg_steps": round(
                    mean(
                        r["steps"] for r in agent_results
                    ),
                    2,
                ),
                "avg_reward": round(
                    mean(
                        r["reward"] for r in agent_results
                    ),
                    2,
                ),
                "avg_danger_hits": round(
                    mean(
                        r["danger_hits"] for r in agent_results
                    ),
                    2,
                ),
                "avg_goal_without_package": round(
                    mean(
                        r["goal_without_package"]
                        for r in agent_results
                    ),
                    2,
                ),
            }

            summary.append(summary_row)

            print(
                f"{agent}: "
                f"success={summary_row['success_rate_percent']}% | "
                f"steps={summary_row['avg_steps']} | "
                f"danger={summary_row['avg_danger_hits']}"
            )

    # --------------------------------------------------------
    # Export results
    # --------------------------------------------------------

    save_csv(
        OUTPUT_DIR / "scenario_details.csv",
        details,
    )

    save_csv(
        OUTPUT_DIR / "scenario_summary.csv",
        summary,
    )

    # ========================================================
    # SCENARIO SUCCESS RATE COMPARISON
    # ========================================================

    scenario_ids = [s["id"] for s in SCENARIOS]

    random_rates = []
    qlearning_rates = []

    for scenario_id in scenario_ids:
        for row in summary:
            if row["scenario"] == scenario_id:
                if row["agent"] == "Random":
                    random_rates.append(
                        row["success_rate_percent"]
                    )
                elif row["agent"] == "Q-learning":
                    qlearning_rates.append(
                        row["success_rate_percent"]
                    )

    x = list(range(len(scenario_ids)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        [i - width / 2 for i in x],
        random_rates,
        width,
        label="Random Agent",
        color="#94A3B8",
    )

    ax.bar(
        [i + width / 2 for i in x],
        qlearning_rates,
        width,
        label="Q-learning Agent",
        color="#2563EB",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ids)

    ax.set_ylabel("Delivery Success Rate (%)")
    ax.set_xlabel("Scenario")
    ax.set_title("Scenario Evaluation - Agent Comparison")
    ax.set_ylim(0, 105)

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "scenario_success_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # SCENARIO DANGER HITS COMPARISON
    # ========================================================

    random_danger = []
    qlearning_danger = []

    for scenario_id in scenario_ids:
        for row in summary:
            if row["scenario"] == scenario_id:

                if row["agent"] == "Random":
                    random_danger.append(
                        row["avg_danger_hits"]
                    )

                elif row["agent"] == "Q-learning":
                    qlearning_danger.append(
                        row["avg_danger_hits"]
                    )

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        [i - width / 2 for i in x],
        random_danger,
        width,
        label="Random Agent",
        color="#94A3B8",
    )

    ax.bar(
        [i + width / 2 for i in x],
        qlearning_danger,
        width,
        label="Q-learning Agent",
        color="#2563EB",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_ids)

    ax.set_ylabel("Average Danger Hits per Episode")
    ax.set_xlabel("Scenario")
    ax.set_title("Scenario Evaluation - Danger Hits")

    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "scenario_danger_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nScenario evaluation completed!")
    print("Saved: scenario_results/scenario_details.csv")
    print("Saved: scenario_results/scenario_summary.csv")

    print(
        "\nNote: Do not directly compare raw rewards "
        "between different reward presets."
    )


if __name__ == "__main__":
    main()
