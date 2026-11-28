"""Extra analyses used in the final report (Project 09).

Run from the repository root:

    python report_extra_analysis.py

Outputs (all deterministic, fixed seeds) are written to ``report_results/``:

* convergence_summary.csv  - convergence-speed proxy for the five alpha/gamma
                             settings of experiment.py (Midterm rule).
* baseline_stats.csv       - mean and std of every metric per scenario and agent, computed
                             from scenario_results/scenario_details.csv.
* optimal_paths.csv        - length of the shortest *safe* delivery route per
                             map (breadth-first search over (row, col, has_package)).
* altered_maps.csv         - tabular Q-learning (default hyper-parameters) on the
                             original and the altered obstacle maps. The column
                             mean_train_seconds is wall-clock time and therefore
                             machine dependent; all other columns are deterministic.
* f1_diagnostic.csv        - what the agent learns under the ``bad_reward`` preset.
* learning_curves_gray.png - grey-scale version of the alpha/gamma learning curves.
* learned_paths.png        - greedy paths of the trained agent on four maps.

It does not modify student_core.py and does not change any earlier results.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from environment.gridworld import GridWorld  # noqa: E402
from starter.student_core import q_learning, select_action  # noqa: E402

OUT = ROOT / "report_results"
OUT.mkdir(exist_ok=True)

SEEDS = [11, 22, 33, 44, 55]
ALPHA, GAMMA, EPSILON, EPISODES = 0.2, 0.95, 0.2, 1000
DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def greedy_rollout(env: GridWorld, q_table: dict) -> dict:
    state = env.reset()
    total, path, danger = 0.0, [state[:2]], 0
    delivered = False
    while not env.is_terminal():
        action = select_action(state, q_table, env.actions, 0.0)
        res = env.step(action)
        total += float(res.reward)
        danger += int(res.info["hit_danger"])
        delivered = delivered or bool(res.info["delivered"])
        state = res.state
        path.append(state[:2])
    return {"delivered": delivered, "steps": env.steps_taken, "reward": total,
            "danger_hits": danger, "path": path}


def train(map_name: str, preset: str, seed: int, episodes: int = EPISODES) -> tuple:
    random.seed(seed)
    env = GridWorld(map_name=map_name, reward_preset=preset)
    q_table = q_learning(env, episodes, ALPHA, GAMMA, EPSILON)
    return env, q_table


def shortest_safe_steps(map_name: str) -> int:
    """BFS over (row, col, has_package): start -> pickup -> goal, never entering DANGER."""
    env = GridWorld(map_name=map_name)
    grid = env.cell_grid()
    h, w = len(grid), len(grid[0])
    start = env.reset()
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        (r, c, p), d = queue.popleft()
        for dr, dc in DELTAS.values():
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or grid[nr][nc] in ("WALL", "DANGER"):
                continue
            np_ = p or grid[nr][nc] == "PICKUP"
            if grid[nr][nc] == "GOAL" and np_:
                return d + 1
            nxt = (nr, nc, np_)
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, d + 1))
    raise RuntimeError("no safe route")


def n_states(map_name: str) -> int:
    env = GridWorld(map_name=map_name)
    free = sum(1 for row in env.cell_grid() for c in row if c != "WALL")
    return free * 2


# --------------------------------------------------------------------------
# 1. convergence proxy (Midterm rule)
# --------------------------------------------------------------------------
def convergence_episode(returns, window: int = 100, hold: int = 100):
    """First episode at which the rolling-mean return is >= 0 and stays >= 0
    for ``hold`` consecutive episodes. Means are rounded to 9 decimals so that
    an exact 0.0 is not lost to floating-point noise."""
    x = np.asarray(returns, dtype=float)
    rolling = np.array([x[i - window + 1:i + 1].mean() if i >= window - 1 else np.nan
                        for i in range(len(x))])
    ok = np.round(rolling, 9) >= 0
    for i in range(window - 1, len(x) - hold + 1):
        if ok[i:i + hold].all():
            return i + 1
    return None


def run_convergence() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "experiment_results" / "episode_rewards.csv")
    rows = [{"setting": col, "convergence_episode": convergence_episode(df[col])}
            for col in df.columns if col != "episode"]
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "convergence_summary.csv", index=False)
    return out


def run_baseline_stats() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "scenario_results" / "scenario_details.csv")
    rows = []
    for (scen, agent), g in df.groupby(["scenario", "agent"], sort=False):
        rows.append({
            "scenario": scen, "agent": agent, "runs": len(g),
            "success_percent": round(100 * g["success"].mean(), 2),
            "steps_mean": round(g["steps"].mean(), 2), "steps_std": round(g["steps"].std(ddof=1), 2),
            "reward_mean": round(g["reward"].mean(), 2), "reward_std": round(g["reward"].std(ddof=1), 2),
            "danger_hits_mean": round(g["danger_hits"].mean(), 2),
            "goal_without_package_mean": round(g["goal_without_package"].mean(), 2),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "baseline_stats.csv", index=False)
    return out


# --------------------------------------------------------------------------
# 2. optimal paths and altered maps
# --------------------------------------------------------------------------
MAPS_FOR_PATHS = [("easy", "default"), ("medium", "default"), ("hard", "default"),
                  ("medium_altered", "default"), ("custom_test_02", "default")]


def run_optimal_and_altered() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    opt_rows, alt_rows, example_paths = [], [], {}
    for name, preset in MAPS_FOR_PATHS:
        opt = shortest_safe_steps(name)
        opt_rows.append({"map": name, "states": n_states(name), "shortest_safe_steps": opt})
        eps = 2000 if name == "hard" else EPISODES  # same budget as scenario D1
        results = []
        for seed in SEEDS:
            t0 = time.perf_counter()
            env, q = train(name, preset, seed, eps)
            res = greedy_rollout(env, q)
            res["train_seconds"] = time.perf_counter() - t0
            results.append(res)
            if seed == SEEDS[0]:
                example_paths[name] = (env.cell_grid(), res["path"])
        succ = [r for r in results if r["delivered"]]
        alt_rows.append({
            "map": name, "episodes": eps, "seeds": len(SEEDS),
            "success_percent": 100 * len(succ) / len(results),
            "mean_steps_if_success": float(np.mean([r["steps"] for r in succ])) if succ else "",
            "mean_reward": float(np.mean([r["reward"] for r in results])),
            "mean_train_seconds": round(float(np.mean([r["train_seconds"] for r in results])), 2),
            "danger_hits": int(sum(r["danger_hits"] for r in results)),
            "shortest_safe_steps": opt,
            "all_runs_optimal_length": all(r["delivered"] and r["steps"] == opt for r in results),
        })
    opt_df, alt_df = pd.DataFrame(opt_rows), pd.DataFrame(alt_rows)
    opt_df.to_csv(OUT / "optimal_paths.csv", index=False)
    alt_df.to_csv(OUT / "altered_maps.csv", index=False)
    return opt_df, alt_df, example_paths


# --------------------------------------------------------------------------
# 3. F1 diagnostic
# --------------------------------------------------------------------------
def run_f1_diagnostic() -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        env, q = train("medium", "bad_reward", seed)
        visited = {s for (s, _a) in q}
        greedy = Counter(select_action(s, q, env.actions, 0.0) for s in visited)
        rows.append({
            "seed": seed, "q_entries": len(q),
            "nonzero_q_entries": sum(1 for v in q.values() if v != 0.0),
            "visited_states": len(visited),
            "greedy_action_counts": dict(greedy),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "f1_diagnostic.csv", index=False)
    return out


# --------------------------------------------------------------------------
# 4. figures
# --------------------------------------------------------------------------
def plot_learning_curves() -> None:
    df = pd.read_csv(ROOT / "experiment_results" / "episode_rewards.csv")
    styles = [("-", "k"), ("--", "0.35"), (":", "k"), ("-.", "0.35"), ((0, (5, 1, 1, 1, 1, 1)), "0.6")]
    fig, ax = plt.subplots(figsize=(6.2, 3.1), dpi=200)
    for (ls, col), name in zip(styles, [c for c in df.columns if c != "episode"]):
        smooth = df[name].rolling(20).mean()
        ax.plot(df["episode"], smooth, linestyle=ls, color=col, linewidth=1.3, label=name)
    ax.set_ylim(-150, 20)
    ax.set_xlim(0, 300)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Mean return (MA-20)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curves_gray.png")
    plt.close(fig)



# --------------------------------------------------------------------------
# 4. figures
# --------------------------------------------------------------------------
def plot_paths(example_paths: dict) -> None:
    """Plot learned greedy paths on all five maps."""

    names = [
        "easy",
        "medium",
        "hard",
        "medium_altered",
        "custom_test_02",
    ]

    display_names = {
        "easy": "Easy",
        "medium": "Medium",
        "hard": "Hard",
        "medium_altered": "Medium Altered",
        "custom_test_02": "Custom Test 02",
    }

    # 2 rows x 3 columns for five maps
    fig, axes = plt.subplots(
        2, 3,
        figsize=(9.5, 5.8),
        dpi=200,
    )

    axes = axes.flatten()

    for ax, name in zip(axes, names):

        grid, path = example_paths[name]

        h, w = len(grid), len(grid[0])

        # Draw map cells
        for r in range(h):
            for c in range(w):

                cell = grid[r][c]

                face = {
                    "WALL": "0.15",
                    "DANGER": "white",
                    "PICKUP": "0.85",
                    "GOAL": "0.85",
                }.get(cell, "white")

                ax.add_patch(
                    plt.Rectangle(
                        (c, h - 1 - r),
                        1,
                        1,
                        facecolor=face,
                        edgecolor="0.6",
                        linewidth=0.4,
                        hatch="////" if cell == "DANGER" else None,
                    )
                )

                label = {
                    "START": "S",
                    "PICKUP": "P",
                    "GOAL": "G",
                    "DANGER": "D",
                }.get(cell)

                if label:
                    ax.text(
                        c + 0.5,
                        h - 1 - r + 0.5,
                        label,
                        ha="center",
                        va="center",
                        fontsize=7,
                        fontweight="bold",
                        color="k",
                        zorder=5,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="none",
                            pad=0.4,
                        ),
                    )

        # Draw learned path
        xs = [c + 0.5 for r, c in path]
        ys = [h - 1 - r + 0.5 for r, c in path]

        ax.plot(
            xs,
            ys,
            color="k",
            linewidth=1.2,
            marker=".",
            markersize=3,
            zorder=3,
        )

        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title(
            f"{display_names[name]} ({len(path) - 1} steps)",
            fontsize=8,
        )

    # Hide the unused sixth subplot
    for ax in axes[len(names):]:
        ax.set_visible(False)

    fig.suptitle(
        "Learned Greedy Paths on Five Maps",
        fontsize=11,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    fig.savefig(
        OUT / "learned_paths.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    base = run_baseline_stats()
    print("\nMean/std per scenario and agent:\n", base.to_string(index=False))

    conv = run_convergence()
    print("\nConvergence proxy:\n", conv.to_string(index=False))

    opt, alt, paths = run_optimal_and_altered()

    print("\nShortest safe routes:\n", opt.to_string(index=False))
    print("\nTrained agent on original / altered maps:\n", alt.to_string(index=False))

    f1 = run_f1_diagnostic()
    print("\nF1 (bad_reward) diagnostic:\n", f1.to_string(index=False))

    plot_learning_curves()
    plot_paths(paths)

    print(f"\nSaved results and figures to {OUT}")


# --------------------------------------------------------------------------
# Run script
# --------------------------------------------------------------------------
if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "--paths-only":

        print("Generating learned paths for five maps...")

        example_paths = {}

        for name, preset in MAPS_FOR_PATHS:

            episodes = 2000 if name == "hard" else EPISODES

            env, q_table = train(
                name,
                preset,
                SEEDS[0],
                episodes,
            )

            result = greedy_rollout(env, q_table)

            example_paths[name] = (
                env.cell_grid(),
                result["path"],
            )

            print(
                f"{name}: "
                f"{result['steps']} steps, "
                f"delivered={result['delivered']}"
            )

        plot_paths(example_paths)

        print(
            f"\nUpdated figure: {OUT / 'learned_paths.png'}"
        )

    else:
        main()
