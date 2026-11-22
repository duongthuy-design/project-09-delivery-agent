"""Rendering helpers for the Streamlit starter UI.

This module only draws the grid; it contains no AI logic. Kept separate
from ``environment/gridworld.py`` so the environment package has no UI
dependency.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from environment.gridworld import (
    CELL_DANGER,
    CELL_GOAL,
    CELL_PICKUP,
    CELL_START,
    CELL_WALL,
    GridWorld,
)

_CELL_COLORS = {
    CELL_START: "#BFDBFE",   # soft blue — start
    "EMPTY": "#F8FAFC",
    CELL_WALL: "#334155",    # slate — wall
    CELL_PICKUP: "#FDE68A",  # amber — pickup
    CELL_GOAL: "#86EFAC",    # green — goal
    CELL_DANGER: "#FCA5A5",  # rose — danger (distinct from goal)
}

_CELL_LABELS = {
    CELL_START: "S",
    "EMPTY": "",
    CELL_WALL: "",
    CELL_PICKUP: "P",
    CELL_GOAL: "G",
    CELL_DANGER: "D",
}


def plot_grid(env: GridWorld, path=None) -> Figure:
    """Return a matplotlib Figure showing the current grid and agent."""
    grid = env.cell_grid()
    height, width = env.height, env.width

    fig, ax = plt.subplots(figsize=(min(1.0 * width, 9), min(1.0 * height, 9)))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    for r in range(height):
        for c in range(width):
            cell_type = grid[r][c]
            color = _CELL_COLORS.get(cell_type, "#f8fafc")
            ax.add_patch(Rectangle((c, r), 1, 1, facecolor=color, edgecolor="#cbd5e1", linewidth=0.8))
            label = _CELL_LABELS.get(cell_type, "")
            if label:
                ax.text(c + 0.5, r + 0.5, label, ha="center", va="center", fontsize=9, color="#1e293b")


    # ========================================================
    # DRAW ROBOT DELIVERY PATH
    # ========================================================

    if path is not None and len(path) > 1:

        # Convert grid positions into cell-center coordinates.
        path_x = [
            col + 0.5
            for row, col in path
        ]

        path_y = [
            row + 0.5
            for row, col in path
        ]

        # Draw the route followed by the robot.
        ax.plot(
            path_x,
            path_y,
            color="#F97316",
            linewidth=2.5,
            linestyle="--",
            marker="o",
            markersize=3,
            alpha=0.85,
            zorder=2,
        )



    agent_r, agent_c = env.position
    ax.add_patch(
        Rectangle(
            (agent_c + 0.15, agent_r + 0.15),
            0.7,
            0.7,
            facecolor="#2563EB" if env.state[2] else "#3B82F6",
            edgecolor="#1E3A8A",
            linewidth=1.2,
            zorder=4,
        )
    )
    ax.text(
        agent_c + 0.5,
        agent_r + 0.5,
        "@",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        fontweight="bold",
        zorder=5,
    )


    fig.tight_layout()
    return fig


# ============================================================
# Q-VALUE HEATMAP
# ============================================================

def plot_q_heatmap(
    env: GridWorld,
    q_table: dict,
    has_package: bool = False,
) -> Figure:
    """Visualize max Q-value for each grid state."""

    grid = env.cell_grid()

    heatmap = []

    # Calculate the highest Q-value at each position.
    for row in range(env.height):

        row_values = []

        for col in range(env.width):

            cell_type = grid[row][col]

            # Walls cannot be visited.
            if cell_type == CELL_WALL:
                row_values.append(float("nan"))
                continue

            state = (row, col, has_package)

            # Check whether the agent learned this state.
            learned = any(
                (state, action) in q_table
                for action in env.actions
            )

            if learned:

                best_q = max(
                    q_table.get((state, action), 0.0)
                    for action in env.actions
                )

                row_values.append(float(best_q))

            else:

                # Unknown states are shown in gray.
                row_values.append(float("nan"))

        heatmap.append(row_values)

    # --------------------------------------------------------
    # Draw heatmap
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(
            max(6, env.width * 0.8),
            max(4, env.height * 0.8),
        )
    )

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#E2E8F0")

    image = ax.imshow(
        heatmap,
        cmap=cmap,
        interpolation="nearest",
        origin="upper",
    )

    # Draw grid lines.
    ax.set_xticks(
        [x - 0.5 for x in range(env.width + 1)],
        minor=True,
    )

    ax.set_yticks(
        [y - 0.5 for y in range(env.height + 1)],
        minor=True,
    )

    ax.grid(
        which="minor",
        color="white",
        linewidth=1,
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False,
    )

    # --------------------------------------------------------
    # Draw walls and important locations
    # --------------------------------------------------------

    for row in range(env.height):

        for col in range(env.width):

            cell_type = grid[row][col]

            if cell_type == CELL_WALL:

                ax.add_patch(
                    Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        facecolor="#334155",
                        zorder=3,
                    )
                )

            else:

                label = _CELL_LABELS.get(
                    cell_type, ""
                )

                if label:

                    ax.text(
                        col,
                        row,
                        label,
                        ha="center",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                        color="black",
                        zorder=4,
                    )

    # --------------------------------------------------------
    # Labels and colorbar
    # --------------------------------------------------------

    ax.set_title(
        "Q-value Heatmap - "
        + (
            "After Pickup"
            if has_package
            else "Before Pickup"
        )
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    fig.colorbar(
        image,
        ax=ax,
        label="Maximum Q-value",
        fraction=0.046,
        pad=0.04,
    )

    fig.tight_layout()

    return fig


# ============================================================
# POLICY ARROWS VISUALIZATION
# ============================================================

def plot_policy_arrows(
    env: GridWorld,
    q_table: dict,
    has_package: bool = False,
) -> Figure:
    """Visualize the greedy policy learned by Q-learning."""

    grid = env.cell_grid()

    action_arrows = {
        "UP": "↑",
        "DOWN": "↓",
        "LEFT": "←",
        "RIGHT": "→",
    }

    fig, ax = plt.subplots(
        figsize=(
            max(6, env.width * 0.8),
            max(4, env.height * 0.8),
        )
    )

    # --------------------------------------------------------
    # Draw map and learned policy arrows
    # --------------------------------------------------------

    for row in range(env.height):
        for col in range(env.width):

            cell_type = grid[row][col]

            color = _CELL_COLORS.get(
                cell_type,
                "#F8FAFC",
            )

            # Draw the background cell.
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    facecolor=color,
                    edgecolor="#CBD5E1",
                    linewidth=1,
                )
            )

            # No policy arrow inside walls.
            if cell_type == CELL_WALL:
                continue

            state = (
                row,
                col,
                has_package,
            )

            # Do not draw an action for a completed delivery.
            if not (
                cell_type == CELL_GOAL
                and has_package
            ):

                # Only display policy for states that
                # have at least one recorded Q-value.
                learned = any(
                    (state, action) in q_table
                    for action in env.actions
                )

                if learned:

                    action_values = [
                        q_table.get(
                            (state, action),
                            0.0,
                        )
                        for action in env.actions
                    ]

                    # Equal Q-values do not indicate
                    # a clear learned preference.
                    if max(action_values) != min(action_values):

                        best_action = max(
                            env.actions,
                            key=lambda action: q_table.get(
                                (state, action),
                                0.0,
                            ),
                        )

                        arrow = action_arrows.get(
                            best_action,
                            "",
                        )

                        ax.text(
                            col,
                            row,
                            arrow,
                            ha="center",
                            va="center",
                            fontsize=20,
                            fontweight="bold",
                            color="#2563EB",
                            zorder=3,
                        )

            # Label important map locations.
            label = _CELL_LABELS.get(
                cell_type,
                "",
            )

            if label:
                ax.text(
                    col - 0.35,
                    row - 0.30,
                    label,
                    ha="left",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="#1E293B",
                    zorder=4,
                )

    # --------------------------------------------------------
    # Configure figure
    # --------------------------------------------------------

    ax.set_xlim(
        -0.5,
        env.width - 0.5,
    )

    ax.set_ylim(
        env.height - 0.5,
        -0.5,
    )

    ax.set_aspect("equal")

    ax.set_xticks(
        range(env.width)
    )

    ax.set_yticks(
        range(env.height)
    )

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    ax.set_title(
        "Learned Policy - "
        + (
            "After Pickup"
            if has_package
            else "Before Pickup"
        )
    )

    fig.tight_layout()

    return fig

