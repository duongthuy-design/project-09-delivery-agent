
# Project 09 — Autonomous Campus Delivery Agent

**Course:** Artificial Intelligence  
**Project Type:** Reinforcement Learning — Final Project  
**Algorithm:** Tabular Q-learning  
**Programming Language:** Python 3.11  
**Application Framework:** Streamlit

**GitHub Repository:**  
https://github.com/duongthuy-design/project-09-delivery-agent

---

## 1. Project Overview

### 1.1 Problem Description

Reinforcement Learning (RL) is a branch of Artificial Intelligence in which an autonomous agent learns a policy through repeated interactions with its environment.

This project develops an autonomous campus delivery robot that operates in a deterministic grid-world environment.

The robot must:

1. Navigate from its starting position (START).
2. Reach a designated package collection point (PICKUP).
3. Collect the package.
4. Deliver the package to a destination (GOAL).
5. Avoid walls and hazardous areas.
6. Minimize unnecessary movement and maximize cumulative reward.

The environment represents a simplified campus building with corridors, obstacles, delivery locations, and dangerous areas.

The learning algorithm is tabular Q-learning, implemented without deep reinforcement learning libraries.

### 1.2 Project Objectives

The main objectives are:

- Implement tabular Q-learning from first principles.
- Train an autonomous delivery agent.
- Apply epsilon-greedy exploration and exploitation.
- Compare Q-learning against a random baseline.
- Analyze reward design and unsuccessful policies.
- Evaluate performance across different map difficulties.
- Visualize the learned policy and training process.
- Provide an interactive Streamlit application.
- Conduct reproducible experiments across multiple random seeds.

### 1.3 Grid-World Environment

The environment contains the following elements:

| Symbol | Cell Type | Description |
|---|---|---|
| S | START | Initial position of the robot |
| P | PICKUP | Package collection location |
| G | GOAL | Package delivery destination |
| D | DANGER | Hazardous area with a reward penalty |
| # | WALL | Impassable obstacle |
| @ | ROBOT | Current position of the agent |

Three maps are available:

- **Easy:** A relatively open environment suitable for basic navigation.
- **Medium:** A larger environment with obstacles and hazardous cells.
- **Hard:** A complex environment requiring more careful route planning.

---

## 2. Mathematical Formulation

The delivery task is modeled as a Markov Decision Process (MDP):

\[
MDP = \langle S, A, P, R, \gamma \rangle
\]

where:

- \(S\): State space
- \(A\): Action space
- \(P\): State transition dynamics
- \(R\): Reward function
- \(\gamma\): Discount factor

### 2.1 State Space

Each state is represented by:

\[
s = (r, c, p)
\]

where:

- \(r\): Robot row coordinate
- \(c\): Robot column coordinate
- \(p\): Boolean value indicating whether the robot carries a package

The package status is important because the robot has different objectives before and after collecting the package.

### 2.2 Action Space

The robot can perform four actions:

```python
ACTIONS = ("UP", "DOWN", "LEFT", "RIGHT")
```

Movement is restricted by the map boundaries and walls.

### 2.3 Reward Function

Reward configurations are defined in:

`environment/config.json`

The project provides three presets:

- `default`: Standard delivery reward configuration.
- `bad_reward`: A deliberately weak reward design for failure analysis.
- `improved_reward`: An improved configuration with stronger incentives for correct delivery and safer behavior.

The original reward configurations are preserved.

### 2.4 Q-learning Algorithm

The agent updates its Q-values using the Bellman temporal-difference rule:

\[
Q(s,a) \leftarrow Q(s,a) +
\alpha \left[
r+\gamma\max_{a'}Q(s',a')-Q(s,a)
\right]
\]

where:

- \(\alpha\): Learning rate
- \(\gamma\): Discount factor
- \(r\): Immediate reward
- \(s'\): Next state
- \(Q(s,a)\): Estimated value of taking action \(a\) in state \(s\)

For terminal transitions, future reward is not bootstrapped.

### 2.5 Epsilon-Greedy Policy

The agent selects actions according to an epsilon-greedy policy:

- With probability \(\epsilon\), select a random action.
- Otherwise, select an action with the highest Q-value.

During greedy evaluation, epsilon is set to zero.

---

## 3. System Architecture

The project separates environment simulation, learning logic, visualization, and experimental evaluation.

### 3.1 Project Structure

```text
project-09-delivery-agent/
│
├── environment/
│   ├── gridworld.py
│   ├── config.json
│   └── maps/
│       ├── easy.json
│       ├── medium.json
│       └── hard.json
│
├── data/
│   └── scenarios.json
│
├── starter/
│   ├── __init__.py
│   ├── student_core.py
│   ├── visualization.py
│   └── app.py
│
├── tests/
│   └── test_project_09_sanity.py
│
├── experiment.py
├── evaluate_agent.py
├── reward_hacking_experiment.py
├── scenario_evaluation.py
│
├── experiment_results/
├── reward_hacking_results/
├── scenario_results/
│
├── requirements.txt
└── README.md
```

The result directories contain files generated by experiment scripts. They can be regenerated by running the corresponding experiments.

### 3.2 Main Components

**Environment — `environment/gridworld.py`**

Handles:

- Map loading
- State transitions
- Robot movement
- Package collection
- Delivery completion
- Reward calculation
- Hazard detection
- Episode termination

**Q-learning — `starter/student_core.py`**

Implements:

- `select_action()`
- `q_learning()`
- Epsilon-greedy action selection
- Tabular Q-value updates

**Visualization — `starter/visualization.py`**

Provides:

- Grid-world rendering
- Robot path visualization
- Q-value heatmaps
- Learned policy arrows

**Streamlit Application — `starter/app.py`**

Provides the interactive interface for training, running, visualizing, and comparing agents.

---

## 4. Installation and Setup

### 4.1 Requirements

Recommended environment:

- Python 3.11
- pip
- Streamlit
- Matplotlib
- Pandas
- Other dependencies specified in `requirements.txt`

### 4.2 Clone the Repository

```bash
git clone https://github.com/duongthuy-design/project-09-delivery-agent.git

cd project-09-delivery-agent
```

To work with the final-development branch:

```bash
git switch final-development
```

### 4.3 Create a Virtual Environment

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3.11 -m venv .venv

source .venv/bin/activate
```

### 4.4 Install Dependencies

```bash
python -m pip install -r requirements.txt
```

If pytest is not installed, install it separately:

```bash
python -m pip install pytest
```

---

## 5. Running the Streamlit Application

Start the application from the project root:

```bash
python -m streamlit run starter/app.py
```

The application will normally be available at:

http://localhost:8501

### 5.1 Training the Agent

The user can configure:

- Map: Easy, Medium, or Hard
- Reward preset
- Training episodes
- Alpha (learning rate)
- Gamma (discount factor)
- Epsilon (exploration rate)
- Random seed

A recommended initial configuration is:

| Parameter | Value |
|---|---|
| Map | easy |
| Reward | default |
| Episodes | 1000 |
| Alpha | 0.20 |
| Gamma | 0.95 |
| Epsilon | 0.20 |
| Seed | 42 |

Click **Train Q-learning** to train the agent.

The trained Q-table is retained in Streamlit session state for subsequent execution and visualization.

### 5.2 Agent Execution

The interface provides:

- **Reset Episode:** Return the robot to START while retaining the trained Q-table.
- **Step Random:** Execute one random action.
- **Step Trained:** Execute one action using the trained policy.
- **Run Trained Agent:** Execute the learned policy with animated movement.
- **Robot Speed:** Adjust animation delay between steps.

The interface also displays:

- Current robot position
- Package collection status
- Total steps
- Accumulated reward
- Episode status
- Action history
- Robot delivery path

---

## 6. Learning Analytics and Visualization

### 6.1 Learning Curve

The application records the total reward accumulated during each training episode.

It displays:

- Episode Reward
- Moving Average over 50 episodes
- Average Reward over the last 100 episodes

The curve helps visualize whether training performance improves over time.

Because the training policy uses epsilon-greedy exploration, individual episode rewards may fluctuate.

### 6.2 Download Training Results

The application provides a button:

**Download Training Results (CSV)**

The downloaded file includes:

- Episode
- Episode Reward
- Moving Average (50)

The CSV is downloaded through the browser. It is not automatically saved into the project directory.

### 6.3 Q-value Heatmap

The Q-value Heatmap visualizes the maximum learned Q-value at each grid position:

\[
V(s)=\max_a Q(s,a)
\]

Two modes are available:

- Before Pickup
- After Pickup

Cells without recorded Q-values are visually distinguished from learned states.

### 6.4 Learned Policy Arrows

Policy Arrows display the greedy action associated with each learned state.

The visualization uses:

- ↑ UP
- ↓ DOWN
- ← LEFT
- → RIGHT

States without a clear learned action preference are not assigned a directional arrow in the visualization.

### 6.5 Path Visualization and Animation

The robot's movement is animated on the grid.

A dashed path shows the positions visited during the episode.

The user can adjust the animation speed.

---

## 7. Random Agent vs. Q-learning Comparison

The Streamlit application supports comparison between a random baseline and the trained Q-learning policy.

Evaluation metrics include:

1. Delivery Success Rate (%)
2. Average Steps
3. Average Reward
4. Average Danger Hits

For the interactive comparison:

- The random baseline is evaluated over multiple episodes.
- Q-learning uses the current trained greedy policy.
- Both agents use the same selected map and reward configuration.

The default random baseline uses 30 evaluation runs.

Because the interactive Q-learning evaluation uses one trained policy on a deterministic map, this comparison alone does not measure variability across training seeds.

The separate scenario evaluation script addresses that limitation.

---

## 8. Benchmark Scenarios

The benchmark scenarios are defined in:

`data/scenarios.json`

| ID | Class | Map | Reward Preset |
|---|---|---|---|
| N1 | Normal | easy | default |
| D1 | Difficult | hard | default |
| D2 | Difficult | medium | default |
| F1 | Failure | medium | bad_reward |
| X1 | Extra | medium | improved_reward |

### N1 — Normal Delivery

Evaluates basic delivery performance and path efficiency on the Easy map.

### D1 — Difficult Navigation

Evaluates delivery performance and hazard avoidance on the Hard map.

### D2 — Danger Avoidance

Evaluates whether the agent can complete delivery while avoiding hazardous cells on the Medium map.

### F1 — Bad Reward Design

Examines behavior when the reward function does not sufficiently encourage efficient and correct delivery.

The configuration can reward reaching GOAL without collecting the package.

However, the observed trained policy in the current experiments failed to complete delivery and did not exhibit the proposed reward-exploitation behavior.

### X1 — Improved Reward Design

Evaluates how improved reward incentives affect successful delivery, step efficiency, and hazard avoidance.

F1 and X1 use the same Medium map but different reward presets.

---

## 9. Running Experiments

All scripts should be executed from the project root directory.

### 9.1 Basic Agent Evaluation

```bash
python evaluate_agent.py
```

Trains and evaluates a delivery policy, then prints the delivery path, total steps, and reward.

### 9.2 Hyperparameter Experiment

```bash
python experiment.py
```

Runs controlled Q-learning experiments and saves results to:

`experiment_results/`

### 9.3 Reward Design Experiment

```bash
python reward_hacking_experiment.py
```

Compares:

- F1 — `bad_reward`
- X1 — `improved_reward`

Generated results are saved in:

`reward_hacking_results/`

These experiments examine the effects of reward design. Observed failures must not automatically be interpreted as demonstrated reward hacking.

### 9.4 Full Scenario Evaluation

```bash
python scenario_evaluation.py
```

This script automatically reads all five benchmark scenarios from `data/scenarios.json`.

**Q-learning evaluation settings:**

- Training seeds: 11, 22, 33, 44, 55
- Episodes: 1000 for Easy and Medium
- Episodes: 2000 for Hard
- Alpha: 0.20
- Gamma: 0.95
- Epsilon: 0.20
- Greedy evaluation epsilon: 0.0

**Random baseline:**

- 30 independent evaluation episodes per scenario

The script evaluates:

- Delivery success
- Episode steps
- Cumulative reward
- Danger hits
- GOAL visits without a package

### 9.5 Generated Files

The evaluation creates:

```text
scenario_results/
├── scenario_details.csv
├── scenario_summary.csv
├── scenario_success_comparison.png
└── scenario_danger_comparison.png
```

**scenario_details.csv**

Contains individual evaluation results for each seed or random run.

**scenario_summary.csv**

Contains aggregated metrics for each scenario and agent.

**scenario_success_comparison.png**

Compares delivery success rates across the five scenarios.

**scenario_danger_comparison.png**

Compares the average number of danger encounters.

---

## 10. Experimental Results

The following results were obtained using the evaluation configuration described above.

### 10.1 Delivery Success Rate

| Scenario | Random Agent | Q-learning Agent |
|---|---:|---:|
| N1 | 63.33% | 100% |
| D1 | 0.00% | 100% |
| D2 | 30.00% | 100% |
| F1 | 63.33% | 0% |
| X1 | 30.00% | 100% |

Random results are based on 30 runs per scenario.

Q-learning results are based on five independently trained policies using different random seeds.

### 10.2 Average Steps

| Scenario | Random Agent | Q-learning Agent |
|---|---:|---:|
| N1 | 207.67 | 10.00 |
| D1 | 671.47 | 46.00 |
| D2 | 158.30 | 15.00 |
| F1 | 254.27 | 378.00 |
| X1 | 158.30 | 15.00 |

Average Steps includes both successful and unsuccessful evaluation episodes.

### 10.3 Average Danger Hits

| Scenario | Random Agent | Q-learning Agent |
|---|---:|---:|
| N1 | 0.00 | 0.00 |
| D1 | 0.53 | 0.00 |
| D2 | 0.60 | 0.00 |
| F1 | 3.40 | 0.00 |
| X1 | 0.60 | 0.00 |

### 10.4 Discussion

**Normal and Difficult Scenarios**

Q-learning achieved successful delivery across all five evaluated training seeds for N1, D1, and D2.

On D1 and D2, the evaluated greedy policies also avoided recorded danger encounters.

**Bad Reward Scenario**

In F1, Q-learning failed to deliver the package in all five evaluations.

The agent reached the episode step limit of 378 steps.

The recorded number of GOAL visits without a package was zero for these policies.

Therefore, the experiment demonstrates a failure associated with the reward configuration, but does not demonstrate actual reward exploitation by the trained agent.

**Improved Reward Scenario**

In X1, Q-learning achieved successful delivery across all five evaluated seeds, using 15 steps and recording no danger hits.

The improvement from F1 to X1 illustrates the importance of appropriately designed reward signals.

### 10.5 Evaluation Limitations

- Five training seeds provide preliminary evidence but do not cover all possible training outcomes.
- The environment uses deterministic state transitions.
- A policy successful on one map configuration may not generalize to modified obstacle layouts.
- Zero danger encounters in F1 does not imply that the agent learned safe navigation.
- Raw cumulative rewards should not be directly compared between different reward presets because their reward scales differ.
- Interactive Streamlit comparisons and multi-seed script evaluations use different evaluation protocols.

---

## 11. Testing

Run the unit tests using:

```bash
python -m pytest -q tests
```

The latest verified test result was:

```text
22 passed
```

The test suite checks environment behavior and selected Q-learning functionality.

Passing the tests does not guarantee optimal policies or successful generalization to every map.

The application and experiment scripts should also be checked through direct execution.

---

## 12. Reproducibility

To reproduce the experiments:

1. Use Python 3.11.
2. Install the project dependencies.
3. Keep the supplied map and reward configuration files unchanged.
4. Run `scenario_evaluation.py`.
5. Inspect the generated CSV and PNG files.
6. Compare the results using the same seeds and hyperparameters.

Random seeds are configured in the evaluation scripts to support reproducibility.

The project does not require external LLM APIs or pretrained Q-tables.

The Q-learning algorithm is trained locally when the application or experiment scripts are executed.

---

## 13. Project Contributions and Design

The implementation emphasizes a clear separation between:

- Environment dynamics
- Reinforcement learning
- Visualization
- Interactive execution
- Experimental evaluation

The completed application includes:

- Tabular Q-learning
- Epsilon-greedy action selection
- Interactive map and reward selection
- Animated robot execution
- Package delivery status tracking
- Path visualization
- Learning Curve
- Downloadable training results
- Q-value Heatmap
- Policy Arrows
- Random Agent comparison
- Danger Hits evaluation
- Multi-seed scenario evaluation
- CSV and PNG experiment outputs

The environment's original reward presets have been retained.

---

## 14. Limitations and Future Improvements

Potential extensions include:

- Side-by-side animated playback of Random and Q-learning agents
- More extensive multi-seed experiments
- Evaluation on altered or unseen obstacle maps
- Improved exploration and tie-breaking analysis
- Additional ablation studies
- More comprehensive automated tests
- Further analysis of reward exploitation behavior

The current Streamlit application animates the trained policy and provides quantitative Random Agent comparisons. It does not yet provide simultaneous animated playback of both agents.

---

## 15. Submission Notes

The final submission package should follow the instructor's separate submission guide.

The required submission structure includes:

- Midterm presentation slides in PDF format
- Runnable project source code
- Final report in the prescribed format
- Instructions for reproducing the experiments

The final report should document the experimental methodology, results, limitations, and interpretation of reward-design failures.

Follow the submission guide's DOCX report requirement and naming conventions when preparing the final ZIP.

Do not include virtual environments, local IDE configuration files, caches, or secrets in the submission package.

---

## 16. References

[1] C. J. C. H. Watkins and P. Dayan, "Q-learning," *Machine Learning*, vol. 8, no. 3–4, pp. 279–292, 1992.  
DOI: 10.1007/BF00992698

[2] R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed. Cambridge, MA, USA: MIT Press, 2018.

[3] S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, 4th ed. Hoboken, NJ, USA: Pearson, 2020.
