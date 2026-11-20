
import random

from environment.gridworld import GridWorld
from starter.student_core import q_learning, select_action


# 1. Set random seed for reproducibility
random.seed(42)

# 2. Create the training environment
train_env = GridWorld(
    map_name="easy",
    reward_preset="default"
)

# 3. Train the Q-learning agent
print("Training Q-learning agent...")

q_table = q_learning(
    env=train_env,
    episodes=1000,
    alpha=0.5,
    gamma=0.9,
    epsilon=0.2
)

print("Training completed!")
print("Q-table entries:", len(q_table))


# 4. Create a new environment for evaluation
eval_env = GridWorld(
    map_name="easy",
    reward_preset="default"
)

state = eval_env.reset()

total_reward = 0.0
delivered = False
picked_up = False
path = [state[:2]]

print("\n--- EVALUATION START ---")

# 5. Run the trained policy
while not eval_env.is_terminal():

    # epsilon = 0: use the learned Q-values
    action = select_action(
        state=state,
        q_table=q_table,
        actions=eval_env.actions,
        epsilon=0.0
    )

    # Execute the selected action
    result = eval_env.step(action)

    total_reward += result.reward

    if result.info["picked_up"]:
        picked_up = True

    if result.info["delivered"]:
        delivered = True

    state = result.state
    path.append(state[:2])

    print(
        f"Step {eval_env.steps_taken}: "
        f"Action={action}, "
        f"State={state}, "
        f"Reward={result.reward}"
    )


# 6. Display evaluation results
print("\n--- EVALUATION RESULTS ---")

print("Package picked up:", picked_up)
print("Delivery successful:", delivered)
print("Total steps:", eval_env.steps_taken)
print("Total reward:", total_reward)
print("Path:", path)

if delivered:
    print("\nSUCCESS: The agent delivered the package!")
else:
    print("\nFAILED: The agent did not complete the delivery.")

    if result.info["hit_danger"]:
        print("Reason: Entered a danger cell.")
    elif result.info["timed_out"]:
        print("Reason: Maximum steps reached.")
