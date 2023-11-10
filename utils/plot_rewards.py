from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

# Load TensorBoard event file
log_path = '../runs/Olympus/summaries'
event_acc = event_accumulator.EventAccumulator(log_path)
event_acc.Reload()

# Specify the tags for the desired rewards/metrics
tags_to_plot = [
    'detailed_rewards/orient/iter', 
    'detailed_rewards/is_done/iter', 
    'detailed_rewards/is_within_threshold/iter', 
    'detailed_rewards/torque_clip/iter',
    'detailed_rewards/action_clip/iter',
    'detailed_rewards/collision/iter',
    'losses/a_loss',
    'losses/bounds_loss',
    'losses/c_loss',
    'losses/entropy',
    'episode_lengths/iter',
    'rewards/step'
    ]

try:
    for tag in tags_to_plot:
        # Create a new plot for each tag
        plt.figure(figsize=(10, 5))
        
        reward_events = event_acc.Scalars(tag)
        steps = [event.step for event in reward_events]
        rewards = [event.value for event in reward_events]

        # Plot the metrics
        plt.plot(steps, rewards)
        
        # Set title and labels
        plt.title(tag.replace('detailed_rewards/', '').replace('/iter', ''))
        plt.xlabel('Steps')
        plt.ylabel('Reward')

        # Save the plot as a PDF. NB: DO NOT USE EPS
        plt.savefig(f'../plots/{tag.replace("/", "_")}_plot.pdf', format='pdf')
        
except KeyError as e:
    print("KeyError:", e)