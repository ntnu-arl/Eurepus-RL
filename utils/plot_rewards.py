from tensorboard.backend.event_processing import event_accumulator
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# List of log directories
log_directories = [
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_2/Olympus/summaries',
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_3/Olympus/summaries',
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_5/Olympus/summaries',
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_7/Olympus/summaries',
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_9/Olympus/summaries',
    '/Olympus-ws/in-air-stabilization/Runs - interpolated with fixed penalty rewards/run_10/Olympus/summaries',
]

# Specify the tags for the desired rewards/metrics

data_to_plot = {
    'detailed_rewards/orient/iter' : ['Orientation reward','Batch reward sum'], 
    'detailed_rewards/is_done/iter' : ['Terminal reward','Batch reward sum'], 
    'detailed_rewards/inside_threshold/iter' : ['Within threshold reward','Batch reward sum'], 
    'detailed_rewards/inside_threshold_reset/iter' : ['Terminal reward','Batch reward sum'], 
    'detailed_rewards/torque_clip/iter' : ['Torque clip penalty','Batch reward sum'],
    'detailed_rewards/action_clip/iter' : ['Action clip penalty','Batch reward sum'],
    'detailed_rewards/collision/iter' : ['Collision reward','Batch reward sum'],
    'detailed_rewards/orient_integral/iter' : ['Orientation integral reward','Batch reward sum'],
    'detailed_rewards/change_dir/iter' : ['Motor direction change penalty','Batch reward sum'],
    'detailed_rewards/regularize/iter' : ['Torque regularization penalty','Batch reward sum'],
    'detailed_rewards/velocity/iter' : ['Velocity penalty','Batch reward sum'],

    'losses/a_loss' : ['Actor loss','loss'],
    'losses/bounds_loss' : ['Bounds loss','loss'],
    'losses/c_loss' : ['Critic Loss','loss'],
    'losses/entropy' : ['Entropy','entropy'],
    'episode_lengths/iter' : ['Episode length','Episode length'],

    'rewards/iter'  : ['Total game reward','Reward']
}

# Dictionary to store rewards data for each tag
tag_rewards_data = {tag: {'steps': [], 'rewards': []} for tag in data_to_plot.keys()}

# Loop over log directories
for log_path in log_directories:
    print("extracting data: ", log_path)
    event_acc = event_accumulator.EventAccumulator(log_path)
    event_acc.Reload()

    for tag in data_to_plot.keys():
        try:
            reward_events = event_acc.Scalars(tag)
            steps = [event.step for event in reward_events]
            rewards = [event.value for event in reward_events]

            # Store data for each run
            tag_rewards_data[tag]['steps'].append(steps)
            batch_size = 4096
            if "detailed" in tag:
                tag_rewards_data[tag]['rewards'].append([r/batch_size for r in rewards])
            else:
                tag_rewards_data[tag]['rewards'].append(rewards)

        except KeyError as e:
            print(f"KeyError: {e} - Skipping tag: {tag}")

for tag in data_to_plot.keys():
    tag_rewards_data[tag]['steps'] = [steps[:1999] for steps in tag_rewards_data[tag]['steps']]
    tag_rewards_data[tag]['rewards'] = [rewards[:1999] for rewards in tag_rewards_data[tag]['rewards']]


terminal_1 = tag_rewards_data["detailed_rewards/is_done/iter"]['rewards']
terminal_2 = tag_rewards_data['detailed_rewards/inside_threshold_reset/iter']['rewards']
tag_rewards_data["detailed_rewards/is_done/iter"]['rewards'] = [[a+b for a,b in zip(t1,t2)] for t1, t2 in zip(terminal_1, terminal_2)]

# Compute mean and standard deviation for each tag
for tag, data in tag_rewards_data.items():
    all_rewards = np.array(data['rewards'])
    mean_rewards = np.mean(all_rewards, axis=0)
    std_rewards = np.std(all_rewards, axis=0)

    # Now you can use 'mean_rewards' and 'std_rewards' as needed
    print(f"Tag: {tag}")
    print("Mean Rewards:", mean_rewards)
    print("Standard Deviation:", std_rewards)

    # Plot the mean curve with shaded standard deviation
    plt.figure(figsize=(5.4, 2.4))
    plt.plot(data['steps'][0], mean_rewards, color=[0, 0.4470, 0.7410], label='Mean')

    min_rewards = np.min(all_rewards, axis=0)
    max_rewards = np.max(all_rewards, axis=0)
    plt.fill_between(data['steps'][0], min_rewards, max_rewards, alpha=0.3, color=[0.9290, 0.6940, 0.1250], label='Min-Max Range')

    plt.fill_between(data['steps'][0], mean_rewards - std_rewards, mean_rewards + std_rewards, alpha=0.5, color=[0, 0.4470, 0.7410], label='Std Dev')
    
    # Set title and labels
    plt.title(f"{data_to_plot[tag][0]}", fontsize=10)
    # plt.ylabel(data_to_plot[tag][1], fontsize=10)
    plt.xlabel("Epochs")
    plt.legend()

    plt.gcf().subplots_adjust(bottom=0.2)

    # Save the plot as a PDF and PGF. NB: DO NOT USE EPS
    plt.savefig(f'../plots/pgf/{tag.replace("/", "_")}_mean_std_plot.pgf', format='pgf')
    plt.savefig(f'../plots/pdf/{tag.replace("/", "_")}_mean_std_plot.pdf', format='pdf')


######################################## FOR SINGLE EVENT FILE (no std) ################################################

# # Load TensorBoard event file
# log_path = '/Olympus-ws/in-air-stabilization/Runs - interpolated with termination/run_3'
# event_acc = event_accumulator.EventAccumulator(log_path)
# event_acc.Reload()

# # Specify the tags for the desired rewards/metrics

# data_to_plot = {
#     'detailed_rewards/orient/iter' : ['Orientation reward','Batch reward sum'], 
#     'detailed_rewards/is_done/iter' : ['Terminal reward','Batch reward sum'], 
#     'detailed_rewards/is_within_threshold/iter' : ['Within threshold reward','Batch reward sum'], 
#     'detailed_rewards/inside_threshold/iter' : ['Within threshold reward','Batch reward sum'], 
#     'detailed_rewards/torque_clip/iter' : ['Torque clip reward','Batch reward sum'],
#     'detailed_rewards/action_clip/iter' : ['Action clip reward','Batch reward sum'],
#     'detailed_rewards/collision/iter' : ['Collision reward','Batch reward sum'],
#     'detailed_rewards/orient_integral/iter' : ['Orientation integral reward','Batch reward sum'],
#     'detailed_rewards/change_dir/iter' : ['Change direction reward','Batch reward sum'],
#     'detailed_rewards/regularize/iter' : ['Regularization reward','Batch reward sum'],
#     'detailed_rewards/velocity/iter' : ['Velocity reward','Batch reward sum'],

#     'losses/a_loss' : ['Actor loss','loss'],
#     'losses/bounds_loss' : ['Bounds loss','loss'],
#     'losses/c_loss' : ['Critic Loss','loss'],
#     'losses/entropy' : ['Entropy','entropy'],
#     'episode_lengths/iter' : ['Episode lengths','Episode length'],
#     'rewards/iter'  : ['Total generalized reward','Reward']
# }


# for tag in data_to_plot.keys():
#     try:
#         # Create a new plot for each tag
#         plt.figure(figsize=(6, 2.5))
        
#         reward_events = event_acc.Scalars(tag)
#         steps = [event.step for event in reward_events]
#         rewards = [event.value for event in reward_events]

#         # Plot the metrics
#         plt.plot(steps, rewards)
        
#         # Set title and labels
#         plt.title(data_to_plot[tag][0], fontsize=10)
#         # plt.xlabel('Epochs', fontsize=10)
#         plt.ylabel(data_to_plot[tag][1], fontsize=10)

#         # Save the plot as a PDF and PGF. NB: DO NOT USE EPS
#         plt.savefig(f'../plots/pgf/{tag.replace("/", "_")}_plot.pgf', format='pgf')
#         plt.savefig(f'../plots/pdf/{tag.replace("/", "_")}_plot.pdf', format='pdf')

        
#     except KeyError as e:
#         print(f"KeyError: {e} - Skipping tag: {tag}")
