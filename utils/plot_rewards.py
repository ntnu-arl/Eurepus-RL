from tensorboard.backend.event_processing import event_accumulator
import matplotlib

import matplotlib.pyplot as plt
matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Load TensorBoard event file
log_path = '/Olympus-ws/in-air-stabilization/Event_Files/3D_sehr_gut_mitt_interpolation_kern'
event_acc = event_accumulator.EventAccumulator(log_path)
event_acc.Reload()

# Specify the tags for the desired rewards/metrics

data_to_plot = {
    'detailed_rewards/orient/iter' : ['Orientation reward','Batch reward sum'], 
    'detailed_rewards/is_done/iter' : ['Terminal reward','Batch reward sum'], 
    'detailed_rewards/is_within_threshold/iter' : ['Within threshold reward','Batch reward sum'], 
    'detailed_rewards/inside_threshold/iter' : ['Within threshold reward','Batch reward sum'], 
    'detailed_rewards/torque_clip/iter' : ['Torque clip reward','Batch reward sum'],
    'detailed_rewards/action_clip/iter' : ['Action clip reward','Batch reward sum'],
    'detailed_rewards/collision/iter' : ['Collision reward','Batch reward sum'],
    'detailed_rewards/orient_integral/iter' : ['Orientation integral reward','Batch reward sum'],
    'detailed_rewards/change_dir/iter' : ['Change direction reward','Batch reward sum'],
    'detailed_rewards/regularize/iter' : ['Regularization reward','Batch reward sum'],
    'detailed_rewards/velocity/iter' : ['Velocity reward','Batch reward sum'],

    'losses/a_loss' : ['Actor loss','loss'],
    'losses/bounds_loss' : ['Bounds loss','loss'],
    'losses/c_loss' : ['Critic Loss','loss'],
    'losses/entropy' : ['Entropy','entropy'],
    'episode_lengths/iter' : ['Episode lengths','Episode length'],
    'rewards/iter'  : ['Total generalized reward','Reward']
}

tags_to_plot = [
    'detailed_rewards/orient/iter', 
    'detailed_rewards/is_done/iter', 
    'detailed_rewards/is_within_threshold/iter',
    'detailed_rewards/inside_threshold/iter', 
    'detailed_rewards/torque_clip/iter',
    'detailed_rewards/action_clip/iter',
    'detailed_rewards/collision/iter',
    'detailed_rewards/orient_integral/iter',
    'losses/a_loss',
    'losses/bounds_loss',
    'losses/c_loss',
    'losses/entropy',
    'episode_lengths/iter',
    'rewards/iter'
    ]

for tag in data_to_plot.keys():
    try:
        # Create a new plot for each tag
        plt.figure(figsize=(6, 2.5))
        
        reward_events = event_acc.Scalars(tag)
        steps = [event.step for event in reward_events]
        rewards = [event.value for event in reward_events]

        # Plot the metrics
        plt.plot(steps, rewards)
        
        # Set title and labels
        plt.title(data_to_plot[tag][0], fontsize=10)
        # plt.xlabel('Epochs', fontsize=10)
        plt.ylabel(data_to_plot[tag][1], fontsize=10)

        # Save the plot as a PDF and PGF. NB: DO NOT USE EPS
        plt.savefig(f'../plots/pgf/{tag.replace("/", "_")}_plot.pgf', format='pgf')
        plt.savefig(f'../plots/pdf/{tag.replace("/", "_")}_plot.pdf', format='pdf')

        
    except KeyError as e:
        print(f"KeyError: {e} - Skipping tag: {tag}")
