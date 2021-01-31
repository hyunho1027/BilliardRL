from mlagents.envs import UnityEnvironment
from agent.sac import SACAgent

# -------------- Config -------------- #
train_mode = True 
load_model = False
load_path = None

train_episode = 1000000 if train_mode else 0
test_episode = 10000
run_episode = train_episode + test_episode
train_start_episode = 5000

print_interval = 10
save_interval = 100

use_visual = True # if False, use vector observation
maxlen = 100000
batch_size = 32

use_dynamic_alpha = True
static_log_alpha = -2.0

actor_lr = 1e-4
critic_lr = 3e-4
alpha_lr = 3e-4
tau = 5e-3
gamma = 0.99

# -------------- Config -------------- #

env_name = "./env/Billiard"
env = UnityEnvironment(file_name=env_name)
default_brain = env.brain_names[0]
brain = env.brains[default_brain]

action_size = brain.vector_action_space_size[0]
agent = SACAgent(action_size, use_visual,
                 actor_lr, critic_lr, alpha_lr,
                 batch_size, maxlen,
                 use_dynamic_alpha, static_log_alpha,
                 tau, gamma)
if load_model: agent.load(load_path)

scores, actor_losses, critic1_losses, critic2_losses, alpha_losses = [], [], [], [], []
for episode in range(run_episode):
    if episode == train_episode:
        if train_mode:
            agent.save()
            train_mode = False

    env_info = env.reset(train_mode=train_mode)[default_brain]
    state = env_info.visual_observations[0] if use_visual\
                                            else env_info.vector_observations[0]
    done, score = False, 0
    while not done:
        action = agent.act(state, train_mode)
        env_info = env.step(action)[default_brain]
        next_state = env_info.visual_observations[0] if use_visual\
                                                     else env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        score += reward
        if train_mode:
            agent.remember(state[0], action[0], [reward], next_state[0], [done])

            if episode > train_start_episode:
                actor_loss, critic1_loss, critic2_loss, alpha_loss = agent.learn()
                actor_losses.append(actor_loss)
                critic1_losses.append(critic1_loss)
                critic2_losses.append(critic2_loss)
                alpha_losses.append(alpha_loss)

                if (episode+1) % save_interval == 0:
                    agent.save()
        state = next_state
    scores.append(score)

    if (episode+1) % print_interval == 0: 
        avg = lambda x: sum(x)/max(len(x), 1)
        print(f"{episode+1} Episode / Score: {avg(scores)}")
        agent.write(avg(scores), avg(actor_losses), avg(critic1_losses), avg(critic2_losses), avg(alpha_losses), episode)
        scores, actor_losses, critic1_losses, critic2_losses, alpha_losses = [], [], [], [], []

env.close()