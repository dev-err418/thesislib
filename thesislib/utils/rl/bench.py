from itertools import count


class MedBench:
    def __init__(self, agent, num_episodes=50):
        self.agent = agent
        self.rewardList = []
        self.episode_duration = []
        self.episode_loss = []
        self.num_episodes = num_episodes

    def run_trial(self, debug=True):
        for idx in range(self.num_episodes):
            self.agent.reset_env()
            rewards = 0
            for jdx in count():
                if debug and jdx % 10 == 0:
                    print("Step %d in Episode: %d" % (jdx, idx))
                reward, done = self.agent.step()
                # optimize
                loss = self.agent.update()
                self.episode_loss.append(loss)
                rewards += reward
                if done:
                    num_runs = jdx + 1
                    if debug:
                        print("Completed Episode: %d in %d steps. Final Reward: %d" %(idx, num_runs, reward))
                    self.episode_duration.append(num_runs)
                    self.rewardList.append(rewards)
                    break
