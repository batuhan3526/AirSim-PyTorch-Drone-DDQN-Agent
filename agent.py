import math
import random
from collections import deque
import airsim
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from setuptools import glob
from env import DroneEnv
from torch.utils.tensorboard import SummaryWriter
import time

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

writer = SummaryWriter()  #"runs/Mar03_14-55-58_DESKTOP-QGNSALL"

class DQN(nn.Module):
    def __init__(self, in_channels=1, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 84, kernel_size=4, stride=4)
        self.conv2 = nn.Conv2d(84, 42, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(42, 21, kernel_size=2, stride=2)
        self.fc4 = nn.Linear(21*4*4, 168)
        self.fc5 = nn.Linear(168, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        return self.fc5(x)


class Agent:
    def __init__(self, useGPU=False, useDepth=False):
        self.useGPU = useGPU
        self.useDepth = useDepth
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 30000
        self.gamma = 0.8
        self.learning_rate = 0.001
        self.batch_size = 512
        self.max_episodes = 10000
        self.save_interval = 10
        self.episode = -1
        self.steps_done = 0

        if self.useGPU:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')

        self.dqn = DQN()
        self.env = DroneEnv(useGPU, useDepth)
        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.dqn.parameters(), self.learning_rate)


        print('Using device:', self.device)
        if self.device.type == 'cuda':
            print(torch.cuda.get_device_name(0))

        # LOGGING
        cwd = os.getcwd()
        self.save_dir = os.path.join(cwd, "saved models")
        if not os.path.exists(self.save_dir):
            os.mkdir("saved models")

        if self.useGPU:
            self.dqn = self.dqn.to(self.device)  # to use GPU

        # model backup
        files = glob.glob(self.save_dir + '\\*.pt')
        if len(files) > 0:
            files.sort(key=os.path.getmtime)
            file = files[-1]
            checkpoint = torch.load(file)
            self.dqn.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.episode = checkpoint['episode']
            self.steps_done = checkpoint['steps_done']
            print("Saved parameters loaded"
                  "\nModel: ", file,
                  "\nSteps done: ", self.steps_done,
                  "\nEpisode: ", self.episode)

        else:
            if os.path.exists("log.txt"):
                open('log.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('last_episode.txt', 'w').close()
            if os.path.exists("last_episode.txt"):
                open('saved_model_params.txt', 'w').close()

        obs = self.env.reset()
        tensor = self.transformToTensor(obs)
        writer.add_graph(self.dqn, tensor)
    def transformToTensor(self, img):
        if self.useGPU:
            tensor = torch.cuda.FloatTensor(img)
        else:
            tensor = torch.Tensor(img)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.float()
        return tensor

    def convert_size(self, size_bytes):
        if size_bytes == 0:
            return "0B"
        size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return "%s %s" % (s, size_name[i])

    def act(self, state):
        self.eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1
        if random.random() > self.eps_threshold:
            #print("greedy")
            if self.useGPU:
                action = np.argmax(self.dqn(state).cpu().data.squeeze().numpy())
                return int(action)
            else:
                data = self.dqn(state).data
                action = np.argmax(data.squeeze().numpy())
                return int(action)

        else:
            action = random.randrange(0, 4)
            return int(action)

    def memorize(self, state, action, reward, next_state):
        self.memory.append(
            (
                state,
                action,
                torch.cuda.FloatTensor([reward]) if self.useGPU else torch.FloatTensor([reward]),
                self.transformToTensor(next_state),
            )
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.cat(states)
        actions = np.asarray(actions)
        rewards = torch.cat(rewards)
        next_states = torch.cat(next_states)

        if self.useGPU:
            next_q_values = self.dqn(next_states).cpu().detach().numpy()
            max_next_q = torch.cuda.FloatTensor(next_q_values[[range(0, self.batch_size)], [actions]])
            current_q = torch.cuda.FloatTensor(self.dqn(states)[[range(0, self.batch_size)], [actions]])
            expected_q = rewards.to(self.device) + (self.gamma * max_next_q).to(self.device)
        else:
            next_q_values = self.dqn(next_states).detach().numpy()
            max_next_q = next_q_values[[range(0, self.batch_size)], [actions]]
            current_q = self.dqn(states)[[range(0, self.batch_size)], [actions]]
            expected_q = rewards + (self.gamma * max_next_q)

        loss = F.mse_loss(current_q.squeeze(), expected_q.squeeze())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):

        score_history = []
        reward_history = []
        if self.episode == -1:
            self.episode = 1

        for e in range(1, self.max_episodes + 1):
            start = time.time()
            state = self.env.reset()
            steps = 0
            score = 0
            while True:
                state = self.transformToTensor(state)

                action = self.act(state)
                next_state, reward, done = self.env.step(action)

                self.memorize(state, action, reward, next_state)
                self.learn()

                state = next_state
                steps += 1
                score += reward
                if done:
                    print("----------------------------------------------------------------------------------------")
                    print("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}".format(self.episode, reward, round(score/steps, 2), score, self.eps_threshold, self.steps_done))
                    score_history.append(score)
                    reward_history.append(reward)
                    with open('log.txt', 'a') as file:
                        file.write("episode:{0}, reward: {1}, mean reward: {2}, score: {3}, epsilon: {4}, total steps: {5}\n".format(self.episode, reward, round(score/steps, 2), score, self.eps_threshold, self.steps_done))

                    if self.useGPU:
                        print('Total Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory))
                        print('Allocated Memory:', self.convert_size(torch.cuda.memory_allocated(0)))
                        print('Cached Memory:', self.convert_size(torch.cuda.memory_reserved(0)))
                        print('Free Memory:', self.convert_size(torch.cuda.get_device_properties(0).total_memory - (torch.cuda.max_memory_allocated() + torch.cuda.max_memory_reserved())))

                        # tensorboard --logdir=runs
                        memory_usage_allocated = np.float64(round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1))
                        memory_usage_cached = np.float64(round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1))

                        writer.add_scalar("memory_usage_allocated", memory_usage_allocated, self.episode)
                        writer.add_scalar("memory_usage_cached", memory_usage_cached, self.episode)

                    writer.add_scalar('epsilon_value', self.eps_threshold, self.episode)
                    writer.add_scalar('score', score, self.episode)
                    writer.add_scalar('reward', reward, self.episode)
                    writer.add_scalar('Total steps', self.steps_done, self.episode)
                    writer.add_scalars('General Look', {'epsilon_value': self.eps_threshold,
                                                    'score': score,
                                                    'reward': reward}, self.episode)

                    # save checkpoint
                    if self.episode % self.save_interval == 0:
                        checkpoint = {
                            'episode': self.episode,
                            'steps_done': self.steps_done,
                            'state_dict': self.dqn.state_dict(),
                            'optimizer': self.optimizer.state_dict()
                        }
                        torch.save(checkpoint, self.save_dir + '//EPISODE{}.pt'.format(self.episode))

                    self.episode += 1
                    end = time.time()
                    stopWatch = end - start
                    print("Episode is done, episode time: ", stopWatch)

                    break
        writer.close()