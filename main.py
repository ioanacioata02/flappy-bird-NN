import gymnasium
import flappy_bird_gymnasium
from collections import deque
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
import signal
import sys
import os
from torch.optim.lr_scheduler import LambdaLR
import random
import numpy as np
import torch

BUFFER_SIZE = 50000
BATCH_SIZE = 32
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_END = 0.0001
EPSILON_DECAY = 0.9925
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 50
NUM_EPISODES = 2000
RAND_ZERO = 0.85
IMG_SIZE = 128
PATIENCE = 8

step_count = 0


def adaptive_lr_schedule(step, start_lr=1e-4, end_lr=1e-7, max_steps=2e5):
    progress = min(step / max_steps, 1.0)
    new_lr = ((1 - progress) * start_lr + progress * end_lr) / LEARNING_RATE
    return new_lr


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

        )
        self.fc = nn.Sequential(
            nn.Linear(self._calculate_fc_input(input_shape), 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self.apply(self.init_weights)

    def _calculate_fc_input(self, input_shape):
        with torch.no_grad():
            return np.prod(self.conv(torch.zeros(1, *input_shape)).shape)

    def init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            torch.nn.init.uniform(m.weight, -0.01, 0.01)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), actions, rewards, np.array(next_states), dones

    def __len__(self):
        return len(self.buffer)


def epsilon_greedy_policy(state, epsilon, model, num_actions, last_flap, counter):
    if random.random() < epsilon:
        return 0 if random.random() < RAND_ZERO else 1
    else:
        if counter - last_flap < PATIENCE:
            return 0
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax(dim=1).item()
        return action


def save_agent(model, target_model, optimizer, replay_buffer, epsilon, episode, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'target_model_state_dict': target_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epsilon': epsilon,
        'episode': episode
    }
    torch.save(checkpoint, filepath)
    print(f"Agent saved to {filepath}")


def thresholding(img, threshold_value=128):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, img_thresholded = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)
    return img_thresholded


def dilation(img, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    img_dilated = cv2.dilate(img, kernel, iterations=1)
    return img_dilated


def erosion(img, kernel_size=(5, 5)):
    kernel = np.ones(kernel_size, np.uint8)
    img_eroded = cv2.erode(img, kernel, iterations=1)
    return img_eroded


def background_removal(current_frame, prev_frame, threshold=30):
    prev_gray = prev_frame.convert('L')
    prev_gray_np = np.asarray(prev_gray, dtype=np.uint8)
    current_gray_np = np.asarray(current_frame, dtype=np.uint8)

    frame_diff = cv2.absdiff(current_gray_np, prev_gray_np)

    _, diff_thresh = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    return diff_thresh


def preprocess_frame(img, prev_frame=None, resize=(IMG_SIZE, IMG_SIZE), threshold_value=128, kernel_size=(5, 5)):
    img_gray = img.convert('L')

    img_resized = img_gray.resize(resize)

    if prev_frame is not None:
        prev_resized = prev_frame.resize(resize)
        img_resized = background_removal(img_resized, prev_resized, threshold=5)

    img_resized_np = np.asarray(img_resized, dtype=np.float32)

    _, img_thresholded = cv2.threshold(img_resized_np, threshold_value, 255, cv2.THRESH_BINARY)

    img_dilated = cv2.dilate(img_thresholded, np.ones(kernel_size, np.uint8), iterations=1)

    img_eroded = cv2.erode(img_dilated, np.ones(kernel_size, np.uint8), iterations=1)

    img_normalized = img_eroded / 255.0

    img_normalized = np.expand_dims(img_normalized, axis=0)

    return img_normalized


def create_model_folder(models_dir):
    model_id = len([f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))])
    model_folder = os.path.join(models_dir, f"model_{model_id + 1}")
    os.makedirs(model_folder)
    return model_folder


def save_model_checkpoint(model, target_model, optimizer, replay_buffer, epsilon, episode, model_folder,
                          is_final=False):
    if is_final:
        model_path = os.path.join(model_folder, "final_model.pth")
    else:
        model_path = os.path.join(model_folder, f"model_episode_{episode}.pth")

    save_agent(model, target_model, optimizer, replay_buffer, epsilon, episode, model_path)

    print(f"Model saved to {model_path}")


def train_dqn(model, target_model, buffer, optimizer):
    global step_count
    if len(buffer) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

    q_values = model(states).gather(1, actions)
    with torch.no_grad():
        next_action = model(next_states).argmax(1, keepdim=True)
        next_q_values = target_model(next_states).gather(1, next_action)
        target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    loss = nn.MSELoss()(q_values, target_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step_count += 1
    scheduler.step()


env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False, background=None)

input_shape = (4, IMG_SIZE, IMG_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(input_shape, 2).to(device)
target_model = DQN(input_shape, 2).to(device)
target_model.load_state_dict(model.state_dict())

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = LambdaLR(optimizer, lr_lambda=lambda step: adaptive_lr_schedule(step))
replay_buffer = ReplayBuffer(BUFFER_SIZE)

epsilon = EPSILON_START

models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

current_model_folder = create_model_folder(models_dir)
print(f"Current model folder: {current_model_folder}")

best_model_path = os.path.join(models_dir, "best.pth")


def handle_interrupt(signal, frame):
    print("Interrupt received, saving model and exiting...")
    save_agent(model, target_model, optimizer, replay_buffer, epsilon, episode, "flappy_bird_agent.pth")
    sys.exit(0)


def crop_image(image, top=0, left=0, bottom=0, right=0):
    width, height = image.size
    return image.crop((left, top, width - right, height - bottom))


signal.signal(signal.SIGINT, handle_interrupt)
max_reward = -100
prev_frame = None
for episode in range(NUM_EPISODES):
    observations, info = env.reset()
    action = 0
    counter = 0
    last_flap = -10
    observations, reward, done, truncated, info = env.step(action)
    frame = env.render()
    img = Image.fromarray(frame)
    img = crop_image(img, bottom=105)
    state = preprocess_frame(img, prev_frame)
    state = np.concatenate([state] * 4, axis=0)
    done = False
    total_reward = 0
    prev_frame = img

    while not done and total_reward < 1000:
        action = epsilon_greedy_policy(state, epsilon, model, 2, last_flap, counter)
        if action == 1:
            last_flap = counter
        observation, reward, done, truncated, info = env.step(action)
        frame = env.render()
        img = Image.fromarray(frame)
        img = crop_image(img, bottom=105)
        next_state = preprocess_frame(img, prev_frame)
        prev_frame = img
        # next_state_squeezed = np.squeeze(next_state)
        # next_state_display = (next_state_squeezed * 255).astype(np.uint8)
        next_state = np.concatenate([state[1:], next_state], axis=0)

        # cv2.imshow("Next State", next_state_display)
        # cv2.waitKey(1)
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        counter += 1

        train_dqn(model, target_model, replay_buffer, optimizer)

    train_dqn(model, target_model, replay_buffer, optimizer)
    if episode % TARGET_UPDATE_FREQ == 0:
        target_model.load_state_dict(model.state_dict())
    if max_reward < total_reward:
        max_reward = total_reward
        target_model.load_state_dict(model.state_dict())
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    print(
        f"Episode {episode}: Total Reward: {total_reward} Lr: {scheduler.get_last_lr()[0]} Epsilon: {epsilon} Frames: {counter}")
    if episode % 100 == 0:
        save_model_checkpoint(model, target_model, optimizer, replay_buffer, epsilon, episode, current_model_folder)

env.close()
save_model_checkpoint(model, target_model, optimizer, replay_buffer, epsilon, NUM_EPISODES, current_model_folder,
                      is_final=True)
