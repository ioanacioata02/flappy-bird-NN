import gymnasium
import flappy_bird_gymnasium
import numpy as np
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import cv2
import signal
import sys
import os
from scipy.ndimage import label
from torch.optim.lr_scheduler import LambdaLR

BUFFER_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.95
EPSILON_START = 0.5
EPSILON_END = 0.0
EPSILON_DECAY = 0.995
LEARNING_RATE = 1e-4
TARGET_UPDATE_FREQ = 50
NUM_EPISODES = 10
RAND_ZERO = 0.95
IMG_SIZE = 128
import random
import numpy as np
import torch

step_count = 0


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


input_shape = (4, IMG_SIZE, IMG_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DQN(input_shape, 2).to(device)
target_model = DQN(input_shape, 2).to(device)
target_model.load_state_dict(model.state_dict())


def crop_image(image, top=0, left=0, bottom=0, right=0):
    width, height = image.size
    return image.crop((left, top, width - right, height - bottom))


def load_model(model_path, input_shape=input_shape, num_actions=2):
    print(f"Loading model from: {model_path}")
    model = DQN(input_shape, num_actions).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"Model loaded successfully from: {model_path}")
    return model


def run_game_with_model(model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array", use_lidar=False, background=None)

    model = load_model(model_path)
    for _ in range(30):
        total_reward = 0
        observation, _ = env.reset()
        done = False
        prev_frame = None
        frame = env.render()
        img = Image.fromarray(frame)
        frame_np = np.array(frame)
        img = crop_image(img, bottom=105)
        state = preprocess_frame(img, prev_frame)
        state = np.concatenate([state] * 4, axis=0)
        counter = 0
        last_flap = 0
        pipes = 0
        while not done:  # and total_reward < 1000:
            frame = env.render()
            img = Image.fromarray(frame)
            frame_np = np.array(frame)

            try:
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                cv2.imshow("Flappy Bird - Original Frame", frame_bgr)
            except Exception as e:
                print(f"Error displaying frame: {e}")

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            img = crop_image(img, bottom=105)
            next_state = preprocess_frame(img, prev_frame)
            next_state = np.concatenate([state[1:], next_state], axis=0)
            state = next_state
            prev_frame = img

            if counter - last_flap < 8:
                action = 0
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = model(state_tensor).argmax(dim=1).item()
            if action == 1:
                last_flap = counter
            observation, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if reward == 1:
                pipes += 1
            counter += 1
        print(f"reward {total_reward} pipes {pipes}")
        env.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_game_with_model('./flappy_bird_agent6.pth')
