#####################
### AI PARAMETERS ###
#####################
alpha = 0.1
gamma = 0.99
episilon = 1.0
episilon_min = 0.05
episilon_decay = 0.99
episodes = 300_000

import gymnasium as gym
from time import sleep
import pygame
import numpy as np
import random
import os
import csv

pygame.init()
pygame.font.init()
clock = pygame.time.Clock()
env = gym.make("Blackjack-v1", natural=True)

# Q learning matrix
Q = np.zeros((32, 11, 2, 2))

obs, info = env.reset()
done = False
running = True

reward = 0
blackjackPossible = True

def new_game():
    global obs, done, reward, blackjackPossible 
    blackjackPossible = True
    obs, _ = env.reset()
    player_sum, dealer_card, usable_ace = obs

    done = False
    reward = 0

def play(action):
    global obs, done, reward

    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    if done:
        if reward == 1.5:
            print("BLACKJACK! Win +1.5")
        elif reward == 1:
            print("Win +1")
        elif reward == -1:
            print("Lose -1")
        elif reward == 0:
            print("Push +0")

def state_index(obs):
    player_sum, dealer_card, usable_ace = obs
    return (int(player_sum), int(dealer_card), int(usable_ace))

def choose_action(state, episilon):
    if random.random() < episilon:
        return env.action_space.sample()

    return int(np.argmax(Q[state]))

def train():
    global episilon

    with open("blackjack_training.csv", "w", newline="") as file:
        writer = csv.writer(file)

        # Write headers ONCE
        writer.writerow([
            "Episode",
            "Reward",
            "Epsilon",
            "Average_Reward_1000"
        ])

        # Keep recent rewards across episodes
        recent_rewards = []

        for episode in range(episodes):
            obs, info = env.reset()
            state = state_index(obs)

            terminated = False
            truncated = False
            episode_reward = 0.0

            while not (terminated or truncated):
                action = choose_action(state, episilon)

                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = state_index(next_obs)

                if terminated or truncated:
                    target = reward
                else:
                    target = reward + gamma * np.max(Q[next_state])

                Q[state + (action,)] += alpha * (
                    target - Q[state + (action,)]
                )

                episode_reward += reward
                state = next_state

            # Decay exploration
            episilon = max(
                episilon_min,
                episilon * episilon_decay
            )

            # Update rolling average
            recent_rewards.append(episode_reward)

            if len(recent_rewards) > 1000:
                recent_rewards.pop(0)

            average_reward = sum(recent_rewards) / len(recent_rewards)

            # Save episode results
            writer.writerow([
                episode + 1,
                episode_reward,
                episilon,
                average_reward
            ])

            if (episode + 1) % 10000 == 0:
                print(
                    f"Episode {episode + 1}/{episodes} | "
                    f"Reward: {episode_reward:.1f} | "
                    f"Average (last 1000): {average_reward:.4f} | "
                    f"Epsilon: {episilon:.3f}"
                )

    print("Training complete! Results saved to blackjack_training.csv")

train()

while running:
    if done:
        print()
        print("------------------New round---------------------")
        print()
        new_game()

    player_sum, dealer_card, usable_ace = obs

    print()
    print(f"Dealer: {dealer_card}")
    print(f"You:    {player_sum}")

    if usable_ace:
        print("You have a usable ace")

    print("Hit (h), stand (s), train (t), or quit (q): ", end="")
    userInput = input()

    if userInput == "h": # Hit
        blackjackPossible = False
        play(1)

    if userInput == "s": # Stand
        if player_sum == 21 and blackjackPossible == True:
            print("BLACKJACK! ", end="")
        else:
            blackjackPossible = False

        play(0)

    if userInput == "t":
        while not done:
            state = state_index(obs)
            action = int(np.argmax(Q[state]))
            play(action)
    
    if userInput == "q": # Quit
        running = False
        print("Thanks for playing")

    clock.tick(30)

env.close()
pygame.quit()
