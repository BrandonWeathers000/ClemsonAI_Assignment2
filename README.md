# Introduction
This repo is dedicated to storing code for Clemson's CPSC 4420/6420: AI Assignment 2.
It uses OpenAI's gymansium blackjack library to test various AI agents.

This project has the following parts:
1. Playing blackjack in the terminal.
2. Training an tabular Q-learning AI agent to play blackjack.
3. Vizulizing how different Q-learning parameters influence an agent's preformance.

As per the insturctions, each part of the porject is contained in its down directory (i.e., PartB, PartC, ...).
The insturctions and my answers are in the root directory of this project.

# Dependencies
- python
- python libraries
  - gymnasium 
  - numpy
  - matplotlib
  - pandas
  - pygame

# How to Run
The following instructions are a guide to training your own AI agent.
1. Clone this repo
2. cd into ClemsonAI_Assignment2/PartB
3. python BlackjackCLI.py (You may need to set up a python environment depending on your OS.)
   1. If you want to play blackjack, now you can.
   2. If you want the AI to play that round for you, press "t".
   3. If you want to quit playing, press "q".
4. python plotter.py

Now you have a graph showing how well your agent did.
Feel free to change the parameters defined in lines 4-9 in BlackjackCLI.py.

# My Ultimate Agent
In part G, I am tasked to make an agent that preforms as well as possible.
I used the following parameters for my ultimate agent.
- alpha = 0.1
- gamma = 0.99
- episilon = 1.0
- episilon decay  = 0.99
- episilon_min = 0.05
- episodes = 1,000,000

<img src="./PartG/UltimateTest.png" alt="Error title banner" margin-left="auto" margin-right="auto" width="100%">
