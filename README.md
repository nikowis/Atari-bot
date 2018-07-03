# Atari-bot

Machine learning model created to learn playing games from the [Atari 2600 console](https://en.wikipedia.org/wiki/Atari_2600).
The model was learnt mostly on [Atari Breakout](https://en.wikipedia.org/wiki/Breakout_(video_game))
due to time and compute power limitations, but is able to learn multiple games from this console at superhuman level without any changes.

Project was implemented in Python using [Keras](https://keras.io/)
and [Gym](https://gym.openai.com/) libraries.

It might be difficult to run this program on Windows operating systems due to Gym dependencies. 
A simple way to get started with this bot is to use [Google Colaboratory](https://colab.research.google.com) using [this](/scripts/atari.ipynb) Jupyter notebook.

A ready to use model that was learnt on Breakout is available in this repository [here](breakout-model8M). 
It was playing for about 80 hours and is able to break on average 40 blocks.

This project was based on two papers created by [DeepMind Technologies Limited](https://deepmind.com/)
([[1]](https://deepmind.com/research/publications/playing-atari-deep-reinforcement-learning/) 
and [[2]](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning/)).

LaTeX report in polish is available for this project [here](docs/raport/raport.tex).

This project was created for academic purposes.
