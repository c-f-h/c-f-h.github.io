---
title: 'Connect-Zero: Reinforcement Learning from Scratch'
date: 2025-04-20T13:12:41+02:00
tags: ['ML', 'RL']
categories: ['Connect-Zero']
---

For a long time I've wanted to get deeper into reinforcement learning (RL), and the project
I finally settled on is teaching a neural network model
how to play the classic game **Connect 4** ([pretty sneaky, sis!](https://www.youtube.com/watch?v=KN3nohBw_CE)).
Obviously, the name "Connect-Zero" is a cheeky nod to AlphaGo Zero
and AlphaZero by DeepMind.
I chose Connect 4 because it's a simple game everyone knows how to play where we can
hope to achieve good results without expensive hardware and high training costs.

{{< figure src="screenshot.png" alt="A screenshot of a Connect 4 game"
  width="200" align="center" >}}

Some ground rules:
- The neural network model gets nothing but the current board state as a 6x7 grid and
  outputs a classifier over the seven columns for which move to play next.
- Like AlphaZero, it doesn't train on any human games, but only on computer-generated ones.
- Unlike AlphaZero, it doesn't do any tree seach to find the best moves, for multiple
  reasons:
  for simplicity, and because I was curious to see how far you could get with a
  simple move predictor. In fact, there's precedent to show that even for Chess, a vastly
  more complicated game, this can work well: see the DeepMind paper
  [Grandmaster-Level Chess Without Search](https://arxiv.org/pdf/2402.04494v1), which
  trained a pure next-move predictor for Chess. However, they used Stockfish as a teacher
  for their model, rather than self-play.
  I also strongly suspect that Connect 4 is shallow enough that tree search might
  trivialize it, and that wouldn't leave much room for any interesting RL.
- Build it from scratch in PyTorch, without relying on any existing RL framework. The
  point is to learn about the techniques more than just getting to the end.

You can actually [play against a version]({{<relref connect-4>}}) of the current model
right now! It will take a couple of blog posts until we catch up to that level, though.
