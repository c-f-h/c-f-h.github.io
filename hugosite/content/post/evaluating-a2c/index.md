---
title: 'Evaluating A2C versus REINFORCE with baseline'
date: 2025-05-11T11:56:00+02:00
tags: ['ML', 'RL']
categories: ['Connect-Zero']
---

With our [implementation of A2C]({{< relref implementing-a2c >}}) ready to go, let's see
it in action.

{{< examplecode "connect-zero" "train/example4-a2c.py" >}}

## The setup

Let's set the ground rules:

- This time we don't do self-play, but instead a pre-training run where we have our
  model face off against the [``RandomPunisher``]({{< relref random-punisher >}}).
- We use AdamW as the optimizer with a learning rate of \(10^{-4}\).
- Value loss weight is \(\alpha = 1.5\); this is higher than the common choice of 0.5
  since we are pretraining and want the model to learn good values quickly.
- [Entropy bonus]({{< relref entropy-regularization >}}) is \(\beta = 0.05\), as before.
- The [reward discount rate]({{< relref the-reinforce-algorithm >}}#assigning-returns)
  is \(\gamma=0.9\). This is on the lower end of common choices and makes the model pay
  most attention to the endgame, which is exactly what we want when training tactics.
- Batch size is 50 games, after which the model is updated.
- As before, each tick on the x-axis of the training plots represents 1000 games played.

## The results

### A2C

Here is the training plot of the run, up until around 200k games played:

{{< figure
    src="pretrain1-a2c.png"
    width="100%"
    alt="Training progress for 200k games with A2C."
    caption="Training progress for 200k games with A2C."
>}}

We can immediately see that this was a very smooth ride with very little noise.
The win rate increases steadily from basically 0 to the 50% mark at around 80k games,
and then further rises to 80% at 200k games.

The entropy bonus did its job and kept the entropy stable at around 0.75 after an
initial drop.

The value network started learning useful information almost immediately, with value loss
dropping consistently over the run from 0.2 to 0.06.
As a result, the standard deviation of the advantage also kept falling: after around 50k
games, it was already lower than that of the raw rewards and continued improving.
This is exactly the variance reduction we want to see as it gives us more stable policy
gradients.

### REINFORCE with baseline

For comparison, let's run the exact same test also with
[REINFORCE with baseline]({{< relref reinforce-with-baseline >}}).
That is, we only change the ``BOOTSTRAP_VALUE`` parameter to ``False`` and leave all other options
unchanged. Here are the training plots:

{{< figure
    src="pretrain1-rwb.png"
    width="100%"
    alt="Training progress for 200k games with REINFORCE with baseline."
    caption="Training progress for 200k games with REINFORCE with baseline."
>}}

At first glance, the results look similar: the win rate has a similar trend, even crossing the
50% mark earlier than with A2C. However, above around 60%, progress noticeably slows down,
and the final win rate after 200k games is closer to 75%.

The value loss is clearly much higher and also needs around 50k games to even start decreasing.
As a result, the variance reduction for the advantage is much weaker here, reducing the standard
deviation of 0.55 for the returns only to around 0.48 for the advantage.

This higher variance in the advantage also leads to a noisier policy loss.

All this aligns with our understanding: the baseline learning is much slower due to the high
variance of the
[Monte Carlo sampling]({{< relref  reinforce-with-baseline >}}#monte-carlo-sampling).
As a result, the model has a poorer value estimate, which
in turn makes it harder for it to improve its policy.

There is one confounding factor: despite using the same entropy bonus, entropy is somewhat
higher here at around 1.1. It's possible that letting entropy decay further could strengthen
exploitation and therefore improve the performance.
Nevertheless, the worse value approximation will remain a significant drawback.



## The shootout

Let's see some actual competition! Using the
[``tournament.py``](https://github.com/c-f-h/connect-zero/blob/main/train/tournament.py)
script from the repo, we can run a round-robin tournament where each model plays
2000 games (1000 going first and 1000 going second) against every other one.
As described above, each model was trained in 200k games against the ``RandomPunisher``.

Below are the resulting win rates. 
Each cell shows the win rate of the left model against the top model.
There are very few drawn games, as adding up the numbers shows.

|                            | vs A2C  | vs RwB | vs RP   |
|----------------------------|:-------:|:------:|:-------:|
| **A2C**                    |   ---   | 55.9%  | 74.4%   |
| **REINFORCE w/baseline**   | 42.1%   |   ---  | 71.5%   |
| **RandomPunisher**         | 21.8%   | 27.2%  |   ---   |

The results confirm our observations: the A2C trained model has higher win rates both
against its RwB competitor as well as against the ``RandomPunisher``.

This would be a good point to switch to self-play, perhaps reduce the value loss weight
and increase the reward discount rate to start learning longer-horizon strategies. The
superior value function of the A2C model serves as an excellent starting point for that.
