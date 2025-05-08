---
title: 'Actor-Critic Algorithms'
date: 2025-05-08T21:56:00+02:00
tags: ['ML', 'RL', 'Math']
categories: ['Connect-Zero']
---

After [implementing and evaluating]({{< relref implementing-rwb >}}) REINFORCE with baseline,
we found that it can produce strong models, but takes a long time to learn an accurate
value function due to the high variance of the
[Monte Carlo samples]({{< relref reinforce-with-baseline >}}#monte-carlo-sampling).
In this post, we'll look at Actor-Critic methods, and in particular the
Advantage Actor-Critic (A2C) algorithm[^1],
a synchronous version of the earlier Asynchronous Advantage Actor-Critic (A3C) method,
as a way to remedy this.

[^1]: These methods have quite a long history and have evolved over time, and it's often hard to
pin down exact references.
Richard Sutton's PhD thesis (1984) and the book by
Sutton and Barto (1998), *"Reinforcement Learning: An Introduction,"* already describe
actor-critic methods with TD learning.
In the deep learning era, these ideas could then be applied to much more complex problems.
The A3C (asynchronous advantage actor-critic) method was introduced by
Mnih et al. (2016),
["Asynchronous Methods for Deep Reinforcement Learning"](https://proceedings.mlr.press/v48/mniha16.html).
A2C was then [popularized by OpenAI](https://openai.com/index/openai-baselines-acktr-a2c/) and others as a
synchronous, simpler and often more efficient variant of A3C.

Before we start, recall that we introduced a [value network]({{< relref reinforce-with-baseline >}}#the-value-network) as a component of our model; this remains the same for A2C, and in fact
we don't need to modify the network architecture at all to use this newer algorithm.
Our model still consists of a residual CNN backbone, a policy head and a value head.
The value head serves as the "critic," whereas the policy head is the "actor".

## Bootstrapping value

In our previous algorithms, we used the discounted return \(G\) for the move \(a\) we made in
state \(s\) in order to encourage or discourage similar actions in the future.
For a winning game, \(G\) slowly increased to a final value of \(\pm1\) for a winning/losing game.
This gave us a general idea of which moves led to wins and losses, but did not assign
precise rewards to individual moves.

Say the action \(a\) has left us in a new state \(s'\). **Temporal Difference (TD) learning**
is the idea that we already have an estimate \(v(s')\) for the value of this next state
(in our case, from our value head) and can use it to update our knowledge of \(v(s)\).

Suppose that the move \(a\) gave us an immediate reward \(R\); in our case, \(R\) will
usually be 0, except if the move won the game, in which case \(R=1\).
We also again use a reward discount rate \(\gamma\le1\) which balances immediate against
future rewards. Then one-step temporal differencing says that we should update
our value function via

\[
    v(s) \gets R + \gamma v(s').
\]

In other words, the target value for the current state is the immediate reward \(R\) plus the
discounted value of the next state \(v(s')\).
The symbol \(\gets\) shouldn't be taken too literally:
it just means that we will use a gradient update (via a squared error loss function) to pull
\(v(s)\) towards the value of the right-hand side.[^2]
To achieve that we'll replace the value loss function with

\[
    \ell^{\textsf{TD}} = \alpha \left(R + \gamma v(s') - v(s)\right)^2,
\]

where \(\alpha > 0\) is again a weighting hyperparameter. We have to read this carefully,
though: the gradient should only be taken for the \(v(s)\) part; the target value
\(v(s')\) should be treated as constant for the purpose of the gradient step.

[^2]: In classical [TD learning](https://en.wikipedia.org/wiki/Temporal_difference_learning)
algorithms, these updates were actually applied to values in a table directly,
modified only by a learning rate.

So it's a slightly weird idea because we use the value function \(v(\cdot)\) to essentially
construct itself.
This is known as **[bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics))**,
much like Baron Munchausen [pulling himself, horse and all, out of a swamp](https://en.wikipedia.org/wiki/Bootstrapping#/media/File:Zentralbibliothek_Solothurn_-_M%C3%BCnchhausen_zieht_sich_am_Zopf_aus_dem_Sumpf_-_a0400.tif)
(although he pulled himself out by his pigtail rather than his bootstraps).

In practice, for our Connect 4 setting where rewards are only assigned at the end of a game, this
means that the method learns "backwards in time": it first has to learn the value of
terminal game states and can then transport this information backwards to earlier states through
the TD mechanism.

There is one edge case we have to treat specially, namely if \(s'\) is a terminal state because
the game ended in a win or draw. Our value head never sees such states during training, but
we can simply fix their values as \(v(s') = 0\) because no further rewards are
possible in a terminal state.

## Bias and variance

In our [earlier discussion]({{< relref reinforce-with-baseline >}}#monte-carlo-sampling), we
concluded that Monte Carlo sampling (as used in REINFORCE) has high variance, but low bias.
For bootstrapping, the situation is reversed: the estimate has **low variance** because it
depends only on the value of the successive state, not the entire trajectory, but **high bias**
because it relies on the current estimate of the value function which may still be far from
accurate.

In terms of credit assignment, this approach has clear advantages: instead of ramping
up smoothly over the duration of a game, the bootstrapped valuecan more sharply distinguish key
decision points such as blunders or forced wins.
If the next state \(s'\) leaves our opponent with a clear winning
move, then this should be reflected in a \(v(s')\) close to -1 and therefore pull the valuation
for \(v(s)\) towards -1 as well.

Another way to look at it is that this bootstrapping technique uses one step of lookahead to
improve the value estimate. There are more advanced techniques which use several time steps
or blend them with Monte Carlo estimates to achieve a tradeoff of variance and bias,
but for now we stick to this simplest variant.

## Updating the policy network

The update of the policy network (or "actor" in the actor-critic nomenclature) works
similarly to [REINFORCE with baseline]({{< relref reinforce-with-baseline >}}#using-value-to-estimate-advantage) by computing an advantage \(A\) and using that to
weight the log loss. However, we use a different formula for the advantage:

\[
    A = R + \gamma v(s') - v(s).
\]

Note that this is just the error term of the value update and comes with the same
tradeoffs: by replacing the observed discounted return \(G\) from the baseline version
with the bootstrapped value estimate \(R + \gamma v(s')\),
it has higher bias due to a potentially inaccurate value, but the variance is lower,
leading to further variance reduction compared to REINFORCE and its baseline variant.

The total loss function for A2C, including an
[entropy bonus]({{< relref entropy-regularization >}}) to encourage exploration, is then

\[
    \ell^{\textsf{A2C}} = \underbrace{-A \log p_a(s)}_{\textsf{policy loss}}
    + \underbrace{\alpha (R + \gamma v(s') - v(s))^2}_{\textsf{value loss}}
    - \underbrace{\beta H(p(s))}_{\textsf{entropy bonus}}.
\]

To summarize, this version of A2C combines one-step bootstrapping from TD learning with a
baseline-adjusted policy update.
It combines the strengths of REINFORCE with baseline (via advantage estimation) and TD learning
(lower-variance updates).
In the next installment of the series, we'll implement the A2C method for our Connect 4 agent.
