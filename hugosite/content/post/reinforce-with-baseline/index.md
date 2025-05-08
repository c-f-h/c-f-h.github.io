---
title: 'REINFORCE with Baseline'
date: 2025-04-29T08:42:00+02:00
tags: ['Math', 'ML', 'RL']
categories: ['Connect-Zero']
---

[In the previous post]({{< relref model-design >}}), we introduced a stronger model
but observed that it's quite challenging to achieve
a high level of play with [basic REINFORCE]({{< relref the-reinforce-algorithm >}}),
due to the high variance and noisy gradients of the algorithm which often lead to unstable
learning and slow convergence.
Our first step towards more advanced algorithms is a modification called
"REINFORCE with baseline" (see, e.g.,
[Sutton et al. (2000)](https://proceedings.neurips.cc/paper_files/paper/1999/file/464d828b85b0bed98e80ade0a5c43b0f-Paper.pdf)).

## The value network

Given a board state \(s\), [recall that]({{< relref basic-setup-and-play >}}) our model
currently outputs seven raw logits which are then
transformed via softmax into the probability distribution \(p(s)\) over the seven possible
moves. Many advanced algorithms in RL assume that our network also outputs
a second piece of information: the **value** \(v(s)\), a number between -1 and 1 which,
roughly speaking, gives an estimate of how confident the model is in winning from the
current position.

There are different ways to implement this: there could be an entirely separate value
network, or, more commonly, a separate **value head** is attached to the CNN feature layers,
in parallel to the already existing MLP **policy head** which outputs the move logits.[^3]
This way the already computed
features can be reused for estimating the board state value. We'll get into the
technical implementation for our particular model later. For now, the schematic below
should illustrate the general idea.

[^3]: Here "head" refers to a small network module attached to the shared
feature layers that produces a particular kind of output.

```goat {caption="Schematic of two 'heads', one for policy and one for value, being attached to the same base feature layers."}
           +-------------------+
           |       Input       |
           +---------+---------+
                     |
                     v
     +-------------------------------+
     |        Feature layers         |
     +--------+-------------+--------+
              |             |
              v             v
     +-------------+   +-------------+
     | Policy head |   | Value head  |
     +--------+----+   +----+--------+
              |             |
              v             v
     +-------------------------------+
     |    Output: (policy, value)    |
     +--------+-------------+--------+
```


We'll have to answer two questions at this point: how to train this value network,
and how it can help us to improve the RL update.

## Training the value network

Recall that each board state in the REINFORCE policy update is
[already associated]({{< relref the-reinforce-algorithm >}}#assigning-returns)
with a "return" \(G \in [-1,1]\) which gives us some information on how we are doing in
the current board state. Since we used discounted returns, \(G\) increases for
every move in a winning game until a final value of \(+1\), whereas for a losing game
it decreases to a final value of \(-1\).

This is a clear candidate to train our value network on: we can simply use the squared
error \((G - v(s))^2\) as the loss function which, when minimized,
pushes the output \(v(s)\) of the
value network towards the observed return \(G\) for the current state \(s\).
Since we already have a loss function for the policy network,
as well as a second loss term from
[entropy regularization]({{< relref on-entropy >}}#entropy-regularization) which
encourages exploration,
we add this third
term on top of that, scaled with a parameter \(\alpha\), the value loss weight.
A typical value for this hyperparameter is \(\alpha=0.5\).
So our loss function for a single sample becomes

\[
    \ell = \underbrace{-G \log p_a(s)}_{\textsf{policy loss}}
    + \underbrace{\alpha (G - v(s))^2}_{\textsf{value loss}}
    - \underbrace{\beta H(p(s))}_{\textsf{entropy bonus}}.
\]

Minimizing this term via gradient descent should therefore find a balance between
training our policy to make better moves, training the value network to predict the return,
and keeping move entropy sufficiently high.

In this formulation, we are training the value network, but we are not directly using it
to improve the policy network. It's plausible that by attaching it to our feature layers,
we are learning better features and therefore indirectly
slightly improving the policy as well. Nevertheless, we can get more benefits out of the
value network by using it directly during the policy update as well, as we will see below.

### Monte Carlo sampling

The way we obtain value estimates in this version of the algorithm is by sampling many
possible games and using the direct, observed outcome in the form of the return \(G\)
to estimate the value. This is known as a
[Monte Carlo method](https://en.wikipedia.org/wiki/Monte_Carlo_method), named after the
famous casino, where many outcomes are randomly sampled and some form of statistic is
derived from these samples.

A crucial point is that this method has relatively **high variance:**
every new game we sample starting from a fixed state \(s\) can lead to wildly
varying outcomes.
On the other hand, it has **low bias** because we use the true returns (which are only
available after finishing the current game) without any further approximations.
In other words: we need many samples to arrive at a good estimation
of the value (because of the high variance), but once we have collected them, the
estimate of the true value is very accurate (low bias).

[Later in the series]({{< relref actor-critic >}}), we will encounter ways to trade off lower
variance against a higher bias. For now, though, we stick with Monte Carlo sampling.

The high variance in the observed returns \(G\)
is precisely why the basic REINFORCE algorithm suffers from noisy updates.
As we'll see next, using a learned baseline \(v(s)\)
helps mitigate the noisiness of the policy update, even though \(v(s)\)
itself is learned from noisy returns.


## Using value to estimate advantage

The core idea of REINFORCE with baseline is to use the observed **advantage** as the
reward scaling for the policy loss, where the advantage \(A\) is defined as the difference
between the actual return of the current move minus the model's value estimate, i.e.,

\[
    A = G - v(s).
\]

The total loss function for REINFORCE with baseline is therefore the slight modification

\[
    \ell^{\textsf{RwB}} = \underbrace{-(G - v(s)) \log p_a(s)}_{\textsf{policy loss}}
    + \underbrace{\alpha (G - v(s))^2}_{\textsf{value loss}}
    - \underbrace{\beta H(p(s))}_{\textsf{entropy bonus}}.
\]

The factor before the \(\log\) in the policy loss indicates how strongly
the move we made in state \(s\) is rewarded or punished. Let's go through a few examples
to understand how this modified term makes sense.

### Some practical examples on advantage

Assume the model finds itself in a state which it evaluates as roughly even, meaning
it assigns a value \(v(s)\approx0\).
However, the action \(a\) it took in this state left it in a very advantageous position
close to winning, and therefore the actual return \(G\) is close to \(1\).
Then two things happen: first, as discussed above, the value loss will push \(v(s)\)
higher to reflect that there is a good move in the current state \(s\) which is likely
to lead to a win. Second, the move will be rewarded with an advantage \(A\) close to
one to make the model more likely to play this "surprisingly good" move in the future.

On the other hand, if the model thought it was doing well (\(v(s)>0\)) but the move
it chose was a blunder and therefore \(G<0\), the advantage will be negative. In fact,
the model will be punished even more severely for this move than in the basic algorithm
because \(A\) is even more negative than \(G\), which is exactly what we want in order to
discourage this "surprisingly bad" move.

There is one effect that may be counterintuitive at first: if the model is in
a winning position and does play the winning move, then both \(v(s)\) and \(G\)
are close to \(1\) and therefore the advantage \(A\) is roughly zero. This means the
model no longer receives additional rewards for playing the move it already knows wins
the game. But in fact that's fine: the model already knows how to win from here, we
don't need to keep pushing the model output for this move higher and higher. By not
placing a high reward on the obvious thing to do, the gradient signal is stronger
for moves which actually lead to a better than expected outcome.
Also, if the model ever deviates from the winning move, the punishment will be severe
because then \(A\) falls off sharply.

The general idea is that because \(v(s)\) already captures some information about the
expected return
\(G\), typically \(A\) will have smaller fluctuations than the return \(G\) itself.
This is known as **variance reduction** and is exactly what makes the algorithm with
baseline more robust and less noisy compared to basic REINFORCE.

As an example we have encountered in our Connect 4 context, the baseline can help the
model to better learn defensive moves. If the model assigns negative value to three opposing
pieces being in a row because it usually loses from there, then finding the saving move
that blocks the opponent's win will yield a high advantage and reward this move strongly.

Hopefully, this should clarify the basic theory behind REINFORCE with baseline.
In the [next post]({{< relref implementing-rwb >}}), we'll actually implement and evaluate
the algorithm.
