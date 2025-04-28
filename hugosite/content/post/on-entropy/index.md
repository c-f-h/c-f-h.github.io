---
title: 'On Entropy'
date: 2025-04-23T20:57:00+02:00
tags: ['ML', 'RL', 'Math']
categories: ['Connect-Zero']
---

[The last time]({{< relref policy-collapse >}}), we ran our first self-play training loop
on a simple MLP model and observed catastrophic policy collapse. Let's first understand
some of the math behind what happened, and then how to combat it.

## What is entropy?

Given a probability distribution \(p=(p_1,\ldots,p_C)\)
over a number of categories \(i=1,\ldots,C\), such as the distribution over the columns our
Connect 4 model outputs for a given board state, entropy measures the "amount of
randomness" and is defined as[^1]

\[
    H(p) = -\sum_{i=1}^{C} p_i \log p_i.
\]

[^1]: Note on notation: We refer to an entire distribution by \(p\) and to an individual
probability within it with a subscript, e.g., \(p_i\).


A few examples will help here. Let's first look at the "most random"
distribution where every category has chance \(p_i=1/C\), then

\[
    H(p) = -C \frac1C \log \frac1C = \log C.
\]

For our case with \(C=7\) possible moves, this means the uniform random distribution has
entropy \(H(p) \approx 1.946\ldots\)

<figure>
<svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="60" y="60" width="40" height="100" fill="#4CAF50" />
  <rect x="120" y="65" width="40" height="95" fill="#4CAF50" />
  <rect x="180" y="62" width="40" height="98" fill="#4CAF50" />
  <line x1="40" y1="160" x2="240" y2="160" stroke="#555" stroke-width="2" />
</svg>
<svg width="300" height="200" xmlns="http://www.w3.org/2000/svg">
  <rect x="60" y="147" width="40" height="13" fill="#4CAF50" />
  <rect x="120" y="0" width="40" height="160" fill="#4CAF50" />
  <rect x="180" y="138" width="40" height="22" fill="#4CAF50" />
  <line x1="40" y1="160" x2="240" y2="160" stroke="#555" stroke-width="2" />
</svg>
<figcaption>
<p>A high-entropy (left) and a low-entropy (right) probability distribution
  with three categories.</p>
</figcaption>
</figure>

On the other hand, the "least random", fully deterministic distribution which 
always chooses the same of the \(C\) labels with probability 100% has

\[
    H(p) = -1\log 1 = 0.
\]

These are the two possible extremes: "fully random" with maximum entropy \(\log C\) or fully deterministic with minimum entropy \(0\). All other distributions fall somewhere
in between in terms of their entropy.

## In the context of our policy model

While playing a number of games, our model encountered a sequence of \(n\) board states
\((s_k)\). In each state the model
produced a probability distribution \(p(s_k)\) over the seven possible moves, each of
which has an associated entropy \(H(p(s_k))\). (Recall that the model actually outputs
raw logits, and we obtain the probabilities \(p(s_k)\) by applying the softmax operator.)

The plotted values in the entropy plot
were simply the averaged values of the entropies in all board states it encountered since
the last data point,

\[
    \frac 1n \sum_k H(p(s_k)).
\]

This lets us estimate the amount of randomness the model exhibits on average.

Note that the
model can have very different entropies in different situations: ideally, if there is a
winning move on the board, the model should play it with very low entropy, but in the
beginning of the game, it might play more randomly.

Referring back to the entropy plot from
[the previous post]({{< relref policy-collapse  >}}), we observe that our model started out
pretty close to a "fully random" state since at the beginning, the average entropy was
above 1.8, which is close to the maximum possible entropy of 1.946... we saw above. This is
due to the way PyTorch initializes our tensors by default. By the end of training,
entropy had dropped to almost zero, which means the model was moving essentially
deterministically in every situation.

This means that the model didn't have any chance to learn from new states because the
optimizer single-mindedly pushed the probability for the "known winning" moves higher and
higher until no randomness was left. We need to find a way to keep entropy higher for longer
so that our model has plenty of time to experiment. Once the model has a strong
grasp on the required strategies and tactics, we can allow entropy to drop so that it
can consistently play the strongest moves. This is referred to as the balance between
**exploration and exploitation** in RL.

Let's examine some possible strategies for introducing additional randomness to our model.

## Ideas to increase entropy

### Dropout
This might seem like it could work, but in practice it simply doesn't do the job. It does
introduce some extra variation during training, but in eval mode, which we use to sample
new games, dropout is usually disabled, so it wouldn't help at all to create a wider
variety of games. We could keep the dropout enabled throughout, but that runs into its own
issues since the probability distribution we use to sample games is then not the same we
use during the RL policy update. This violates the so-called **"on-policy"** nature of the
REINFORCE algorithm which requires that sampling is done using the same policy as training.
Also, the added variance from dropout is probably not strong enough to encourage consistent
exploration.

### Temperature
Temperature is a standard way to increase the amount of randomness when sampling from a probabilistic model. The idea is simply to divide the raw logits from the model output by a parameter \(T\) before passing them through softmax. For \(T>1\), this pushes the logits closer to 0 and therefore brings the probabilities closer to another. However, in addition to running into the off-policy problem again if we do this only during the sampling period, it also can't fully deal with the situation where the model has already collapsed into a deterministic state since temperature doesn't do much if the raw logits already have wildly different orders of magnitude.

### \(\varepsilon\)-greedy sampling
This is a common technique in other branches of RL. The idea is to introduce a probability epsilon, \(\varepsilon>0\), which describes the chance that during sampling a move, you completely ignore the model outputs and just sample a random move from a uniform distribution (i.e., every move with the same chance) instead.

Sadly, since we are using an on-policy algorithm, this doesn't really work for us. It can force a move which the model wanted to play with probability \(p_a(s)\) essentially zero, which means that the policy gradient

\[
  \nabla \ell = -\frac{G \nabla p_a(s)}{p_a(s)}
\]

can become very large, destabilizing training. (Refer back to the post on the
[REINFORCE algorithm]({{< relref the-reinforce-algorithm >}})
if you want to recall where this fomula came from.)

### Entropy regularization
The idea here is to make sure entropy doesn't drop too far in the first
place. The way to do this is straightforward: if we want higher entropy, why
don't we simply tell the optimizer about it by including it in our loss function? In other
words, we choose a parameter \(\beta>0\), called the entropy bonus, and modify our loss
function by including an extra term which is just the scaled entropy \(H(p(s))\) at the
current state \(s\),

\[
    \ell = -G \log p_a(s) - \beta H(p(s)).
\]

Note the sign: since we minimize the loss, larger entropy \(H\) is rewarded more strongly
if \(\beta\) is larger.
This additional term is minimal for a uniform distribution and penalizes all
other distributions.
This introduces a new hyperparameter \(\beta\) which we need to
tune carefully, but it's worth the effort to avoid the collapse we observed.

This is the approach we'll use going forward.

## Bonus math: gradient computation

What effect does the entropy bonus have on the loss gradient? We can easily compute by the
product rule that

\[
    \frac{\partial H(p)}{\partial p_i} = -(1 + \log p_i).
\]

Therefore, by the chain rule, the gradient with respect to the model parameters is

\[
    \nabla H(p(s)) = -\sum_{i=1}^C (1 + \log p_i(s)) \nabla p_i(s),
\]

and our overall modified loss gradient becomes

\[
    \nabla \ell =
    -G \frac{\nabla p_a(s)}{p_a(s)} + \beta \sum_{i=1}^C (1 + \log p_i(s)) \nabla p_i(s).
\]

Our previous gradient was active only for the particular probability at entry \(a\) we
wanted to reinforce or weaken with the return \(G\),
but now we have contributions for all entries of the
probability distribution. Since \(1 + \log p_i < 0 \Leftrightarrow p_i < \frac1e\), where
\(1/e\approx 0.368\), probabilities lower than roughly 0.368 are pushed upwards, whereas
probabilites above 0.368 are pushed downwards by this gradient contribution.

There are some
additional mathematical subtleties since the probabilities have to remain normalized to 1,
but the most important thing to take away is that this gradient term acts as a barrier
against probabilities dropping to zero.


## Outlook

All right, this was a bit more of a math-heavy post; [next time]({{< relref entropy-regularization >}}) we'll implement the
entropy bonus and run another experiment.
