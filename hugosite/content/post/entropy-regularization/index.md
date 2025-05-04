---
title: 'Entropy Regularization'
date: 2025-04-24T20:15:00+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL']
categories: ['Connect-Zero']

connect4: true
---

Based on our [discussion on entropy]({{< relref on-entropy >}}), our plan is to implement
entropy regularization via an entropy bonus in our loss function.

> [!important] Example Code
> Runnable example code for this post:  
> [``connect-zero/train/example2-entropy.py``](https://github.com/c-f-h/connect-zero/blob/main/train/example2-entropy.py).


## Implementing the entropy bonus

The formula for entropy which we have to implement,

\[
    H(p) = -\sum_{i=1}^{C} p_i \log p_i,
\]

is simple enough: multiply the probabilities for the seven possible moves with their
log-probabilities, sum and negate. However, there is one numerical problem we
have to worry about: masking out an illegal move \(i\) leads to a zero probability
\(p_i=0\) and a log-probability \(\log p_i = -\infty\). However, due to the rules of
IEEE 754 floating point numbers, multiplying zero with \(\pm\infty\) is undefined and
therefore results in NaN (not a number). For the entropy formula, however, the
contribution should be 0.

We could explicitly check for NaNs, but a simple
and very common workaround is to set the logits (i.e., the pre-softmax model outputs)
for illegal moves not to \(-\infty\), but instead
add a large negative number, say ``-1e9``, to them. This produces tiny probabilities which
won't disturb our result in any meaningful way; the resulting entropy contributions
are essentially zero.

In practice, this means that the function masking invalid moves could look like this:

```py
def mask_invalid_moves_batch(boards, logits, mask_value=-torch.inf):
    # check which top rows are already filled, and mask out those logits
    illegal_mask = torch.where(boards[:, 0, :] == 0, 0.0, mask_value)
    return logits + illegal_mask
```

The tensor ```illegal_mask``` is 0 for valid moves and ``mask_value`` for invalid ones.
To actually implement the entropy bonus, we modify the ``update_policy`` function from
the [REINFORCE algorithm]({{< relref the-reinforce-algorithm >}}) as follows:

```py
    # mask out illegal moves and compute logprobs of the actually taken actions
    # use finite mask instead of -inf to avoid nans in entropy
    masked_logits = mask_invalid_moves_batch(states, policy_logits_batch, mask_value=-1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)                # (B, C)
    entropy = -(log_probs * torch.exp(log_probs)).sum(1)           # (B,)

    log_probs_taken = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze()
```

Note that since we already have the log-probs, we just apply ``exp`` to them to get
the actual probabilities. Finally the loss is modified as follows:

```py
    policy_loss = -(returns * log_probs_taken).sum()          # as before
    total_loss = policy_loss - ENTROPY_BONUS * entropy.sum()  # add entropy term
```

Then we call ``total_loss.backward()`` for backpropagation, with the remaining code
unchanged.

Here ``ENTROPY_BONUS`` is a hyperparameter which we set to 0.05 for now.
You'll need to tune this by monitoring your actual entropy plot.
In our case, the choice 0.05 results in high but still reasonable entropy for early
exploration of around 1.0. (Recall that this is roughly half the maximum possible entropy.)

## A new training run

With these changes in place, we can start another self-play training run.
We keep the model architecture and the hyperparameters mostly the same as in
[the first one]({{< relref policy-collapse >}}),
except that we increase the interval between model checkpoints from 100 to 400 batches
to get a better look at the dynamics within these cycles.

{{< figure src="no_collapse2.png" alt="Training plots without policy collapse"
  align="center"
  caption="Plots of win rate, entropy, policy loss and game length over a self-play training run with entropy bonus 0.05."
>}}

Again we see the cyclical drops back towards 50% win rate which happen whenever we
reach a new checkpoint, where the model initially starts playing itself.
It then typically starts dominating its previous checkpoint rapidly, which is
a great sign.
Most importantly, this time
there are no signs of collapse; game length varies considerably, and entropy generally
seems to oscillate in the vicinity of 1.0 instead of collapsing to zero thanks to
the entropy bonus \(\beta=0.05\).

## Game analysis

Here is a game the model, which has been trained only for a few minutes, played against
itself. The first thing we notice is that it's much longer at 26 moves.

<div id="game-container" class="connect4-container"
    data-human="-1" data-cpu="-1"
    data-movelist="[5, 4, 1, 1, 3, 2, 1, 2, 3, 3, 2, 2, 1, 1, 1, 2, 4, 4, 4, 2, 3, 5, 0, 4, 0, 3]">
</div>

It's quite instructive to look at some moves together with the probabilities they were
sampled with to get a feeling for how well the model understands various positions.
I use the notation "Y3" for Yellow (the second player) playing in column 3.
Percentages always refer to the chance \(p_a(s)\) of playing that move according to the
softmax output.

- Move 8 (Y3): the top-rated move at 49% (entropy 1.36). The model already seems to favor
  moves which set up several connections (horizontal and vertical at once).
- Move 10 (Y4): the block is returned. Continuing its own column 3 was rated about even with
  both moves at around 34%.
- Move 13 (R2): a very highly rated move at 86% (entropy 0.58). Getting three in a row is
  very attractive to the model.
- Move 14 (Y2): and Yellow does the same with a 72% chance, even putting itself one move
  away from a win!
- Move 15 (R2): a blunder! Red should have blocked column 5. For some reason this
  pointless-looking move has 88% confidence.
- Move 16 (Y3): Yellow blunders back. In its defense, the winning move (5) actually had a
  72% chance, but bad luck strikes and favors this 9% move over it.
- Move 17 (R5): this time Red succeeds in the block, although it's still only 31% sure that
  this is the right thing to do.
- Move 24 (Y5): setting up another possible win with 60% confidence. Red misses it, and
  finally in move 26, Yellow brings it home with 37% confidence.

This isn't stellar play by any means. The model does seem to favor putting several
of its pieces in a row, which is a good sign. On the other hand, it's often completely blind
to immediate one-move threats from its opponent, like in move 15. Then again, immediate
wins seem to be rated relatively highly already, and missing the one in move 16 was just
due to poor luck.

Overall, defense seems to be much harder to grasp than offense,
which perhaps isn't surprising: a winning move ends the game immediately, but
a losing blunder still requires the (currently poor) opponent to capitalize on it.

But what this shows is that entropy regularization is working: the model plays a wide
variety of moves, explores different board states and slowly learns about the game.

It would absolutely be possible to continue training this model and see if it can reach
a decent level of play, but perhaps it's time we first invested a bit more effort into
model design. In [the next post]({{< relref random-punisher >}}), we first design a
benchmark opponent to measure our progress in absolute terms.
