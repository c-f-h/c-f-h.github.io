---
title: 'Introducing a Benchmark Opponent'
date: 2025-04-26T09:50:00+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL']
categories: ['Connect-Zero']
---

[Last time]({{< relref entropy-regularization >}}) we saw how the entropy bonus
enables self-play training without running into policy collapse.
However, the model we trained was quite small and probably not capable of very strong play.
Before we dive into the details of an improved model architecture, it would be very helpful
to have a decent, fixed benchmark to gauge our progress.

## A benchmark opponent

The only model with fixed performance we have right now is the ``RandomPlayer`` from the
[basic setup post]({{< relref basic-setup-and-play >}}). Obviously, that's not a challenging
bar to clear. But it turns out that with some small tweaks, we can turn the fully random
player into a formidable opponent for our starter models.

The algorithm is very simple:
1. If there is a winning move on the board, play it.
2. If not, and if the opponent has a winning move, block it.
3. Otherwise, play a random move.

We call this model the ``RandomPunisher`` since, although it doesn't have any concept of
strategy, it will ruthlessly punish tactical mistakes.

Here's an implementation of that idea in PyTorch:

```py
@torch.jit.script
def find_best_move(board):
    """
    Finds a winning move for the current player (represented as +1) in the
    given board state (R, C).
    Then checks for potential winning moves of the opponent and blocks them.
    Otherwise, moves randomly.

    Returns logits, shape (C,).
    """
    cols = board.shape[-1]
    for B in (board, -board):
        for c in range(cols):
            # Check if the move is valid
            if B[0, c] == 0:
                _, win = make_move_and_check(B, c)
                if win:
                    choice = torch.tensor(c, device=board.device)
                    logits = nn.functional.one_hot(choice, num_classes=cols).float()
                    # Avoid log(0) by adding a small epsilon
                    return torch.log(logits + 1e-12)
    return torch.zeros((cols,), dtype=torch.float32, device=board.device)

class RandomPunisher(nn.Module):
    """Plays a winning or blocking move if available, otherwise plays a random move."""
    def forward(self, x):
        # Store original shape and determine batch size
        original_shape = x.shape
        if x.ndim == 2:        # Single board state (R, C)
            x = x.unsqueeze(0) # Add a batch dimension: (1, R, C)

        batch_size = x.size(0)
        logits = torch.stack([find_best_move(x[i]) for i in range(batch_size)])

        if len(original_shape) == 2:
            logits = logits.squeeze(0)    # -> (C,)
        return logits
```

The function ``find_best_move`` is the core of the strategy; it calls the function
``make_move_and_check``, which we already used in our earlier post, to check each valid
move to see if it would result in a win. It does this first for the actual input
``board``, and if no winning move was found, repeats the procedure for ``-board``. This
is the board from the opponent's view and finds any potentially winning moves we have to
block.

For any move it chooses, it applies log to a one-hot encoding (plus some small
number) of that move so that we end up with a logit of 0 for the chosen move and
sufficiently large negative logits for the others. After softmax, this effectively
results in probability 1 for the chosen move.

The actual ``RandomPunisher`` module then simply calls this function in a loop for each
board state in the input batch. This is not the most efficient implementation,
but the use of the ``@torch.jit.script`` decorator, which tells PyTorch to just-in-time
compile the function, goes a long way towards speeding up the generally slow Python loops.

Despite the simplicity of this algorithm, it's challenging for a model which
doesn't yet have a strong grasp of basic tactics to achieve
a consistently positive win rate against this guy.

For instance, I continued the self-play loop for the ``SimpleMLPModel``
from [the last post]({{< relref entropy-regularization >}}),
and despite some attempts at tuning the hyperparameters
(starting with a learning rate of 1e-3, then reducing to 1e-4;
starting with an entropy bonus of 0.05, then decreasing to 0.03), it was difficult
to get consistently beyond a 35% win rate against the ``RandomPunisher``.

Presumably, the simple MLP model lacks the capacity for strong play,
and we need a better design, which is the topic of [the next post]({{< relref model-design >}}).