---
title: 'Basic Setup and Play'
date: 2025-04-20T15:35:41+02:00
tags: ['Python', 'PyTorch']
categories: ['Connect-Zero']
---

Let's get into a bit more technical detail on how our
[Connect 4-playing model]({{< relref connect-zero >}})
will be set up, and how a basic game loop works.
Throughout all code samples we'll always assume the standard PyTorch imports:

```py
import torch
import torch.nn as nn
import torch.nn.functional as F
```

## Board state

The current board state will be represented by a 6x7 PyTorch `int8` tensor,
initially filled with zeros.

```py
    board = torch.zeros((ROWS, COLS), dtype=torch.int8, device=DEVICE)
```

The board is ordered such that ``board[0, :]`` is the top row.
A non-empty cell is represented by +1 or -1. To simplify things, we always
represent the player whose move it currently is by +1, and the opponent by -1.
This way we don't need any separate state to keep track of whose move it is.
After a move has been made, we simply flip the board by doing

```py
    board = -board
```

## Model protocol

Any model that wants to play a game of Connect 4 will have to follow a simple protocol:
it takes in the current board state as described above and outputs a `float32` tensor of
seven numbers which represent the probability of playing a move in each of the seven 
columns. A model which takes in a state and outputs a recommended action is called
a **policy model** in RL parlance.

We will keep the model output in raw logits, that is, arbitrary numbers between minus and
plus infinity, with no activation function applied to them.
To convert these to probabilities, we use the
[softmax operator](https://en.wikipedia.org/wiki/Softmax_function) which applies
an exponential function to each number and then normalizes them to add up to 1.
Finally, we choose a move by sampling from the resulting random distribution
\((p_1, p_2, \ldots, p_7)\) over the seven columns.

There is one slight complication: once a column is full, i.e., ``board[0, c] != 0``,
it's no longer valid to play a
move there, and we can't rely on the model to always output zero probabilities for full
columns. So after obtaining the raw logits, we set any of them corresponding to illegal
moves to minus infinity, which will result in a zero probability for that column.

So a simple function to sample a valid move from a model could look like this:

```py
def sample_move(model, board: torch.Tensor) -> int:
    """Sample a random move using the model's output logits."""
    logits = model(board)                       # Get raw logits from model
    illegal_moves = torch.where(board[0, :] == 0, 0.0, -torch.inf)
    logits += illegal_moves                     # Mask out illegal moves
    probs = F.softmax(logits, dim=-1)           # Convert logits to probabilities
    return torch.multinomial(probs, 1).item()   # Sample the distribution
```

For performance and batching, we extend the model protocol to also allow
an entire batch of boards for evaluation, so with a batch size `B` it will then map
tensors of size
```
    (B, 6, 7) -> (B, 7)
```

Probably the simplest possible model you could write is one that just makes random
moves:

```py
class RandomPlayer(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:   # Single board state: (6, 7)
            return torch.zeros((7,))
        else:             # A batch of board states: (B, 6, 7)
            return torch.zeros((x.size(0), 7),)   # -> (B, 7)
```

That's all we need: a vector of constant logits is mapped to constant probabilities by
softmax, and we don't need to worry about illegal moves either because ``sample_move``
takes care of that.

## Playing a game

There is one other function we need, ``make_move_and_check(board, move)``, which
returns the new board state after making a move in the given column,
as well as a flag indicating whether the move resulted in a win.
It's pretty straightforward but a bit tedious to
write because of the various directions you have to check for a winning row,
so I'm not reproducing it here, but you can check it out in the repo.

With that, we have everything to write a complete game loop:

```py
def play(model1, model2):
    """Have two models play against each other. Returns the winner (1 or 2)
    or 0 for a draw."""
    model1.eval()
    model2.eval()
    winner = 1
    with torch.no_grad():
        board = torch.zeros((6, 7), dtype=torch.int8, device=DEVICE)

        while True:
            # Get move from the model, play it and check for a win
            move = sample_move(model1, board, output_probs=output)
            board, win = make_move_and_check(board, move)

            if win:
                return winner

            elif torch.all(board[0, :] != 0):  # Check if the top row is full   
                return 0    # Draw

            board = -board                  # Flip the board for the other player
            winner = 3 - winner             # Alternate between 1 and 2
            model1, model2 = model2, model1 # Swap models for the next turn
```

There is another terminal condition we have to check for here: if there was no win but
the entire board
got filled up, which we check for by simply examining the top row, the game ended in a draw.

We now have the basic mechanism for self-play set up and can generate any number of
games by having two models play against each other. In practice we play an entire
batch of games at once because it's significantly more efficient; it complicates the
basic game loop above a bit but it's still pretty straightforward.

Another simple extension of this function we will need for training is that
it should be able to return not just the final result, but a full list of all encountered
board states, the moves that the model made in those states, and a vector of "returns" which
indicates if the model won or lost the game. So after playing one or a batch of games,
we'll get three tensors

```
    all_states:   (N, 6, 7)    dtype=torch.int8
    all_moves:    (N,)         dtype=torch.long
    all_returns:  (N,)         dtype=torch.float32
```

Here ``N`` is the total number of moves the model made across all games played.
We'll talk about the exact form the returns take
[in the next post]({{< relref the-reinforce-algorithm >}}), but for now just think
of them as being +1 for a win, 0 for a draw, and -1 for a loss.
