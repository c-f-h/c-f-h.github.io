---
title: 'A First Training Run and Policy Collapse'
date: 2025-04-21T17:45:00+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL', 'Math']
categories: ['Connect-Zero']

connect4: true
---

With the [REINFORCE algorithm]({{< relref the-reinforce-algorithm >}}) under our belt,
we can finally attempt to start training some models for
[Connect 4]({{< relref connect-zero >}}).
However, as we'll see, there are still some hurdles in our way before we get anywhere.
It's good to set your expectations accordingly because rarely if ever do things go
smoothly the first time in RL.

> [!important] Example Code
> Runnable example code for this post:  
> [``connect-zero/train/example1-collapse.py``](https://github.com/c-f-h/connect-zero/blob/main/train/example1-collapse.py).


## A simple MLP model

As a fruitfly of Connect 4-playing models, let's start with a simple multilayer perceptron
(MLP) model that follows the [model protocol]({{< relref basic-setup-and-play >}}) we
outlined earlier: that means that it has an input layer taking a 6x7 `int8` board state
tensor, a few simple hidden layers consisting of just a linear layer and a ReLU activation
function each, and an output layer of 7 neurons without any activation function---that's
exactly what we meant earlier when we said that the model should output raw logits.

That's straightforward to describe in PyTorch:

```py
class SimpleMLPModel(nn.Module):
    """Create a simple MLP model for Connect4."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Flatten(),
            nn.Linear(ROWS * COLS, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, COLS)
        ])
    
    def forward(self, x):
        # Store original shape and determine batch size
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)              # -> (1, 6, 7)
        x = x.float()  # Ensure input is float32

        for layer in self.layers:
            x = layer(x)

        # If the original input was a single instance, remove the batch dimension
        if len(original_shape) == 2:
            x = x.squeeze(0)                # -> (7,)
        return x
```

There's some extra code in the ``forward`` method which just makes sure that our model can
deal both with a single board input with shape ``(6, 7)`` as well as with a batch of boards,
``(B, 6, 7)``.

For a strong board-game playing model, we'll want at least a few convolution layers
[later on]({{< relref model-design >}}),
but as we'll see shortly, playing strength is not the first thing we have to worry
about.

## The self-play loop

Let's write the self-play loop. It's common in this type of training to have the
model play not against its current "live" version, but to take checkpoints of it at
regular intervals and have it compete against that.
So we have the live ``model`` and its checkpointed version ``model_cp`` against which
the main model plays, and we copy the weights over from the former to the latter at
regular intervals.

We'll run 50 games per batch, where we have to make sure that each model gets to move
first an equal number of times since the first player has a significant advantage in
Connect 4. (Strictly speaking, [Connect 4 is a solved game](https://en.wikipedia.org/wiki/Connect_Four#Mathematical_solution)---under perfect play the first player always
wins.) We have a function ``play_multiple_against_model`` which lets each model go first
equally and also collects the needed information for REINFORCE (states, actions, and
returns) discussed in the previous post.
Then, every 100 batches (what we somewhat arbitrarily call an epoch here),
we update the checkpoint model and save it to disk for good measure.

Here's the code for the self-play loop:

```py
def self_play_loop(model_constructor, games_per_batch=50, batches_per_epoch=100,
        learning_rate=1e-3, cp_file="last_cp.pth"):
    # Create two copies of the model
    model = model_constructor()
    model_cp = model_constructor()

    # initialize the Adam optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # ----------------- MAIN TRAINING LOOP ----------------- #
    epoch = 0
    while True:
        for batchnr in range(batches_per_epoch):
            # play a batch of games
            board_states, actions, returns = play_multiple_against_model(
                model, model_cp, num_games=games_per_batch)

            # apply the REINFORCE policy update rule
            update_policy(model, optimizer, board_states, actions, returns)

            if batchnr % 20 == 19:
                print(f"Batch {batchnr+1} / {batches_per_epoch} done.")

        # Save model state to a checkpoint file
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, cp_file)

        # Copy the model state to the checkpoint model
        model_cp.load_state_dict(model.state_dict())
        epoch += 1
        print(f"\nEpoch {epoch} done. Model saved to {cp_file}.")

if __name__ == "__main__":
    DEVICE = torch.device("cpu")
    self_play_loop(SimpleMLPModel, games_per_batch=50, batches_per_epoch=100,
        cp_file="last_cp.pth")

```

There's something very important missing here: monitoring and plotting.
We won't get far in RL without extensive monitoring because things
**will** go wrong in totally unexpected ways, and we won't even know what's wrong if
we don't have the capability to plot all kinds of quantities of interest.

The actual code in the repo for this project does have ``matplotlib``-based
visualization which I stripped out for the example code here because it's essentially
plumbing and not that interesting. Just keep in mind that you do need it. You could
always use [tensorboard](https://www.tensorflow.org/tensorboard), for example.

## The results

Let's run the training loop and observe what happens. We use a learning rate of
\(10^{-3}\) and a reward discount rate of \(\gamma=0.9\).
The whole thing runs really quickly even on a modern laptop CPU; the results below
took perhaps a minute or two to produce.

{{< figure src="collapse.png" alt="Training plots of a policy collapse situation"
  align="center"
  caption="Plots of win rate, entropy, policy loss and game length over our first self-play training run."
>}}

Here I had the monitoring framework produce four plots:

- **Win rate:** This is the fraction of games won by ``model``.
  The win rate does increase initially, indicating that the model might be learning some
  basic patterns. New data is added to the plot every 20 batches, and you
  can clearly see that at every fifth data point, corresponding to the 100 batches per
  epoch, the win rate shows a sharp spike back down towards 50%. This is expected since
  we start playing against a new identical checkpoint of our model at those intervals.
  What is not expected is the win rate flatlining at 50% at the end of the run.

- **Entropy:** In essence, the amount of randomness exhibited by the model. We'll go
  into more detail on this later.
  For now, observe that it tapers off at a level close to zero towards the end.
  This means that the model is now behaving almost completely deterministically
  without any random exploration.

- **Policy loss:** This is just the loss function of the [REINFORCE algorithm]({{< relref
  the-reinforce-algorithm >}}). It certainly looks very noisy and there's no clear downward
  trend, but in fact this is not too unusual for this method; the basic REINFORCE algorithm
  is known for its noisy gradients and high variance. Unlike in a standard ML application,
  where we are aiming at a fixed target (our dataset), during self-play our opponent is
  rapidly changing, and we are also constantly exploring new parts of the game tree even
  if our opponent remains relatively steady; therefore even a noisy but overall flat policy
  loss curve doesn't necessarily mean that the model has stopped learning anything useful.
  All in all I would consider this the least concerning out of the four plots.

- **Game length:** This is the average number of moves played by ``model``,
  so the total number of moves per game is roughly double that. Just like the win rate, this
  completely flattens out at a fixed value of 4.5 towards the end of the run. 


## What went wrong

So what happened here? The model quickly collapsed into a state where the optimizer had
squeezed all the randomness (or entropy) out of it, long before it had time to explore the
game tree enough to learn how to play well. It essentially found a very poor local minimum
where it had "learned" to play one and the same game over and over against itself,
picking up a win as the first player and a loss as the second player, which explains
the win rate sitting precisely at 50%.

This is the game it kept replaying like a broken record:

<div id="game-container" class="connect4-container"
    data-human="-1" data-cpu="-1"
    data-movelist="[2, 2, 3, 2, 2, 2, 5, 2, 4]">
</div>

You can see that the game is nine moves long and ends in a quick win for the first player,
which explains the average game length (for one player) ending up at 4.5 moves.

This example shows immediately why having good monitoring is crucial: if we had, say,
only a loss plot, we would have no idea of what went wrong here; in fact, we might not
even realize that there is something terminally wrong and waste a lot of time training
a broken model.

The policy network getting stuck in such a poor local minimum without any hope of making
it out is known as **policy collapse**, and we'll look at ways to avoid it in the next post.