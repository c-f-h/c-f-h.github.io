---
title: 'Implementing and Evaluating REINFORCE with Baseline'
date: 2025-05-01T23:05:00+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL']
categories: ['Connect-Zero']

connect4: true
onnx: true
---

[Having introduced]({{< relref reinforce-with-baseline >}}) REINFORCE with baseline on a
conceptual level, let's implement it for our Connect 4-playing CNN model.

## Adding the value head

In the constructor of the [``Connect4CNN`` model class]({{< relref model-design >}}),
we set up the new network for estimating the board state value \(v(s)\)
which will consume the same 448 downsampled features that the policy head receives:

```py
        self.value_head = nn.Sequential(
            nn.Linear(64 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
```

It's very similar in structure to the MLP policy head, with some minor differences:

- We use 64 neurons on the two hidden layers instead of 128 since estimating value is
  presumably an easier task than finding the best next move.
- We omit the layer normalization since it doesn't seem to provide any significant benefits
  for the value head.
- There is only a single output, and we apply a ``tanh`` activation function to it which
  gives us exactly the range \([-1,1]\) that we want the value
  \(v(s)\) to have.

{{< figure src="tanh.svg" alt="The tanh function" caption="The tanh function"
  width="400" align="center" >}}

In the ``forward`` method of the model class, we apply the value head to the features
and return both policy and value:

```py
        # ... code as before ...
        # Apply residual CNN blocks
        x = F.relu(self.resblock1(x) + x)
        x = F.relu(self.resblock2(x) + x)
        x = F.relu(self.resblock3(x) + x)

        # Downsample columnwise: (B, 64, 1, 7) -> (B, 7 * 64)
        x = self.downsample(x).view(-1, 7 * 64)

        # Fully connected layers (MLP head, outputs logits): (B, 7)
        p = self.fc_layers(x)
        # Value head: (B, 1) -> (B,)
        v = self.value_head(x).squeeze(-1)
        
        if len(original_shape) == 2:
            # Remove singleton batch dimension
            p = p.squeeze(0)
            v = v.squeeze(0)
        return p, v
```

All that has changed here is that we invoke the new value head with the same input that
also the policy head receives. We then return both policy and value
as a tuple ``(p, v)``.

## Implementing the new update function

We previously already modified the ``update_policy`` function from the
[basic REINFORCE algorithm]({{< relref the-reinforce-algorithm >}}#Implementation)
by introducing [entropy regularization]({{< relref entropy-regularization >}});
now we make a few further adjustments as follows:

```py
    # compute model outputs for the batch
    logits, value = model(states)

    # mask out illegal moves and compute logprobs of the actually taken actions
    masked_logits = mask_invalid_moves_batch(states, logits, mask_value=-1e9)  # (B, 7)
    log_probs = F.log_softmax(masked_logits, dim=1)                # (B, 7)
    entropy = -(log_probs * torch.exp(log_probs)).sum(1)           # (B,)
    # select log probs of actually taken actions: (B, 7) -> (B, 1) -> (B,)
    log_probs_taken = torch.gather(log_probs, dim=1,
        index=actions.unsqueeze(1)).squeeze(1)                     # (B,)

    # calculate RwB loss = -sum(A_t * log p_(a_t)(s_t), t)
    ## IMPORTANT! Detach so gradients don't flow back into the value network
    advantage = (returns - value).detach()
    policy_loss = -(advantage * log_probs_taken).sum()

    # reinforce with baseline: estimate value from Monte Carlo samples
    value_loss = F.mse_loss(value, returns, reduction='sum')

    total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss \
        - ENTROPY_BONUS * entropy.sum()
    
    # perform gradient update
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
```

The salient changes here are:

- The model now outputs two tensors:
  ``logits, value = model(states)``. The logits are converted into probabilities as before.
- We compute the observed advantage using ``(returns - value).detach()``. Detaching the
  tensor is an important step!
  It tells PyTorch that we don't want gradients to flow back via this tensor
  into the value network during backpropagation. Rather, we simply want to plug the
  numerical value of the advantage into the policy update function. Otherwise, the policy
  update could modify the value, but value should only be learned via the ``value_loss``
  function.
- We modify the ``policy_loss`` to use the advantage rather than the direct returns for
  weighting, as discussed in the previous post.
- We add the squared error \((G - v(s))^2\), summed over all samples in the batch,
  to the total loss with a weight of ``VALUE_LOSS_WEIGHT`` which we typically set to 0.5.
  This will cause the value head to attempt to learn the return \(G\).

## Minor other changes

We also have to update our other utility functions to be able to handle models which output
both move logits and a value. In particular, we need to change the function
``play_multiple_against_model`` which we use to sample games against an opponent model to
be aware of this change. This is very simple though since it has no need for the value
function and can simply discard the second output of the model.

## Evaluating REINFORCE with baseline in self-play

It's time to let the residual CNN model with the brand new value head play itself using the
[self-play loop]({{< relref policy-collapse >}}#the-self-play-loop).
Here's a run down of the hyperparameters and other choices:

- Learning rate is \(10^{-4}\); AdamW is used as the optimizer.
- Value loss weight is \(\alpha = 0.5\).
- [Entropy bonus]({{< relref entropy-regularization >}}) is \(\beta = 0.05\).
- The [reward discount rate]({{< relref the-reinforce-algorithm >}}#assigning-returns)
  is \(\gamma=0.9\).
- A batch consists of 50 games; the model is updated via the update described above after
  each full batch.
- The model always plays against a recent checkpoint copy of itself.
- Every 100 batches (5000 games) we evaluate the performance of the current model against
  the checkpoint; if it achieved a win rate of over 52% in the last 1000 games,
  it becomes the new checkpoint model.
- Every 20 batches (1000 games) we evaluate the win rate of the current model against
  the [``RandomPunisher``]({{< relref random-punisher >}}) using 100 games,
  purely for tracking our progress.
  Both this win rate and some other stats are plotted at these intervals.
  Notably, this time around, we add the
  value loss, the standard deviation of the returns \(G\), and the standard deviation
  of the advantage \(G - v(s)\) to our stats to be plotted.

Let's look at some snapshots of the training progress at various milestones.

{{< figure src="Figure_1.png" width="100%" align="center"
  caption="Self-play training after 350k games.">}}

Initial training is slow and noisy as the model fumbles to learn the basics of the
game and only has itself as an
(unreliable) opponent and teacher. We could definitely speed this phase up significantly
by using the ``RandomPunisher`` as an opponent instead.
Entropy also drops precariously low to around
0.2, but eventually works itself out thanks to the entropy bonus.
Value and advantage are still very noisy here.

After around 300k games, the model convincingly breaks the previous benchmark of a 50%
win rate against the ``RandomPunisher``. Here things get a lot better: entropy stabilizes
above 0.7, and the value network really starts learning now.

{{< figure src="Figure_3.png" width="100%" align="center"
  caption="Self-play training after 550k games.">}}

In the next phase up to around 550k games, we now see textbook behavior of the value
network. The value loss decreases steadily to around 0.28, and the
standard deviation of the advantage is already noticeably, if not dramatically, lower
(0.52) than the standard deviation of the raw returns (0.57); this is variance reduction
in effect.

The win rate against the ``RandomPunisher`` reference opponent steadily climbs to around
75%. Entropy remains steady in the 1.0--1.2 range.

{{< figure src="Figure_5.png" width="100%" align="center"
  caption="Self-play training after 850k games.">}}

The previous trends still continue unabated at the 850k games mark:
value loss has further dropped to 0.23, and the advantage stddev is now decisively
lower at 0.47 than the returns stddev at 0.55.

Win rate has now increased to around 86%.

{{< figure src="Figure_6.png" width="100%" align="center"
  caption="Self-play training after 1.6M games.">}}

Remarkably, the improvements continue even much deeper into the run: after almost
1.6M games, the win rate has climbed further to 94%, occasionally even pushing 98%.
Value loss is down to 0.14, and advantage stddev has further dropped to 0.37 versus
the returns stddev at 0.50, proving that variance reduction keeps improving as well.

After the initial drop,
entropy has remained stable above 1.0 for the entire remainder of the run.
Also, the policy loss never seems to do anything particularly interesting, mostly
oscillating noisily around the -0.04 to -0.03 range, even as the win rate steadily
increases.

## Conclusion

The model we trained is now a seriously strong player, completely eviscerating the once so
intimidating ``RandomPunisher``. It even beats the already quite strong model which served
as the interactive opponent in the [teaser post]({{< relref connect-4 >}}) for this series
around 75% of the time.

Most importantly, the entire training run was based entirely on self-play, and there was
no manual tweaking or annealing of any hyperparameters; instead, it was a "set it and
forget it" affair with straightforward parameter choices,
unlike the first experiments with basic REINFORCE.
This drives home the significantly improved robustness of the algorithm with baseline.

Nevertheless, we do see one of the drawbacks of REINFORCE with baseline: as discussed
in the last post,
the Monte Carlo estimate for the value network has high variance, and we observe
that in the fact that the model needs a lot of samples to learn to predict the returns.
Even after 1.6M games, the value estimate seems not to have fully converged yet.

## Up for a game?

If you're feeling lucky, you can try your hand at playing against the model we trained in
the applet below:

<div id="game-container1" class="connect4-container"
    data-human="1" data-cpu="2"
    data-random-first-player="true"
    data-onnx-model="model-mk4-rwb.onnx"
></div>
