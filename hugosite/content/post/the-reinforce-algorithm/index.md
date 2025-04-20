---
title: 'The REINFORCE Algorithm'
date: 2025-04-20T20:29:21+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL', 'Math']
categories: ['Connect-Zero']
---

Let's say we have a Connect 4-playing model and we let it
[play a couple of games]({{< relref basic-setup-and-play >}}).
(We haven't really talked about model architecture until now, so for now just imagine a
simple multilayer perceptron with a few hidden layers which outputs 7 raw logits,
as discussed in the previous post.)

As it goes in life, our model wins some and loses some. How do we make it
actually learn from its experiences? How does the magic happen?

The REINFORCE algorithm is one of the basic workhorses of reinforcement learning, and
we'll start from there.
It's an example of a policy gradient method, which essentially means that it applies
the gradient descent techniques we know and love from basic ML to improve a policy network.
The good news is that it's fundamentally a simple method.

## The basic idea

We want to reward the model for good moves and punish it for bad
ones, but the issue is that at the moment we make a move, it's usually not
obvious yet if it was good or bad. So for now let's just say that moves that ultimately
led to a win were probably good, and moves that led to a loss were probably bad.

If the model found itself in board state \(s\) and took an action \(a\), and it
turned out to be a winning move, we want the model to do more of that. The standard
deep learning way of achieving that is computing the gradient \(\nabla p_a(s)\) (with respect to the model weights)
of the probability \(p_a(s)\) of choosing \(a\), and then taking a gradient step of the
model weights in that direction; this naturally pushes \(p_a(s)\) higher.

How big should our step be? We could come up with all kinds of heuristics for that,
but why not simply stick to a formula that we already know works well?
If the model currently outputs probabilities \( p_i(s) \) for the seven possible actions
\( i = 0, \ldots, 6 \) in state \(s\) and the one we want it to actually take is \(a\),
that's nothing else but a standard classification problem.
The standard loss function for that is
[cross entropy loss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
with the simple formula

\[
    \ell = -\log p_a(s).
\]

As an aside, if we compute the gradient of that, it's just

\[
    \nabla\ell = - \frac{\nabla p_a(s)}{p_a(s)}
\]

and that kind of makes intuitive sense: if the probability \(p_a(s)\) we want to be high
is already close to 1, we take a smaller step, but if \(p_a(s)\) is small, the step is larger.

We have an additional bit of information, and that is the **return** we got from making
move \(a\); it's a real number \(G \in [-1,1] \) representing the cumulative reward for an
action, adjusted for future outcomes. We'll discuss it further below.

If we simply multiply our loss with the return, we get the complete REINFORCE
policy loss function for a single sample,

\[
    \ell = -G \log p_a(s).
\]

It's clear that it takes larger steps for moves that gave us a higher return, and if
the move led to a loss and therefore \(G<0\), we actually take a gradient step in
the opposite direction, which is exactly what we want in order to discourage taking this
action in the future.

## Assigning returns

If we play an entire game (often called a "trajectory" in RL) with \(n\) moves,
we end up with sequences

\[
    (s_t), \quad (a_t), \qquad t=1,\ldots,n
\]

of board states and actions we took in those states.
To obtain the complete REINFORCE policy loss, we can simply sum up the single-state
loss function shown above. But we do need returns \(G_t \in [-1,1]\) for each of the
taken actions. How should we assign them?

We could simply set \(G_t = 1\) for all moves in won games and
\(G_t = -1\) for all moves in lost games, but that seems a bit overconfident.
Maybe we made a few suboptimal moves early on but still managed to pull off the win,
and we don't want to reward those early moves with the same weight.

On the other hand, if we set \(G_n=\pm1\) for the final winning or losing move only
and \(G_t=0\) for all previous moves, we don't learn anything at all about moves
that brought us into an ultimately advantageous position. The usual compromise
in REINFORCE is to choose a reward discount rate \(\gamma<1\) and compute the returns as

\[
    G_t = R \gamma^{n-t}, \qquad t=1,\ldots,n
\]

where \(R\) is 1 for a win and -1 for a loss; common choices are \(\gamma\in[0.9,0.99]\).
This just means that the final move gets the full reward \(R\), and the previous moves
get weighted less and less the earlier in the game they occurred.
This is called **discounted returns**.

Of course, if we played an entire batch of games, we can simply assign the returns for
each game individually and then throw them all together into a full batch of
states, actions, and returns. Our final batched policy loss function is then simply

\[
    \ell = - \sum_{t=1}^n G_t \log p_{a_t}(s_t).
\]


## Implementation

Once we understood the formula, it's actually not too hard to implement in PyTorch.
In practice we don't take the gradient step directly, but pass it to an optimizer
like Adam, which usually performs better than standard SGD.
Here's a basic implementation:

```py
def update_policy(
    model: nn.Module,
    optimizer: optim.Optimizer,
    states:  torch.Tensor,    # (B, 6, 7) - all board states
    actions: torch.Tensor,    # (B,)      - actions taken in those states
    returns: torch.Tensor     # (B,)      - discounted returns for the actions
):
    """Update the policy model using the REINFORCE algorithm."""
    model.train()

    # compute model logit outputs for the batch
    logits = model(states)                                         # (B, 7)

    # mask out illegal moves and compute log probs of all actions
    masked_logits = mask_invalid_moves_batch(states, logits)       # (B, 7)
    log_probs = F.log_softmax(masked_logits, dim=1)                # (B, 7)

    # select log probs of actually taken actions: (B, 7) -> (B, 1) -> (B,)
    log_probs_taken = torch.gather(log_probs, dim=1,
        index=actions.unsqueeze(1)).squeeze(1)

    # calculate REINFORCE loss = - sum(G_t * log p_(a_t)(s_t), t)
    policy_loss = -(returns * log_probs_taken).sum()
    
    # perform gradient update
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
```

There's one new function here, ``mask_invalid_moves_batch``, which performs the
masking operation for illegal moves described in the previous post for an entire
batch; simple stuff.

It's also worth pointing out the use of the
[``torch.gather``](https://pytorch.org/docs/stable/generated/torch.gather.html)
function: we use it to pick out from the tensor of all seven log-probabilities per sample
precisely the one corresponding to the action actually taken by indexing along the
second axis with the ``actions`` tensor. The tensor being used as an index needs to have
the same number of dimensions as the original tensor, which is why we need to call
``actions.unsqueeze(1)`` to add a second singleton axis to ``actions``.

With the REINFORCE algorithm implemented, we now have everything we need to start
training some models.