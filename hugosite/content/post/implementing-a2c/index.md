---
title: 'Implementing A2C'
date: 2025-05-10T18:43:00+02:00
tags: ['Python', 'PyTorch', 'ML', 'RL']
categories: ['Connect-Zero']
---

In the [previous post]({{< relref actor-critic >}}), we outlined the general concept of
Actor-Critic algorithms and A2C in particular; it's time to implement a simple version of A2C
in PyTorch.

## Changing the reward function

As we noted, our model class doesn't need to change at all: it already has the requisite
value head we introduced when we [implemented REINFORCE with baseline]({{< relref implementing-rwb >}}).

First off, we need to change the way the rewards are computed. We introduce a flag
``BOOTSTRAP_VALUE`` which is on when we use A2C. Based on this, we compute the rewards
vector for a game with an ``outcome`` of +1 for a win, 0 for a draw, and -1 for a loss
like this:

```py
def compute_rewards(num_moves: int, outcome: int) -> torch.Tensor:
    if BOOTSTRAP_VALUE:
        # bootstrapping: sparse rewards, only assigned for the winning move
        if outcome != 0:
            move_nr = torch.arange(num_moves, device=DEVICE)
            return ((move_nr == (num_moves - 1)) * outcome).float()
        else:
            return torch.zeros((num_moves,), device=DEVICE)
    else:
        # Monte Carlo sampling: rewards are discounted over the game
        move_nr = torch.arange(num_moves, device=DEVICE)
        return (outcome * REWARD_DISCOUNT**move_nr).flip(dims=(0,))
```

The logic is simple, and I'm showing both the previous REINFORCE path and the new A2C path
for comparison. Previously, we computed gradually discounted returns based on the
``REWARD_DISCOUNT`` parameter. Now, we return **sparse rewards:** a vector which is zero for all
moves except for the final one, where it is +1 or -1 depending on the game result.

Note that -1 is actually not a "real" reward as the game is not over at that point and the
opponent still has to play the winning move, which will give *them* a +1 reward. We'll handle this
specially in the update function.

Perhaps the following diagram helps illustrate the relationships: states have associated values
which are the sums of all discounted future rewards, and actions take us
from one state to the next and have immediate rewards associated to them. In our case, all
the rewards are zero except for a winning action, which has reward 1.

<figure>
<svg viewbox="30 25 570 100" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" fill="currentColor" stroke="currentColor" stroke-width="1.5">
  <!-- States -->
  <rect x="50" y="80" width="80" height="40" fill="none" />
  <text x="90" y="97" text-anchor="middle" font-weight="bold" stroke="none">s₀</text>
  <text x="90" y="113" text-anchor="middle" font-size="12" stroke="none">v(s₀)</text>
  <rect x="200" y="80" width="80" height="40" fill="none" />
  <text x="240" y="97" text-anchor="middle" font-weight="bold" stroke="none">s₁</text>
  <text x="240" y="113" text-anchor="middle" font-size="12" stroke="none">v(s₁)</text>
  <rect x="350" y="80" width="80" height="40" fill="none" />
  <text x="390" y="97" text-anchor="middle" font-weight="bold" stroke="none">s₂</text>
  <text x="390" y="113" text-anchor="middle" font-size="12" stroke="none">v(s₂)</text>
  <rect x="500" y="80" width="80" height="40" fill="none" />
  <text x="540" y="97" text-anchor="middle" font-weight="bold" stroke="none">s₃</text>
  <text x="540" y="113" text-anchor="middle" font-size="12" stroke="none">v(s₃)</text>
  <!-- Arrows with rewards -->
  <path d="M130,80 C140,40 190,40 200,80" fill="none" stroke="#666" marker-end="url(#arrowhead)" />
  <text x="166" y="45" text-anchor="middle" font-size="14" stroke="none">R₀</text>
  <path d="M280,80 C290,40 340,40 350,80" fill="none" stroke="#666" marker-end="url(#arrowhead)" />
  <text x="316" y="45" text-anchor="middle" font-size="14" stroke="none">R₁</text>
  <path d="M430,80 C440,40 490,40 500,80" fill="none" stroke="#666" marker-end="url(#arrowhead)" />
  <text x="466" y="45" text-anchor="middle" font-size="14" stroke="none">R₂</text>
  <!-- Arrowhead definition -->
  <defs>
    <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" stroke-width="0">
      <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
    </marker>
  </defs>
</svg>
<figcaption>
  <p>States s<sub>i</sub> with associated values v(s<sub>i</sub>), and (immediate) rewards
  R<sub>i</sub> for the actions leading from state s<sub>i</sub> to s<sub>i+1</sub>.</p>
</figcaption>
</figure>

Assuming that \(s_3\) is a terminal state in this sketch, we have, as an example,

\[
\begin{align*}
    v(s_1) &= R_1 + \gamma v(s_2) \\
           &= R_1 + \gamma (R_2 + \gamma v(s_3)) \\
           &= R_1 + \gamma R_2,
\end{align*}
\]

since the terminal state has no further rewards and therefore \(v(s_3) = 0\).
For our learned value function, these relationships hold only approximately.


> [!NOTE]
> **"Reward"** is the usual term for the immediate
result of a move (which, in our case, is only nonzero for a winning move), and **"return"**
includes both the immediate and all discounted future rewards. Since we are reusing the
same variables which previously stored returns to now store the sparse rewards, the nomenclature
unfortunately gets a bit muddled in some places.

## Changing the ``update_policy`` function

In the ``update_policy`` function, all the code obtaining the model's logit outputs and
converting them to masked probabilities remains the same, and I won't reproduce it here again.
As always, you can check out the entire code
[in the GitHub repo](https://github.com/c-f-h/connect-zero/blob/main/train/main.py).

```py
    # --- Actor-Critic (A2C): use bootstrapped value estimates --- #

    # get new board states resulting from the moves we made
    new_states = make_moves_batch(states, actions)
    # Since the value function is symmetric, we ask what value the resulting state has
    # for the opponent (-board) and take the negative value of that.
    V_next = -model(-new_states)[1].detach()

    # Check for terminal state (move won the game, or the game is drawn)
    is_terminal = (returns == 1) | is_board_full_batch(new_states)
    # Set value = 0 for terminal states since there can be no further rewards
    V_next[is_terminal] = 0.0
    
    # The -1 signal from losses is not a real move reward; handled below.
    real_rewards = torch.maximum(returns, torch.zeros_like(returns))

    # Value bootstrapping (TD learning):
    # target value is reward from this move + discounted value of next state
    v_target = real_rewards + REWARD_DISCOUNT * V_next
    # For lost games, we force V_next = -1
    v_target[returns == -1] = -REWARD_DISCOUNT
    
    # Weight winning/losing moves higher since they are an important signal and rare
    weight = torch.ones_like(returns)
    weight[returns == 1]  = 2.0
    weight[returns == -1] = 2.0
    
    value_loss = F.mse_loss(value, v_target, weight=weight, reduction='sum')
    advantage = (v_target - value).detach()     # Important! Detach from value network

    # --- Calculate Policy Loss ---
    # Loss = - Σ [ A_t * log π(a_t | s_t) ]
    policy_loss = -(advantage * log_probs_taken).sum()
    
    # add up loss contributions, backpropagate and update model weights
    optimizer.zero_grad()
    total_loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss \
        - ENTROPY_BONUS * entropy.sum()
    total_loss.backward()
    optimizer.step()
```

Hopefully the comments get the basic idea across. Let's go over it in more detail:

- First we compute the new board states resulting from the moves we took and ask the value
  network to rate them to obtain ``V_next``. There's an important point here: in the new board
  position, it will be the opponent's move, so we have to feed ``-new_states`` to the value
  network. This will result in values for the opposing side, but we can flip it again by
  taking the negative value. This exploits the symmetry of the value function in a zero-sum
  game, where our outcome is minus the opponent's outcome.

  It's important to ``detach()`` these value estimates; refer to the discussion in the post on
  [implementing REINFORCE with baseline]({{< relref implementing-rwb >}}) for details.
- We then check which moves resulted in a terminal state (win or draw from a full board) and
  zero out the resulting values: ``V_next[is_terminal] = 0.0``, since no further rewards
  can be earned once the game is over.

  (The value network can't learn this on its own since it never sees terminal states during
  training.)
- We take only the nonnegative parts of the input reward vector to clip out the -1 loss
  signal. As discussed above, this is not an actual move reward, and it will be handled specially.
- Then comes the actual TD bootstrapping formula:
  ``v_target = real_rewards + REWARD_DISCOUNT * V_next``. This is exactly what we discussed in
  the previous post: immediate rewards (0 or 1) plus the discounted estimate of future value.
- Here the special treatment of losing moves comes in: if the reward from ``compute_rewards()``
  was -1, we fix ``v_target = -REWARD_DISCOUNT``.

  This is a special case of the general formula above: since the opponent had a winning
  move in the next state, we assume ``V_next = -1``, and the ``real_reward`` is 0 since
  we didn't play a terminal move.
  It's not strictly necessary to hardcode this: eventually, the value network should learn
  that ``V_next = -1`` on its own.
  However, we can speed up initial learning significantly by incorporating this
  fixed bit of knowledge directly.
- We use a weighted squared error loss function to pull the value estimate towards ``v_target``.
  There's another little tweak here: we weight contributions coming from a +1/-1 endgame
  signal higher by setting their weight to 2.0. You can also treat this as a hyperparameter and
  set it even higher during initial training. The idea is for the network to learn the value
  of game-ending moves as quickly as possible.
- The advantage is computed as ``v_target - value``, as discussed in the previous post.
  Detaching is again crucial here so that the policy loss won't influence the value head.
  The total loss function is then the same as in REINFORCE with baseline.

## Discussion

You can see that there are many subtle details here, and I found even this basic version of A2C
much trickier to get right than the relatively straightforward REINFORCE with baseline.
Especially the handling of endgame rewards needs a lot of care. It's easy to introduce subtle
bugs here where it might look like learning is still happening, but actual performance will be
poor. As always, robust monitoring is important here.

I also noted some tweaks above which were not part of the conceptual description of the
algorithm: forcing lost state value to -1 and weighting terminal move contributions higher.
These are not strictly necessary, but help a lot especially during early training.
They were introduced by monitoring situations which were holding the model back and
implementing targeted fixes for these issues.

### Advantage normalization

There is another optional tweak which I did not include here: normalizing the advantage.
It's easy to implement: just before computing the policy loss, we could do

```py
    advantage_std, advantage_mean = torch.std_mean(advantage)
    advantage = (advantage - advantage_mean) / (advantage_std + 1e-8)
```

What this does is "whiten" the advantage estimates by transforming them to have zero mean
and standard deviation one.
There is no real theoretical justification for this, but many implementations and practical
studies include it; for instance, the
[OpenAI Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/a2c.html#parameters)
implementation does have it as an (off by default) argument.

Many practitioners report increased learning robustness when using this normalization,
but for our Connect 4 setting, my experiences with it were decidedly mixed.
The paper [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990)
from the Google Brain Team also didn't find strong evidence for the normalization being
essential, so I feel justified in omitting it for now.

This concludes our tour of some implementation issues arising in A2C.
[Next up]({{< relref evaluating-a2c >}}) we'll have a little shootout to see if A2C can
actually improve upon REINFORCE with baseline!