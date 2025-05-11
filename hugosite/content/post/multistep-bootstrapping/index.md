---
title: 'Multi-Step Bootstrapping'
date: 2025-05-11T20:10:00+02:00
tags: ['Math', 'Python', 'PyTorch', 'ML', 'RL', 'A2C']
categories: ['Connect-Zero']
---

Until now, we've done one-step lookahead for the TD bootstrapping in the A2C algorithm.
We can significantly improve upon this by looking further ahead.

## Bootstrapping with one step

Looking back at the states-values-rewards diagram in
[Implementing A2C]({{< relref implementing-a2c >}}),
we had state \(s_i\) transitioning into state \(s_{i+1}\) with an immediate reward \(R_i\).
How we actually implemented bootstrapping was subtly different and better described by
this diagram:

<figure class="align-center">
    <svg viewBox="0 40 350 390" style="max-width: 500px" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" fill="currentColor" stroke="currentColor" stroke-width="1.5">
        <!-- Arrowhead definition -->
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto" stroke-width="0">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
        </defs>
        <!-- Left column states -->
        <rect x="60" y="50" width="80" height="40" fill="none" />
        <text x="100" y="67" text-anchor="middle" font-weight="bold" stroke="none">s₀</text>
        <text x="100" y="83" text-anchor="middle" font-size="12" stroke="none">v(s₀)</text>
        <rect x="60" y="180" width="80" height="40" fill="none" />
        <text x="100" y="197" text-anchor="middle" font-weight="bold" stroke="none">s₁</text>
        <text x="100" y="213" text-anchor="middle" font-size="12" stroke="none">v(s₁)</text>
        <rect x="60" y="310" width="80" height="40" fill="none" />
        <text x="100" y="327" text-anchor="middle" font-weight="bold" stroke="none">s₂</text>
        <text x="100" y="343" text-anchor="middle" font-size="12" stroke="none">v(s₂)</text>
        <!-- Right column states (staggered) -->
        <rect x="210" y="115" width="80" height="40" fill="none" />
        <text x="250" y="132" text-anchor="middle" font-weight="bold" stroke="none">s'₀</text>
        <text x="250" y="148" text-anchor="middle" font-size="12" stroke="none">v(s'₀)</text>
        <rect x="210" y="245" width="80" height="40" fill="none" />
        <text x="250" y="262" text-anchor="middle" font-weight="bold" stroke="none">s'₁</text>
        <text x="250" y="278" text-anchor="middle" font-size="12" stroke="none">v(s'₁)</text>
        <rect x="210" y="375" width="80" height="40" fill="none" />
        <text x="250" y="392" text-anchor="middle" font-weight="bold" stroke="none">s'₂</text>
        <text x="250" y="408" text-anchor="middle" font-size="12" stroke="none">v(s'₂)</text>
        <!-- Arrows (from bottom-right to top-left, and bottom-left to top-right) -->
        <line x1="140" y1="90" x2="210" y2="115" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="210" y1="155" x2="140" y2="180" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="140" y1="220" x2="210" y2="245" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="210" y1="285" x2="140" y2="310" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="140" y1="350" x2="210" y2="375" stroke="#666" marker-end="url(#arrowhead)" />
        <!-- Reward texts -->
        <text x="175" y="90" text-anchor="middle" font-size="14" stroke="none">R₀</text>
        <text x="175" y="160" text-anchor="middle" font-size="14" stroke="none">R'₀</text>
        <text x="175" y="225" text-anchor="middle" font-size="14" stroke="none">R₁</text>
        <text x="175" y="290" text-anchor="middle" font-size="14" stroke="none">R'₁</text>
        <text x="175" y="355" text-anchor="middle" font-size="14" stroke="none">R₂</text>
    </svg>
  <figcaption>
  <p>States and rewards diagram for own states s<sub>i</sub> and opponent states s'<sub>i</sub>.</p>
</figcaption>
</figure>

In the left column are the states \(s_i\) where we move,
and in the right column the states \(s'_i\) where our opponent moves.
For each state \(s_i\), we computed the next state \(s'_i\) with the function ``make_moves_batch``.
We then computed the value estimate for these states by doing a feedforward pass through
our model and reading off the output of the value head, yielding \(v(s'_i)\).

Then, our bootstrapping target consisted of the immediate reward \(R_i\) plus the discounted
value of the opponent's state:

\[
    v(s_i) \gets R_i + \gamma v(s'_i).
\]

So the variant we implemented only looks ahead a single move, or, as it's often called in game
theory, "one ply". [^1]

[^1]: The word "ply" is used to unambiguously refer to one action taken by one player. This is
to avoid confusion in games like chess, where one "move" traditionally comprises two half-moves:
one by the white player and one by the black player. So a "move" in chess consists of two plies.

## Bootstrapping with two steps

What if we wanted to look two steps ahead? We can simply apply the bootstrapping formula twice,
replacing \(v(s'_i)\) with \(R'_i + \gamma v(s_{i+1})\):

\[
\begin{align*}
    v(s_i) &\gets R_i + \gamma v(s'_i)     \\
           &= R_i + \gamma (R'_i + \gamma v(s_{i+1})) \\
           &= R_i + \gamma R'_i + \gamma^2 v(s_{i+1}).
\end{align*}
\]

If we combine our and the opponent's rewards into a single effective reward,

\[
    \overline R_i := R_i + \gamma R'_i,
\]

we obtain what looks essentially
like our original bootstrapping formula, just with the squared discount rate:

\[
    v(s_i) \gets \overline R_i + \gamma^2 v(s_{i+1}).
\]

This is the formula for two-step (or 2-ply) TD bootstrapping.
Here we assumed that \(s'_i\) isn't terminal, but if it is (because we made a winning move
or reached a draw),
we set \(\overline R_i := R_i\) and \(v(s_{i+1})=0\), much as we did in the
implementation of the 1-ply formula.
As a result, we obtain \(v(s_i) \gets R_i\) for terminal states, which correctly assigns the
reward for the game-ending move.

Essentially, we are skipping over the opponent's states \(s'_i\) and going directly
from \(s_i\) to \(s_{i+1}\), as indicated by the new arrows in this diagram:

<figure class="align-center">
    <svg viewBox="0 40 350 390" style="max-width: 500px" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif" fill="currentColor" stroke="currentColor" stroke-width="1.5">
        <!-- Arrowhead definitions -->
        <defs>
            <!-- Original arrowhead for diagonal arrows -->
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5"
                orient="auto" stroke-width="0" markerUnits="userSpaceOnUse">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
            <!-- Smaller arrowhead for thicker curved arrows -->
            <marker id="small-arrowhead" markerWidth="10" markerHeight="7" refX="10" refY="3.5"
                orient="35" stroke-width="0" markerUnits="userSpaceOnUse">
                <polygon points="0 0, 10 3.5, 0 7" fill="#666" />
            </marker>
        </defs>
        <!-- Left column states -->
        <rect x="60" y="50" width="80" height="40" fill="none" />
        <text x="100" y="67" text-anchor="middle" font-weight="bold" stroke="none">s₀</text>
        <text x="100" y="83" text-anchor="middle" font-size="12" stroke="none">v(s₀)</text>
        <rect x="60" y="180" width="80" height="40" fill="none" />
        <text x="100" y="197" text-anchor="middle" font-weight="bold" stroke="none">s₁</text>
        <text x="100" y="213" text-anchor="middle" font-size="12" stroke="none">v(s₁)</text>
        <rect x="60" y="310" width="80" height="40" fill="none" />
        <text x="100" y="327" text-anchor="middle" font-weight="bold" stroke="none">s₂</text>
        <text x="100" y="343" text-anchor="middle" font-size="12" stroke="none">v(s₂)</text>
        <!-- Right column states (staggered) -->
        <rect x="210" y="115" width="80" height="40" fill="none" />
        <text x="250" y="132" text-anchor="middle" font-weight="bold" stroke="none">s'₀</text>
        <text x="250" y="148" text-anchor="middle" font-size="12" stroke="none">v(s'₀)</text>
        <rect x="210" y="245" width="80" height="40" fill="none" />
        <text x="250" y="262" text-anchor="middle" font-weight="bold" stroke="none">s'₁</text>
        <text x="250" y="278" text-anchor="middle" font-size="12" stroke="none">v(s'₁)</text>
        <rect x="210" y="375" width="80" height="40" fill="none" />
        <text x="250" y="392" text-anchor="middle" font-weight="bold" stroke="none">s'₂</text>
        <text x="250" y="408" text-anchor="middle" font-size="12" stroke="none">v(s'₂)</text>
        <!-- Diagonal arrows (from bottom-right to top-left, and bottom-left to top-right) -->
        <line x1="140" y1="90" x2="210" y2="115" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="210" y1="155" x2="140" y2="180" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="140" y1="220" x2="210" y2="245" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="210" y1="285" x2="140" y2="310" stroke="#666" marker-end="url(#arrowhead)" />
        <line x1="140" y1="350" x2="210" y2="375" stroke="#666" marker-end="url(#arrowhead)" />
        <!-- Curved arrows within left column (bolder, with smaller arrowhead) -->
        <path d="M60,90 C30,115 30,165 60,180" fill="none" stroke="#666" stroke-width="2.5" marker-end="url(#small-arrowhead)" />
        <path d="M60,220 C30,245 30,295 60,310" fill="none" stroke="#666" stroke-width="2.5" marker-end="url(#small-arrowhead)" />
        <!-- Reward texts for diagonal arrows -->
        <text x="175" y="90" text-anchor="middle" font-size="14" stroke="none">R₀</text>
        <text x="175" y="160" text-anchor="middle" font-size="14" stroke="none">R'₀</text>
        <text x="175" y="225" text-anchor="middle" font-size="14" stroke="none">R₁</text>
        <text x="175" y="290" text-anchor="middle" font-size="14" stroke="none">R'₁</text>
        <text x="175" y="355" text-anchor="middle" font-size="14" stroke="none">R₂</text>
        <!-- Reward texts for curved arrows -->
        <text x="25" y="145" text-anchor="middle" font-size="14" stroke="none">R̅₀</text>
        <text x="25" y="275" text-anchor="middle" font-size="14" stroke="none">R̅₁</text>
    </svg>
  <figcaption>
  <p>Direct reward assignment for bootstrapping state s<sub>i</sub> using s<sub>i+1</sub>.</p>
</figcaption>
</figure>

There's another, slightly more abstract interpretation here: we can in essence forget that a second
player even exists and instead think of our **agent** as interacting with a (stochastic) **environment**
(a [Markov process](https://en.wikipedia.org/wiki/Markov_chain), if you want to be mathematical about it). So our policy chooses an action, the environment (consisting of the game board
and the opponent) reacts to it, a reward is handed out,
and we find ourselves in a new state \(s_{i+1}\).
In this interpretation, the right column of the diagram vanishes completely, and we are back to
standard one-step bootstrapping, where "one step" now means "one action and the environment's
response to it".

Of course, we could push this concept even further and do even more bootstrapping steps,
but for now we're fine with two.


## Effect on variance and bias

Our previous two variants sat at the two extremes of the variance-bias spectrum:

- **Monte Carlo sampling ([RwB]({{< relref reinforce-with-baseline >}})):**
  - high variance (depends on the entire remainder of the game)
  - low bias (uses true result of the game)
- **One-step bootstrapping ([A2C]({{< relref actor-critic >}})):**
  - low variance (only depends on the next state)
  - high bias (depends on potentially inaccurate value function)

Two-step bootstrapping falls in between these two options: there is more
**variance** than one-step bootstrapping, because now our opponent sits "in the loop" between
our current state and our bootstrapping target, whereas before the bootstrapping target depended
solely on the move we played. In fact, this is just Monte Carlo sampling again, but now only for
the single next move of the opponent, not for the entire remainder of the game! But clearly,
MC sampling for shorter episodes should have lower variance than for longer ones.

In terms of **bias**, since we look further into the future, the state we are evaluating
is closer to the end of the game and therefore likely has lower bias.
The one-step MC sampling for the opponent's move eliminates bias as well---we are using
the "true outcome" of the opponent's next action instead of relying on the value estimate.
Finally, by transporting value information backwards two steps at a time,
the high bias for early-game states should decrease much faster.

So there is hope that the new variant sits at a happy medium for both variance and bias.


## Implementation

It turns out that the implementation of this concept is actually both simpler and more efficient
than the previous one-step bootstrapping! This is because the update function already receives
all the successive "own" states \(s_i\) for a given game,
so we don't need to simulate any moves as we did to obtain the intermediate states \(s'_i\).
The values \(v(s_i)\) are also already computed, so we don't need an additional feedforward pass
anymore.

There is one modification needed: we need to know which moves ended up being terminal,
and for this we introduce a new boolean vector ``done`` which is true for the last move we made
in each game:

```py
def compute_rewards(num_moves: int, outcome: int) -> torch.Tensor:
    move_nr = torch.arange(num_moves, device=DEVICE)
    done = (move_nr == (num_moves - 1))
    if BOOTSTRAP_VALUE:
        # bootstrapping: sparse rewards, only assigned for the winning move
        return (done * outcome).float(), done
    else:
        # Monte Carlo sampling: rewards are discounted over the game
        return (outcome * REWARD_DISCOUNT**move_nr).flip(dims=(0,)), done
```

Note that ``done`` is true no matter whether the game ended in a win, loss, or draw.
The new vector is only needed for bootstrapping, but we return it in both cases just for
consistency. It's then passed into the ``update_policy`` function as an additional argument.

The bootstrapping portion of ``update_policy`` is now actually considerably simpler:

```py
    # --- Actor-Critic (A2C) with 2-ply value bootstrapping --- #

    # Next state value after 2 ply (only if not terminal)
    V_next = torch.roll(value.detach(), shifts=-1)  # (B,)

    # For terminal states, fix v(s_next) = 0
    V_next[done] = 0.0

    # Bootstrapping:
    # target value is reward from this move + discounted value of next state
    v_target = returns + REWARD_DISCOUNT * V_next
    
    # Higher weight for winning/losing moves since they are an important signal
    weight = torch.ones_like(returns)
    weight[returns == 1]  = 2.0
    weight[returns == -1] = 2.0
    
    value_loss = F.mse_loss(value, v_target, weight=weight, reduction='sum')
    advantage = (v_target - value).detach()
```

The call to ``torch.roll()`` shifts the value for state i+1 into position i.
(Note again the crucial ``detach()`` to avoid unwanted gradient backflow.)
At the end of a game, this rolls over into the first value of the next game, but we fix this by
zeroing out terminal state values in the next line.

The rest is then just the standard bootstrapping formula with weights as before, and
the remainder of the function stays as it was.
As always, refer to [the repo](https://github.com/c-f-h/connect-zero/blob/main/train/main.py) for
the complete code.

Note that we don't use the squared ``REWARD_DISCOUNT`` as the formula derived above suggested.
In fact this fixes a previous discrepancy in our code: REINFORCE with baseline applies the
discount rate only for our own moves (see ``compute_rewards`` above), so the same should be
true for our A2C implementation. This also aligns with the agent-environment interpetation
introduced above.

Also, we don't implement the precise formula for \(\overline R_i\) with the discounted \(R'_i\);
we just lump both rewards into the aggregate reward, and only one of them can be nonzero anyway.
Multiplying \(R'_i\) with the discount rate would weight losses slightly lower than wins,
and that doesn't seem desirable.


## Results

Let's run the [pretraining experiment]({{< relref evaluating-a2c >}}) again with the same settings
for the A2C algorithm with the new 2-ply bootstrapping method.

After training again for 200k games, here are the resulting plots:

{{< figure
    src="pretrain1-a2c-2p.png"
    width="100%"
    alt="Training progress for 200k games with A2C and 2-ply bootstrapping."
    caption="Training progress for 200k games with A2C and 2-ply bootstrapping."
>}}

These results are fantastic! The win rate exceeds 50% already after 40k games, which is faster than
either of the other methods. It also doesn't taper off before reaching 80% win rate and instead
keeps climbing steadily to almost 90% at the 200k games mark.

Overall, the run retains the lower variance properties of our first A2C run. Interestingly, value
loss ends up at around the same level as for one-step TD even though it has the very noisy
``RandomPunisher`` opponent "in the loop" and has to learn to deal with its variance.
Interestingly, it has a similar (but lower and narrower) initial "hump" as we saw in the RwB
results, which indicates there was less initial variance to overcome.

Variance reduction for the advantage is again good and shows no signs of slowing down after
200k games. Overall the run seems to combine the low-variance characteristics of the first
A2C attempt while reducing bias much more efficiently, as we hoped.

Let's run another round-robin tournament against the models we trained last time:

|                          | vs A2C (2p) | vs A2C (1p) | vs RwB   | vs RP     |
|--------------------------|:-----------:|:-----------:|:--------:|:---------:|
| **A2C (2p)**             |     ---     |  **57.7%**  | **65.0%**| **87.7%** |
| **A2C (1p)**             |    40.7%    |     ---     |   56.3%  |   74.2%   |
| **REINFORCE w/baseline** |    33.5%    |    41.6%    |   ---    |   68.1%   |
| **RandomPunisher**       |    11.8%    |    21.9%    |   30.7%  |   ---     |

These are very convincing numbers for the 2-ply bootstrapping,
improving upon both previous training strategies by a wide margin.

Clearly, we have a winner here. We'll use this simpler and more performant 2-ply bootstrapping
for all A2C runs from now on.
