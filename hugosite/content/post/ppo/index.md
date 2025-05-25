---
title: 'Proximal Policy Optimization (PPO)'
date: 2025-05-25T12:48:00+02:00
tags: ['Math', 'Python', 'PyTorch', 'ML', 'RL', 'PPO']
categories: ['Connect-Zero']
---

The next step after [implementing A2C]({{< relref implementing-a2c >}}) is to move on to
Proximal Policy Optimization (PPO).
Introduced in a [paper by OpenAI researchers](https://arxiv.org/abs/1707.06347) in 2017, it has
become a very popular RL algorithm since.
It can be understood as a simplified variant of Trust Region Policy Optimization (TRPO), and one
of its main advantages is **improved sample efficiency**: although it is an on-policy algorithm, it
can robustly learn multiple times from a batch of generated samples, unlike A2C and REINFORCE.

## The PPO objective function

### A different formulation for policy gradients

As we did previously, let's look at a single state \(s\) where we took action \(a\), resulting
in an advantage \(A\) which is positive or negative depending on how beneficial the action was.
(Advantage computation will use the same TD approach as in [A2C]({{< relref actor-critic >}}),
in particular the improved [two-ply bootstrapping variant]({{< relref multistep-bootstrapping >}})
we later introduced.)

In A2C, the policy part of the objective function corresponded to maximizing

\[
    L^\textsf{A2C} = A \log p_a(s) \to \max,
\]

where \(p_a(s)\) was the probability for taking action \(a\) in state \(s\) under our current
policy.[^1]

[^1]: In A2C we had this term with a negative sign since we used the usual loss minimization
formulation, but PPO is typically introduced as a maximization problem, so we stick to that
convention here.

A first observation is that the policy gradient with respect to the model parameters is

\[
    \nabla L^\textsf{A2C} = A \frac{\nabla p_a(s)}{p_a(s)}
    = A \frac{\nabla p_a(s)}{p^{\textsf{old}}_a(s)}.
\]

Here we introduced the "old" probability \(p^{\textsf{old}}_a(s) := p_a(s)\) at the start of the
current update; we take \(p^\textsf{old}\) as "frozen" in the sense that it does not depend on
the model parameters, but is just the constant value of the probability before the current
policy update. (In PyTorch, we would describe this using ``detach()``.)

Therefore it's clear that we can introduce an objective[^2] based on the probability ratio,

\[
    L^\textsf{CPI} := A \frac{p_a(s)}{p^{\textsf{old}}_a(s)} = A r
    \quad
    \text{with } r := \frac{p_a(s)}{p^{\textsf{old}}_a(s)}
\]
which results in the same policy gradient, \(\nabla L^\textsf{CPI} = \nabla L^\textsf{A2C}\).

[^2]: The superscript CPI is short for "conservative policy iteration" and was used in the original PPO
paper in reference to [(Kakade and Langford 2002)](https://dl.acm.org/doi/abs/10.5555/645531.656005)
where this functional first appeared.

### The clipped PPO surrogate objective

The basic idea of PPO is now to do several policy updates on a batch of collected samples using
this type of gradient. However, if we simply apply the basic A2C policy gradient multiple times,
we could drift too far from the initial policy, leading to unstable training. So the main innovation
of PPO is to introduce a clipping factor \(\epsilon\), usually chosen in \(\epsilon\in[0.1,0.3]\),
and to instead maximize the **clipped objective function**

\[
    L^\textsf{PPO} := \min\left\{ Ar, A \operatorname{clip}(r, 1-\epsilon, 1+\epsilon)  \right\}.
\]

Here

\[
    \operatorname{clip}(x, a, b) = \min(\max(x, a), b)
\]

is the clipping function which restricts \(x\) to remain within the interval \([a,b]\).

At first glance it's quite tricky to parse what exactly this formula achieves.
We can gain a better understanding by noting that \(A\) is the only term which can
become negative here and distinguishing two cases by the sign of \(A\).

If \(A\ge0\), we can pull it out of the minimum and obtain

\[
    \begin{align*}
    L^\textsf{PPO}
    &= A \min\left\{ r, \operatorname{clip}(r, 1-\epsilon, 1+\epsilon) \right\} \\
    &= A \operatorname{clip}(r, r, 1+\epsilon)    \\
    &= A \min(r, 1+\epsilon).
    \end{align*}
\]

The second equality needs some squinting (or careful working out of the minima and maxima),
but in essence, the outer \(\min\) undoes the lower
limit of the clipping function.
The resulting function is identical to \(Ar\) for \(r\le1+\epsilon\) and constant beyond that.

{{< figure src="ppo-clipping.svg" alt="Plot of PPO clipping for A > 0."
  align="center"
  caption="Plot of the PPO clipping function for A = 1 and ε = 0.2."
>}}

Similarly, if \(A<0\), we can pull it out but have to flip the minimum to a maximum,
and by a similar derivation we get

\[
    \begin{align*}
    L^\textsf{PPO}
    &= A \max\left\{ r, \operatorname{clip}(r, 1-\epsilon, 1+\epsilon) \right\} \\
    &= A \operatorname{clip}(r, 1-\epsilon, r)    \\
    &= A \max(r, 1-\epsilon).
    \end{align*}
\]
This time, the function is identical to \(Ar\) for \(r\ge1-\epsilon\) and constant below that.

This gives a different but equivalent way to write the PPO objective which is somewhat easier to
interpret, namely,

\[
    L^\textsf{PPO} = A
    \begin{cases}
      \min(r, 1+\epsilon)         & \text{if } A \ge 0,   \\
      \max(r, 1-\epsilon)         & \text{if } A < 0.
    \end{cases}
\]

In other words, if the advantage is positive, we incentivize an increase in \(p_a(s)\),
but only to a maximum of \((1+\epsilon) p^{\textsf{old}}_a(s)\), beyond which the gradient
becomes zero.
On the other hand, if \(A<0\),  \(p_a(s)\) may drop at most to
a minimum of \((1-\epsilon) p^{\textsf{old}}_a(s)\).
So overall the new policy shouldn't deviate too much from the old one, and
\(\epsilon\) allows us to control the maximum deviation.
In a sense, this prevents overfitting to the sampled batch even when performing multiple
update steps.

This clipping range would be respected exactly if we did an exact maximization of
\(L^\text{PPO}\). In practice, of course, we take a gradient step instead and may overshoot
our desired range. Therefore implementations often include an "early stopping" check based
on the Kullback-Leibler divergence from the old to the new policy distribution,
or [an approximation of it](http://joschu.net/blog/kl-approx.html).
We'll see one version of that once we actually implement PPO.

## The structure of the PPO algorithm

Now that we understand the construction of the clipped PPO objective function, let's put it in
the context of the overall learning loop.
One iteration of the PPO algorithm consists of the following steps:

1. Collect a batch of samples using the current policy.
2. Compute the policy distribution \(p^\textsf{old}\) for these samples from the current
   policy and keep it frozen.
3. Compute advantage \(A\) for each sample.
3. For a small number of PPO epochs (say, 4):
   - compute the PPO loss \(-L^\text{PPO}\), augmented with other terms such as value loss and
     [entropy bonus]({{< relref entropy-regularization >}});
   - update the policy by taking an optimizer step using the gradient of the total loss;
   - (optionally) check for early stopping based on policy divergence.

So PPO takes a number of "mini-epochs" within each iteration, using the same set of samples,
but leaves the old reference policy \(p^\textsf{old}\) unchanged during these epochs.
Only the numerator of the ratio \(r\) changes from iteration to iteration.
Common choices for the number of PPO epochs range from 4 to 10.

Implementations typically also include some sort of mini-batching within the PPO update, but
we skip this here since our training loop already produces reasonably sized batches for PPO
to work on.

The total loss function to be minimized in each PPO mini-epoch,
including value loss and entropy bonus, is

\[
    \ell^{\textsf{PPO}} = \underbrace{-L^\textsf{PPO}}_{\textsf{policy loss}}
    + \underbrace{\alpha (R + \gamma v(s') - v(s))^2}_{\textsf{value loss}}
    - \underbrace{\beta H(p(s))}_{\textsf{entropy bonus}}.
\]

In the next post we'll implement PPO for our Connect 4 training task and evaluate it against the
previous algorithms.
