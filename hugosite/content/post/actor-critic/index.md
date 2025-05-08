---
title: 'Actor-Critic Algorithms'
date: 2025-05-08T21:56:00+02:00
tags: ['ML', 'RL', 'Math']
categories: ['Connect-Zero']

draft: true
---

After [implementing and evaluating]({{< relref implementing-rwb >}}) REINFORCE with baseline,
we found that it can produce strong models, but takes very long to learn a good approximation to
the value function due to the high variance of the
[Monte Carlo samples]({{< relref reinforce-with-baseline >}}).

