---
title: 'Connect 4'
date: 2025-04-20T09:29:36+02:00
categories: ["Connect-Zero"]

connect4: true
onnx: true
---

<div id="game-container1" class="connect4-container"
    data-human="1" data-cpu="2"
    data-random-first-player="true"
    data-onnx-model="export.onnx"
></div>

The computer opponent is a neural network trained using reinforcement learning.
It was exported to ONNX and now runs right here in your browser.
See [Connect-Zero]({{<relref connect-zero>}}) and the follow-up posts for details.

This isn't a particularly strong model yet: it still has many tactical and
strategic blind spots. We'll train much stronger models over the course of the series.
