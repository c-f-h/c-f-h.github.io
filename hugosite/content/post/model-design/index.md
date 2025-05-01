---
title: 'Model Design for Connect 4'
date: 2025-04-28T22:30:00+02:00
tags: ['Python', 'PyTorch']
categories: ['Connect-Zero']
---

With the [fearsome RandomPunisher]({{< relref random-punisher >}}) putting our
[first Connect 4 toy model]({{< relref entropy-regularization >}})
in its place, it's time to design something that stands a chance.

## A design based on CNNs

It's standard practice for board-game playing neural networks to have at least a few
**convolutional neural network (CNN)** layers at the initial inputs. This shouldn't come as
a surprise: the board is a regular grid, much like an image, and CNNs are strong performers
in image processing. In our case, it will allow the model to learn features like
"here are three of my pieces in a diagonal downward row" which are then automatically
applied to every position on the board, rather than having to re-learn these features
individually at each board position.

This isn't the place for me to go into a full-blown writeup on convolutional neural
networks, but the idea is to learn small matrices or tensors called kernels which slide over
the input image and transform each local neighborhood of pixels in the same way.
This saves parameters compared to a fully connected layer and allows the model to learn
uniform features which apply all over the board.

Have a look at this diagram, which shows a 3x3 kernel being moved diagonally over our
game board with one cell for padding on each side:

<figure>
<svg class="diagram" xmlns="http://www.w3.org/2000/svg" width="450" height="400" viewBox="0 0 450 400">
  <defs>
    <pattern id="cell" width="50" height="50" patternUnits="userSpaceOnUse">
      <rect width="50" height="50" fill="white" stroke="#666" stroke-width="1"/>
    </pattern>
  </defs>
  <rect x="0" y="0" width="450" height="400" fill="url(#cell)"/>
  <!-- highlight the original 6x7 Connect-4 area inside the padding -->
  <rect x="50" y="50" width="350" height="300"
        fill="none" stroke="black" stroke-width="3"/>
  <!-- draw 7 kernels moving diagonally -->
  <g fill="blue" fill-opacity="0.2" stroke="gray" stroke-width="2">
    <rect x="0"    y="0"      width="150" height="150"/>
    <rect x="50"   y="50"     width="150" height="150"/>
    <rect x="100"  y="100"    width="150" height="150"/>
    <rect x="150"  y="150"    width="150" height="150"/>
    <rect x="200"  y="200"    width="150" height="150"/>
    <rect x="250"  y="250"    width="150" height="150"/>
  </g>
</svg>
<figcaption>
<p>Six instances of a 3x3 convolution kernel being overlaid on the game board
with one cell of padding on every side.</p>
</figcaption>
</figure>

The architecture of the network is relatively standard and in part inspired by the
[AlphaZero model](https://doi.org/10.1126/science.aar6404):
it uses an initial convolutional layer, then three residual
blocks consisting of two convolutional layers each, a pooling layer,
and finally a MLP head with two hidden fully connected layers with 128 neurons each.
As the activation function we use ReLU throughout.

### Initial feature extraction

To help our model out a bit, we separate the original board out into **three input
channels**: one for our pieces, one for the opponent's pieces, and one for empty cells.
Each of these channels will have 1 for the corresponding feature and 0 otherwise.

The number of channels is an important parameter to tune depending on the complexity of
the problem. Since Connect-4 isn't that complicated, 64 channels are probably sufficient;
for comparison, AlphaZero uses 256 channels on each CNN layer.

Each convolutional layer uses 64 channels, a kernel size of 3x3, and padding 1.
These are standard settings which ensure that the output has the same dimensions
as the input.[^1]

[^1]: You can convince yourself of that if you refer back to the previous
diagram and count how many of the blue 3x3 squares can fit into the padded board in
each direction; it's 6x7 again.

The very first convolutional layer serves as a **feature extractor** and maps the three
input channels to 64 output channels while keeping the board dimension at (6, 7).

### Residual blocks

After that, we have three **residual blocks**.
Each one of them consists of two conv layers with 64 input and output channels such
that they maintain both the spatial dimension and the number of channels.
The input to the residual block is added back in to the output of the second conv
layer before the activation function, which is the defining feature of
residual blocks.
This both improves training robustness, because gradient information can flow "around"
the block in addition to through it, and often gives the block an easier target to learn
because it only needs to learn the deviation from the identity function. The
schematic below shows the precise behavior of the data flow.

Afterwards, we do some kind of pooling to reduce the number of features. We could do
global or column-wise average pooling, but to give the model a bit more to work with,
we instead use another convolutional **downsampling layer**:
it has a kernel size of (6,1), meaning that each convolution kernel
covers an entire column of the board.
With padding 0, this means that our data is reduced from
(64, 6, 7) to (64, 1, 7). This is illustrated in the following diagram:

<figure>
<svg class="diagram" xmlns="http://www.w3.org/2000/svg" width="350" height="300" viewBox="0 0 350 300">
  <defs>
    <pattern id="cell" width="50" height="50" patternUnits="userSpaceOnUse">
      <rect width="50" height="50" fill="white" stroke="#666" stroke-width="1"/>
    </pattern>
  </defs>
  <rect x="0" y="0" width="350" height="300" fill="url(#cell)"/>
  <!-- highlight the original 6x7 Connect-4 area inside the padding -->
  <rect x="0" y="0" width="350" height="300"
        fill="none" stroke="black" stroke-width="3"/>
  <!-- draw 7 kernels moving diagonally -->
  <g fill="blue" fill-opacity="0.2" stroke="gray" stroke-width="2">
    <rect x="100"    y="0"      width="50" height="300"/>
  </g>
</svg>
<figcaption>
<p>The downsampling convolution with filter size (6,1) collapses an entire column into
one value.</p>
</figcaption>
</figure>

### MLP head

This downsampled data is then flattened to a vector of length 64 * 7 = 448. 
To obtain the final output policy, we attach an **MLP head** consisting of
three fully connected layers with output sizes 128, 128, and 7, respectively.
The final layer doesn't have an activation function since we want to output raw logits.

### Normalization

Using some kind of **layer normalization** is usually a good idea as it makes the training
process more robust. However, batch normalization, the most common choice, is not ideal
here due to the way it interacts with an on-policy RL algorithm. BatchNorm
behaves differently during training, where it uses the stats of the current batch, and
evaluation (which we use to sample games), where it uses the collected aggregate stats
to normalize the inputs. As discussed earlier, REINFORCE really relies on using
the same distribution during play and training, and I observed significantly more noisy
behavior when using BatchNorm.

Instead, we use other popular options: for the conv layers, we use **group normalization**
with a group size of 16, and for the fully connected layers **layer normalization**,
which normalizes the activations of each sample with respect to itself only, without
keeping any running stats. As is usual, we remove the bias term on conv and linear layers
before normalization layers because those come with their own bias.

Here's a schematic of the entire network architecture:

```goat {caption="Schematic of the residual CNN architecture."}
+---------------------------+
|   Input Board State       |  [B, 6, 7]
+------------+--------------+
             |
             v
+---------------------------+
|   board_to_channels       |  [B, 3, 6, 7]
+------------+--------------+
             |
             v
+---------------------------+
|     Feature Extraction    |
|   Conv(3, 64, k=3, p=1)   |
|   GroupNorm(16, 64)       |
|   ReLU                    |  [B, 64, 6, 7]
+------------+--------------+
             |
             |------------------------------------.
             v                                    |
+---------------------------+                     |
|     ResBlock 1            |                     |
|   Conv(64,64)+GN+ReLU     |                     v Residual connection
|   Conv(64,64)+GN          |  [B, 64, 6, 7]      |
+------------+--------------+                     |
             |                                    |
             +<-----------------------------------.
             |
             v  ReLU
             |
             |------------------------------------.
             v                                    |
+---------------------------+                     |
|     ResBlock 2            |                     |
|   Conv(64,64)+GN+ReLU     |                     v Residual connection
|   Conv(64,64)+GN          |  [B, 64, 6, 7]      |
+------------+--------------+                     |
             |                                    |
             +<-----------------------------------.
             |
             v  ReLU
             |
             |------------------------------------.
             v                                    |
+---------------------------+                     |
|     ResBlock 3            |                     |
|   Conv(64,64)+GN+ReLU     |                     v Residual connection
|   Conv(64,64)+GN          |  [B, 64, 6, 7]      |
+------------+--------------+                     |
             |                                    |
             +<-----------------------------------.
             |
             v  ReLU
             |
             v
+---------------------------+
|   Downsampling            |
| Conv(64, 64, k=(6,1))     |  [B, 64, 1, 7]
| Flatten                   |  [B, 448]
+------------+--------------+
             |
             v
+---------------------------+
|   MLP Head                |
| Linear(448, 128)+LN+ReLU  |
| Linear(128, 128)+LN+ReLU  |
| Linear(128, 7)            |  [B, 7]
+------------+--------------+
             |
             v
+---------------------------+
|   Output Logits           |  [B, 7]
+---------------------------+
```

The network has a total of around **356k parameters**.
There are obvious ways we could tune this design later:
the number of convolution filters, residual blocks, and depth and width of
the MLP head are all parameters we could tweak to adjust model complexity.


## Implementation

The model described above is straightforward to implement using standard PyTorch tools:

```py
def board_to_channels(board: torch.Tensor) -> torch.Tensor:
    """Converts a board tensor (0=empty, 1=p1, -1=p2) to 3 channel representation."""
    # Create channels: [batch, channels, rows, cols]
    channels = torch.zeros((board.size(0), 3, 6, 7),
                           dtype=torch.float32, device=board.device)

    channels[:, 0, :, :] = (board == 1).float()    # Current player's pieces
    channels[:, 1, :, :] = (board == -1).float()   # Opponent's pieces
    channels[:, 2, :, :] = (board == 0).float()    # Empty cells
    return channels

class Connect4CNN_Mk4(nn.Module):
    """CNN/ResNet model with global average pooling and MLP classifier head."""
    def __init__(self):
        super().__init__()

        # Extract 64 feature channels from the 3 input channels of the board.
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(16, 64),
            nn.ReLU(),
        )

        def make_resblock():
            return nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(16, 64),
            )
        
        self.resblock1 = make_resblock()
        self.resblock2 = make_resblock()
        self.resblock3 = make_resblock()

        self.downsample = nn.Conv2d(64, 64, kernel_size=(6,1), stride=1,
                                    padding=0, bias=False)

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 128, bias=False),
            nn.LayerNorm(128),
            nn.ReLU(),

            nn.Linear(128, 7),      # output raw logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if x.ndim == 2:
            x = x.unsqueeze(0)      # Add temporary batch dimension

        # Convert board to channel representation (3 channels)
        x = board_to_channels(x)       # [B, 3, ROWS, COLS]
        x = self.feature_extractor(x)  # [B, 64, ROWS, COLS]

        # Apply residual CNN blocks (note the "+x" connection before ReLU)
        x = F.relu(self.resblock1(x) + x)
        x = F.relu(self.resblock2(x) + x)
        x = F.relu(self.resblock3(x) + x)

        # Downsample columnwise: [B, 64, 1, 7] -> [B, 7 * 64]
        x = self.downsample(x).view(-1, 7 * 64)

        # Fully connected layers (MLP head, outputs logits): [B, 7]
        x = self.fc_layers(x)

        if len(original_shape) == 2:
            x = x.squeeze(0)
        return x
```

## First evaluation

We can now attempt training this residual network structure using the self-play
loop from [a previous post]({{< relref policy-collapse >}}).
A simple improvement worth implementing is to copy the current model to the checkpoint one
only if it achieves a positive win rate over it, say, 52%.

With careful manual annealing of the learning rate (from 1e-3 through 1e-4 to 1e-5) and of
the entropy bonus (from 0.05 to 0.03), we can push the win rate against the
``RandomPunisher``
to just above 50%. That's a significant improvement over the MLP model as mentioned at the
beginning of this post (35%), but in the end we want to do more than break even
against a model that behaves just one or two steps removed from random.

The issue is that the basic REINFORCE algorithm is well known to be quite noisy, and this
is also evident in the training plots as well as in the difficulty of choosing good
hyperparameters for the optimization.[^2] If we want more robust training and better final
performance, we'll have to graduate to
[more advanced RL algorithms.]({{< relref reinforce-with-baseline >}})

[^2]: It's hard to make definitive statements of the kind that no better performance is
possible using the present setup. Certainly, with much longer training and very careful
hyperparameter tweaking, we could eke out a higher win rate. But in the end we are
always fighting against the limitations of the basic algorithm we are using.
