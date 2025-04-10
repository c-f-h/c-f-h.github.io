---
title: 'Preparing the Data'
date: 2025-04-09T20:57:38+02:00
tags: ['Math', 'C++', 'ML']
categories: ["The Triangle Project"]
---

With the [triangle self-intersection algorithm]({{< relref refining-intersections >}}) ready to go, we
can start gathering the training data for our machine learning setup. But first we have to think about how
exactly we want to represent it.

## Canonical triangles

The [curved triangles]({{< relref the-triangle-problem >}}) we work with are specified by six 3D vectors, so that
would mean 18 floating point numbers as our input data.
But an important insight is that whether a triangle intersects itself doesn't change when we rotate it, translate it,
or uniformly scale it---it's well known that affine transformations of spline control points result in affine transformations of the surface itself.

So we can transform any triangle into the following canonical form without changing the intersection property:
Put the first vertex at \((0,0,0)\), the second at \((1,0,0)\), and the third at \((x,y,0)\) with \(y>0\).
The remaining three edge control points are transformed alongside, but we don't put any special constraints on them.
This means that the only relevant input data for such a canonicalized triangle are the three edge control points
plus the \(x\) and \(y\) coordinates of the third vertex. That's down to 11 floats from 18 originally, a big reduction!

How do we implement this transformation? First we subtract \(P_{200}\) from all six vectors. Then we rotate the triangle such
that the first axis lies along the x axis and the second axis lies in the (x,y)-plane.
Finally we scale the triangle such that the transformed \(P_{020}\) lies at \((1,0,0)\).

The rotation matrix is the slightly tricky part, but with some basic linear algebra it's pretty straightforward.
A good thing to remember is that if you have three unit vectors which follow the
[right-hand rule](https://en.wikipedia.org/wiki/Right-hand_rule),
you can stick them as column vectors into a \(3\times3\) matrix and get a rotation matrix which rotates the
x, y and z axes into that new configuration
given by your three vectors. And a second useful fact is that the inverse of a rotation matrix is just its transpose.
So that means if you put your three vectors into the matrix as row vectors instead, you get a matrix which rotates
your three chosen vectors back to the standard x, y and z axes.

Here's the C++ function which implements the transformation to a canonical triangle.

```C++
CurvedTriangle CurvedTriangle::canonicalize() const
{
    Eigen::Vector3d u = P020 - P200;
    Eigen::Vector3d v = P002 - P200;

    Eigen::Vector3d newX = u / u.norm();    // Normalized vector along the new x-axis

    Eigen::Vector3d w = u.cross(v);         // Vector perpendicular to the triangle plane
    Eigen::Vector3d newZ = w / w.norm();    // Normalized vector along the new z-axis

    // Define the new y-axis using the cross product to ensure right-handed system
    Eigen::Vector3d newY = newZ.cross(newX);

    // Construct the inverse rotation matrix which rotates (newX, newY, newZ)
    // into the standard coordinate system.
    Eigen::Matrix3d R_inv;
    R_inv.row(0) = newX;
    R_inv.row(1) = newY;
    R_inv.row(2) = newZ;

    // Translate and rotate all six vectors
    std::array<Eigen::Vector3d, 6> result{
            R_inv * (P200 - P200), // = (0, 0, 0)
            R_inv * (P020 - P200), // = (u_norm, 0, 0)
            R_inv * (P002 - P200), // = (x, y, 0) with y > 0
            R_inv * (P110 - P200),
            R_inv * (P101 - P200),
            R_inv * (P011 - P200),
    };

    // Scale to unit length for first axis
    const double scale = result[1][0];
    for (auto&& v : result)
        v /= scale;

    return CurvedTriangle{result};
}
```

## The data labels

Obviously, at minimum we just need a single bit to tell us if each canonicalized triangle does self-intersect or not.
The format I used in practice has a bit more detail because I planned to give more structured information to the
ML algorithm. Although at first we'll only train a simple binary classifier, making this data format overkill,
having it available could later allow us to train a more nuanced representation.

The idea is to provide a kind of "heatmap" in the form of a 16x16 bitmap representing the rasterized (u,v)-space.
It has bits set where the intersection curve, if one does exist, lies in the parameter domain.
Both the \((u_1,v_1)\) and the \((u_2,v_2)\) components are rasterized into the same bitmap to keep the data small.
The rasterization is implemented by linearly interpolating between each pair of curve points and mapping into
the discretized (u,v)-space.

```goat {caption="A 16x16 bitmap showing the location of an intersection curve in parameter space. The upper right half is always empty, but for simplicity we store it anyway."}
o
o o 
o o o
o o o o
o o o o o
o o o o o *
o o o o o * o
o o o o o * o o
o o o o o * o o o
o o o o o * o o o o
o o o o o * o o o o o
o o o o o * o o o o o o
o o o o o * o o o o o o o
o o o o o o o o o o o o o o
o o o o o o o o o o o o o o o
o o o o o o o o o o o o o o o o
```

Since each pixel is either 0 or 1, we can compress the 256 pixels of such a bitmap into 32 bytes.
To convert this to a single 0/1 label, we simply check if the entire bitmap is 0 or there are any pixels lit.

## The binary file format

That's all we need. To populate our dataset, we generate six random vectors with coordinates uniformly chosen in \([-1,1]\),
canonicalize them, check for intersections, and compute the bitmap of the intersection curve.
Each sample is written to a binary file that has eleven 32-bit floats (44 bytes) for the canonicalized triangle coordinates
plus 32 bytes for the bitmap for each record, for a total of 76 bytes per record.
A simple binary format like this saves space and is easy and quick to read and write in both C++ and Python.

I multithreaded the generation of these binary data files to speed things up and collected the results in files
`triangles.000`, `triangles.001`, and so on, with each file containing 10,000 samples or 760,000 bytes.
Currently I have 90 of these files, giving us 900,000 samples to train with; later we'll augment our data set further.
All training samples are then concatenated into `triangles.dat`.
I also generated another 20,000 samples for the validation set, `triangles.val`.

Surprisingly, the simple random triangle generation leads to a data set that is very well balanced
between intersecting and non-intersecting triangles: of the 20,000 validation samples, 10,191 are positives.

With the input data set up, we can finally set up a model and start training!
