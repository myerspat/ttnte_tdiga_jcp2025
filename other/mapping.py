import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from igakit import cad
from matplotlib.patches import Polygon
from ttnte.cad import Patch
from ttnte.cad.surfaces import circle
from ttnte.iga import IGAMesh

if __name__ == "__main__":
    
    plt.rcParams.update({
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'axes.labelsize': 14,
    })

    radius = 0.54  # cm

    # Create patch
    patch = Patch(circle(radius), "Fuel")

    # Refine patch
    patch.refine(factor=3, degree=2)

    # Convert to geomdl
    patch.igakit2geomdl()

    fig, ax = plt.subplots()

    # Go through knot vectors
    xvector = np.unique(patch.knotvectors[0])
    yvector = np.unique(patch.knotvectors[1])

    for i in range(len(xvector) - 1):
        for j in range(len(yvector) - 1):
            X, Y = np.meshgrid(
                np.linspace(xvector[i], xvector[i + 1], 100),
                np.linspace(yvector[j], yvector[j + 1], 100),
            )

            coords = patch(
                np.concatenate([X.reshape((-1, 1)), Y.reshape((-1, 1))], axis=1)
            )

            X = coords[:, 0].reshape((100, 100))
            Y = coords[:, 1].reshape((100, 100))

            # Compute boundary points (clockwise)
            # Top edge
            top = np.column_stack((X[0, :], Y[0, :]))
            # Right edge
            right = np.column_stack((X[:, -1], Y[:, -1]))
            # Bottom edge (reversed)
            bottom = np.column_stack((X[-1, ::-1], Y[-1, ::-1]))
            # Left edge (reversed)
            left = np.column_stack((X[::-1, 0], Y[::-1, 0]))

            # Concatenate edges to form closed loop
            boundary = np.vstack((top, right[1:], bottom[1:], left[1:], top[0:1]))

            # Add polygon outline
            outline = Polygon(
                boundary,
                closed=True,
                edgecolor="black",
                facecolor=(0, 0, 0, 0.3) if i == 1 and j == 1 else "none",
                linewidth=1.5,
            )
            ax.add_patch(outline)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim((-radius - 0.01, radius + 0.01))
    plt.ylim((-radius - 0.01, radius + 0.01))
    plt.xlabel(r"$x(\hat{x}, \hat{y})$")
    plt.ylabel(r"$y(\hat{x}, \hat{y})$")
    plt.tight_layout()
    plt.savefig("./figs/physical.png", dpi=300)

    plt.clf()
    fig, ax = plt.subplots()
    for i in range(len(xvector) - 1):
        for j in range(len(yvector) - 1):
            X, Y = np.meshgrid(
                np.linspace(xvector[i], xvector[i + 1], 100),
                np.linspace(yvector[j], yvector[j + 1], 100),
            )

            # Compute boundary points (clockwise)
            # Top edge
            top = np.column_stack((X[0, :], Y[0, :]))
            # Right edge
            right = np.column_stack((X[:, -1], Y[:, -1]))
            # Bottom edge (reversed)
            bottom = np.column_stack((X[-1, ::-1], Y[-1, ::-1]))
            # Left edge (reversed)
            left = np.column_stack((X[::-1, 0], Y[::-1, 0]))

            # Concatenate edges to form closed loop
            boundary = np.vstack((top, right[1:], bottom[1:], left[1:], top[0:1]))

            # Add polygon outline
            outline = Polygon(
                boundary,
                closed=True,
                edgecolor="black",
                facecolor=(0, 0, 0, 0.3) if i == 1 and j == 1 else "none",
                linewidth=1.5,
            )
            ax.add_patch(outline)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim((0 - 0.01, 1 + 0.01))
    plt.ylim((0 - 0.01, 1 + 0.01))
    plt.xlabel(r"$\hat{x}$")
    plt.ylabel(r"$\hat{y}$")
    plt.tight_layout()
    plt.savefig("./figs/parametric.png", dpi=300)

    plt.clf()
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(
        np.linspace(-1, 1, 100),
        np.linspace(-1, 1, 100),
    )

    # Compute boundary points (clockwise)
    # Top edge
    top = np.column_stack((X[0, :], Y[0, :]))
    # Right edge
    right = np.column_stack((X[:, -1], Y[:, -1]))
    # Bottom edge (reversed)
    bottom = np.column_stack((X[-1, ::-1], Y[-1, ::-1]))
    # Left edge (reversed)
    left = np.column_stack((X[::-1, 0], Y[::-1, 0]))

    # Concatenate edges to form closed loop
    boundary = np.vstack((top, right[1:], bottom[1:], left[1:], top[0:1]))

    # Add polygon outline
    outline = Polygon(
        boundary,
        closed=True,
        edgecolor="black",
        facecolor=(0, 0, 0, 0.3),
        linewidth=1.5,
    )
    ax.add_patch(outline)

    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim((-1 - 0.01, 1 + 0.01))
    plt.ylim((-1 - 0.01, 1 + 0.01))
    plt.xlabel(r"$\tilde{x}(\hat{x})$")
    plt.ylabel(r"$\tilde{y}(\hat{y})$")
    plt.tight_layout()
    plt.savefig("./figs/parent.png", dpi=300)
