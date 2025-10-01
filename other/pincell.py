import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
from igakit import cad
from matplotlib.transforms import Affine2D
from ttnte.cad import Patch
from ttnte.cad.surfaces import circle
from ttnte.iga import IGAMesh

if __name__ == "__main__":
    radius = 0.54  # cm
    pitch = 1.26  # cm

    # Define circle
    c = circle(radius)

    # Get box
    l0 = cad.line((-pitch / 2, -pitch / 2), (pitch / 2, -pitch / 2))
    l1 = cad.line((pitch / 2, -pitch / 2), (pitch / 2, pitch / 2))
    l2 = cad.line((-pitch / 2, pitch / 2), (pitch / 2, pitch / 2))
    l3 = cad.line((-pitch / 2, -pitch / 2), (-pitch / 2, pitch / 2))

    # Create patches
    mesh = IGAMesh()
    mesh.add_patch(Patch(cad.ruled(c.boundary(0, 0), l0), "Water"))
    mesh.add_patch(Patch(cad.ruled(c.boundary(1, 1), l1), "Water"))
    mesh.add_patch(Patch(cad.ruled(c.boundary(0, 1), l2), "Water"))
    mesh.add_patch(Patch(cad.ruled(c.boundary(1, 0), l3), "Water"))
    mesh.add_patch(Patch(c, "Fuel"))

    # Connect patches
    mesh.connect()

    # Finalize mesh
    mesh.finalize()
    print(mesh)

    # Plot final mesh
    ax = mesh.plot(plot_ctrlpts=False)
    ax.legend_.remove()
    plt.tight_layout()
    plt.xlabel(None)
    plt.ylabel(None)
    plt.axis("off")
    plt.savefig("./figs/pincell.png", dpi=300)

    # Translate
    quadmeshes = [m for m in ax.collections if isinstance(m, mcoll.QuadMesh)]
    dxs = [0, 0.1, 0, -0.1]
    dys = [-0.1, 0.0, 0.1, 0.0]

    for dx, dy, mesh in zip(dxs, dys, quadmeshes):
        new_transform = Affine2D().translate(dx, dy) + ax.transData

        array = mesh.get_array()
        cmap = mesh.get_cmap()
        norm = mesh.norm
        shading = mesh._shading  # 'flat' or 'auto'

        # This keeps the original mesh but applies a shift visually
        mesh.set_transform(Affine2D().translate(dx, dy) + ax.transData)

        # Add it back to the axis if removed
        if mesh not in ax.collections:
            ax.add_collection(mesh)

    plt.xlim((-(1.26 + 0.2) / 2, (1.26 + 0.2) / 2))
    plt.ylim((-(1.26 + 0.2) / 2, (1.26 + 0.2) / 2))
    plt.savefig("./figs/pincell_patches.png", dpi=300)
