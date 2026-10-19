# The Web apps

The web apps are tabs of one page. Start them with `helicon` (see [Getting started](../getting-started.md))
and pick a tab, or click its box on the Home tab. Each app works on its own.

```{image} ../images/home.png
:alt: The Home tab
:width: 90%
:align: center
```

## Which app do I need?

| You have | You want | Use |
|---|---|---|
| 2D classes from RELION or cryoSPARC, with the table of particles | the **pitch and twist** of the filaments | [HelicalPitch](helicalpitch.md) |
| the same | a **3D map** from the classes, and to tell filaments of different **types** apart | [AbInitio3D](abinitio3d.md) |
| one 2D class average or image | the helical parameters and a **3D map** from that one image | [Denovo3D](denovo3d.md) |
| a 2D class average or a 3D map | to **index the layer lines** of its power spectrum (the classical way) | [HILL](hill.md) |
| a 3D map | its **twist, rise and symmetry** from a cylindrical projection | [HI3D](hi3d.md) |
| a 2D class average | to compare it with **projections of known helical structures** of the EMDB | [HelicalProjection](helicalprojection.md) |
| a helical lattice | to convert between a **2D lattice** and a **helical lattice** | [HelicalLattice](helicallattice.md) |
| 2D classes of a 2D classification | to see **where in the micrographs** the filaments of a class are | [WhereIsMyClass](whereismyclass.md) |

A common order for a new data set: look at the 2D classes with **WhereIsMyClass**, find the pitch with
**HelicalPitch**, find the twist and symmetry with **HILL** or **Denovo3D**, and make a first 3D map
with **Denovo3D** or **AbInitio3D**; refine it in your 3D reconstruction software.

```{toctree}
:hidden:

helicalpitch
abinitio3d
denovo3d
hill
hi3d
helicalprojection
helicallattice
whereismyclass
```
