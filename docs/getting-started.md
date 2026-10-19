# Getting started

## Start the web apps

```sh
helicon
```

Helicon starts the web apps and opens them in your browser, on the Home tab. (`helicon webApps` does
the same.) The Home tab is a diagram of the helical data processing workflow with a box for each
app: hover over a box to see what the app does, and click it to open it. The apps are also the tabs
along the top; each tab works on its own, so use only the ones you need.

```{image} images/home.png
:alt: The Home tab
:width: 90%
:align: center
```

To stop the apps, press `Ctrl-C` in the terminal where you started them.

## Give an app your data

Most apps ask "How to obtain the input …" with the same choices:

| Choice | Meaning |
|---|---|
| **server** | a file on the computer that runs Helicon: click *Browse…* or type its path |
| **upload** | a file from the computer that runs your browser |
| **url** | a web address, such as a file of the EMPIAR archive |
| **emd-xxxxx** | a map of the EMDB, by its ID (apps that take maps) |

*server* is not offered when Helicon runs on a hosting service, which cannot reach your files. On
your own computer it is the quickest way, because nothing is copied. Each app opens on example
data (the public 2D classes of EMPIAR-10940, or an EMDB map), so you can try it before you load
your own.

### 2D classes from RELION or cryoSPARC

Several apps work on the result of a 2D classification:

- the **class averages**: the `*_classes.mrcs` file of a RELION Class2D job, or the `.mrc` or
  `.mrcs` file of a cryoSPARC 2D classification job; and
- the **particle table** that says which class each segment of which filament got: the
  `*_data.star` file of RELION, or the `.cs` file of cryoSPARC.

The helical apps that relate classes to filaments (HelicalPitch, AbInitio3D, WhereIsMyClass) need
both. They need the particles to come from filaments (RELION: helical extraction), so that each
segment knows its filament and its position along it.

## Controls that help

- **Sliders:** click the thumb of a slider, then use the left and right arrow keys (or `A` and
  `D`) to move it one step at a time.
- **Plots:** hover to read values; the toolbar of a plot zooms, moves and saves it.
- **Parameters tab:** most apps have one, with the settings that rarely need a change.
- **Bookmarks:** the address in the browser records the tab and every setting that differs from
  its default, so a copy of the address brings a colleague to the same place.

## Open an app from the file browser

If you browse your files with [`helicon display`](display.md), a Class2D job offers the apps that
fit it as buttons, and opens one with the job's files already loaded.

## Next

Pick your app on [the Web apps page](apps/index.md).
