# Browser (Shinylive) builds of the web app tabs

Exports selected tabs of `helicon webApps` to static, serverless bundles that
run entirely in the visitor's browser via [Shinylive](https://shiny.posit.co/py/docs/shinylive.html)
(Pyodide/WebAssembly) — no Python server, hostable from plain static storage
(GitHub Pages, S3, ...).

## Why only some tabs qualified, and how each blocker was lifted

A tab qualifies only once nothing on its call path *needs*, unconditionally:

- **numba** — Pyodide has no JIT backend for it (no llvmlite/LLVM wasm
  build), so a numba-accelerated function silently falls back to the
  pure-Python path (`lib/numba_compat.py`'s `lazy_jit`). That path is not a
  slow approximation but ~500x slower (measured on
  `helicon.apply_helical_symmetry`, a 48³ volume, warm: 0.025s with numba vs
  12.7s without, same checksum) — unusable for interactive use, and Pyodide
  typically costs another 2-5x on top of that. The fix is a vectorized
  numpy/scipy implementation picked by an `_HAS_NUMBA` dispatch at the call
  site, so the tab no longer needs numba at all: `_apply_helical_symmetry_np`
  (`lib/transforms.py`, with analytical Z-window culling), `_period_scores_np`
  (`webApps/lib/helical_pitch_context.py`, alongside the three HMM kernels
  already vectorized there), and `denovo3d_solver.py`'s
  `build_A_helical_sym_matrix`/`build_A_data_matrix` `loop_kji` kernels are all
  done this way now — every numba kernel on every bundled tab's call path has a
  numpy fallback. Multi-candidate searches also use single-core algorithmic fast
  paths so browser runs finish interactively: `denovo3D` runs `score_only`
  parameter scans (skipping 3D volume/projection rendering until redrawing the
  top candidates) with vectorized Gaussian rendering (`solver_gauss_analytic.py`),
  and `HILL` vectorizes layer-line height/mask evaluation and ring-normalized
  contrast scoring across `(twist, rise)` batches (`lib/layerlines.py`,
  `webApps/lib/layerline_index.py`).
- **finufft** — a compiled extension with no Pyodide/wasm wheel;
  `micropip.install` cannot satisfy it. `lib/transforms.fft_rescale` and
  `webApps/lib/hill_compute.fft_rescale` (HILL's own, separate
  implementation) only ever call it with a *separable* (tensor-product)
  query grid, so the NUFFT is actually an exact direct DFT, not an
  approximation — `_nonuniform_grid_dft` in `lib/transforms.py` computes it
  with plain numpy, and an `_HAS_FINUFFT` dispatch (`importlib.util.find_spec`,
  no import needed just to check) picks it when finufft is unavailable. This
  is the general way to unblock a tab whose only remaining blocker is a
  finufft call with this kind of query grid; a call with an arbitrary
  (non-separable) query grid would still need finufft itself.
- **trackpy** — not pre-bundled in Shinylive's Pyodide image; `HI3D`'s
  `webApps/lib/hi3d_core.py` falls back to a pure numpy/scipy 2D Crocker–Grier
  peak detector (`_locate_2d`) and centre-of-mass refiner (`_refine_com_2d`)
  matching `trackpy.locate` / `trackpy.refine_com` when `trackpy` is absent.
- **cryosparc-tools / torch** — compiled extensions with no Pyodide/wasm
  wheel; every call site that needs them is lazy and reached only for a
  cryoSPARC-specific or Gaussian-to-mesh feature a tab's ordinary (example
  data) flow never touches. (`psutil` is in the same boat — no Pyodide
  wheel — but every call site already tolerates its absence; see
  `lib/system.py`'s `_psutil_or_none`.)
- threads or processes (`ProcessPoolExecutor`, `ThreadPoolExecutor`,
  `concurrent.futures`, `multiprocessing`, `threading.Thread.start()`) —
  Pyodide (outside a SharedArrayBuffer + COOP/COEP setup this build does not
  assume) is single-threaded; `threading.Thread.start()` raises
  `RuntimeError`, and `ProcessPoolExecutor`/`multiprocessing` cannot
  fork/spawn at all. Code that pools work should go through
  `helicon.pool_executor(ExecutorCls, ...)` (`lib/system.py`), which falls
  back to a synchronous in-process `SerialExecutor` when
  `has_real_threads()` is false, instead of constructing the executor
  directly -- or, where the worker count itself is computed up front (most
  call sites), clamp it to 1 with `has_real_threads()` so the executor
  construction is simply never reached. Multi-step loops yield to the asyncio
  event loop (`await asyncio.sleep(0)` in `denovo3d_tab.py`) or push immediate
  status updates (`helicon.shiny.push_output_ui` in `hill_tab.py`) so progress
  bars and Stop buttons remain responsive without worker threads.
- **32-bit WebAssembly (`wasm32`) & ephemeral filesystem** — under Pyodide,
  `np.intp` is 32-bit (`int32`), so `np.bincount` rejects explicit `int64`
  index arrays under NumPy's safe-casting rule; array indexing and `bincount`
  paths use `np.intp`. Also, `lib/cache.py` uses an in-memory no-op
  `DummyMemory` on `sys.platform == "emscripten"` instead of pickling to
  `joblib.Memory` on ephemeral in-RAM `MEMFS`.
- **offline `<head>` JS bundles (`plotly` / `bokeh`)** —
  `helicon.shiny.plotly_dependency()` and `helicon.shiny.bokeh_dependency()`
  (and `helical_lattice_tab._plotly_dependency()` in vendor mode) inject the
  Plotly and Bokeh JS bundles once into `<head>` via `htmltools.HTMLDependency`,
  so standalone exports work offline without CDN fetches and without repeating
  multi-MB inline scripts on every figure update.
- real subprocesses, or the server's own filesystem (the Home tab's
  launchers, a "server" file-picker mode) — already turned off wherever the
  app checks `deployment.is_cloud()`, which is now also true under Pyodide
  (`deployment.is_browser()`).

`numba`, `finufft`, `cryosparc-tools`, and `psutil` are optional extras of
`helicon` (`pyproject.toml`'s `accelerate` group bundles all four), not hard
dependencies — `pip install helicon` with no extras installs cleanly under
micropip, which is what lets `mode="package"` bundles (below) install the
real `helicon` package at all.

As of this writing, every `helicon webApps` tab is bundled and verified
(`tests/test_shinylive_build.py`): **HelicalLattice** (`mode="vendor"`: no
`helicon` import at all, pure numpy/pandas/plotly), **HelicalProjection**,
**HI3D**, **whereIsMyClass**, **HILL**, **HelicalPitch**, **abinitio3D**,
**denovo3D** (all `mode="package"`: the real `helicon` wheel), and **Home**
(`mode="package"`, pure shiny + stdlib; see "Home's links" below). Each tab's
own `Bundle(...)` entry in `build.py`'s `BUNDLES` documents the extra
requirements and the specific blockers that applied to it.

### Home is the landing page

Home is a singleton tab, not a Shiny module (`home_tab_ui()` takes no id;
`home_tab_server(input, session)` has no `output`) -- `_stage` picks
`_APP_PY_TEMPLATE_PLAIN` for it instead of the module-call template every
other bundle uses (`Bundle.is_module=False`).

`main()` exports Home like any other bundle, to `build/shinylive/home/`, then
`promote_to_root()` moves its files up into `build/shinylive/` itself --
Home's own page becomes `build/shinylive/index.html`, in place of
`write_index()`'s plain link list, so a visitor to the site's root lands on
the diagram, not a bullet list. (A build that excludes "home", e.g. `python
build.py helical_lattice`, still gets `write_index()`'s plain list -- there is
no Home exported to promote.)

Outside the browser, Home's diagram clicks an integrated app's own navbar
tab (the normal `helicon webApps` multi-tab page, including a non-browser
hosting-service deployment of it, is still one page with a shared navbar).
Under `deployment.is_browser()` there is no such navbar -- every tab is its
own standalone bundle -- so `home_tab._app_svg` links that app's chip to its
sibling bundle's own page instead (`HomeApp.bundle`, each tab's `BUNDLES`
key), the same way it already links to a hosted site like Map2seq. The link
is the bare `"<bundle>/"`, not `"../<bundle>/"`, matching Home's own page
always sitting at the site root with every sibling one level below it.

### URL input mode

`deployment.host_is_public()` validates a user-supplied URL's host with
`socket.getaddrinfo()` before letting the app fetch it (SSRF guard) — a
guard against a visitor using a shared server to reach that server's own
internal network (localhost, a private subnet, the cloud metadata address).
Pyodide has no raw sockets, so `socket.getaddrinfo()` fails shut there, which
would reject every URL, including legitimate public ones (e.g.
HelicalProjection's own default EMPIAR example URL). But in a browser bundle
there is no shared server to protect that way: the fetch runs in the
visitor's own browser, reaching only what that visitor's own browser could
already reach by typing the URL into its address bar — so
`deployment.url_allowed()` skips `host_is_public()` under
`deployment.is_browser()`, keeping only the scheme check (which already
excludes a local path).

Passing that check is not enough to actually fetch the URL: `requests`
(what `lib.path_utils._get()` and every `*_from_url`/`download_micrograph`
loader use underneath) makes a real socket connection, which Pyodide has
none of either. `path_utils._patch_requests_for_browser()` calls the
`pyodide-http` package's `patch_all()` under `deployment.is_browser()` (a
no-op, and `pyodide-http` not even imported, outside it), which monkeypatches
`requests`' transport to route through the browser's own
`XMLHttpRequest` instead. `pyodide-http` is added to every `mode="package"`
bundle's requirements in `build.py`'s `_stage()` alongside helicon's own base
dependencies, for the same reason those are listed explicitly (see
`wheel_base_requirements`'s docstring): it is needed by every bundle, since
`download_file_from_url` is reachable from the real `helicon` package they
all install, not just the one tab a given bundle wires up.

## Building

```
pip install shinylive   # build-time tool, not a helicon dependency
python tools/shinylive/build.py [bundle ...]   # default: all bundles
```

Output goes to `build/shinylive/<bundle>/` (gitignored); each is a
self-contained static site (`shinylive export` output) — serve it with any
static file server, e.g. `python -m http.server --directory
build/shinylive/helical_lattice 8000`. `build/shinylive/index.html` is the
promoted Home landing page when `home` is built (the default), or a plain link
list otherwise.

The first run downloads Shinylive's web assets (Pyodide + a prebuilt `shiny`
package, ~100 MB) from GitHub and caches them in `~/.cache/shinylive`
(platform-dependent; `shinylive assets info` prints the path); later runs and
rebuilds reuse that cache.

## Files on the visitor's computer ("local folder")

A browser-side copy has no server files, so every tab's "server" input source
(a path field with a Browse button) would have nothing to list. Instead it is
offered as **local folder**, and works as it does on a lab server:

- a **Choose local folder** button above the path field asks the browser for a
  folder (the File System Access API: Chrome and Edge only). The folder is
  mounted read-only at `/local/<name>` in Pyodide's file system; files are read
  from the disk when opened, not copied, and nothing is uploaded. It stays
  attached until the page is closed;
- **Browse...** opens the usual file dialog, confined to `/local`
  (`file_picker_server(..., root=...)`);
- the tab's loaders read the path they are given, as for any server file.

The pieces live in `webApps/local_folder.py` and in `lib/shiny_file_picker.py`
(`source_modes`, `file_picker_field`, `file_picker_fill`). A tab does nothing
for it except ask `deployment.has_server_files()` where it would ask whether
it is *not* on a hosting service (`not deployment.is_cloud()`) -- a browser-side
copy is cloud-like (`is_cloud()` is true) and has server files all the same.
Features that need a real machine (RELION, the file explorer, downloads of
unbounded size) keep asking `is_cloud()`.

## Testing

`tests/test_shinylive_build.py` builds every bundle, serves each from a
plain `http.server`, and drives it with Playwright exactly as a visitor
would -- checking that it renders and that its console has no Python
traceback (not "no console errors": every `mode="package"` bundle that pulls
in pandas also has Pyodide try, and fail, to fetch the optional
pyarrow/pyodide-unix-timezones packages, which is benign noise). It is the
test that would have caught the iframe/module-id surprises the first bundle
(`helical_lattice`) turned up, and the `cache.py` threading bug and missing
`jupyter_bokeh` requirement that HI3D and HILL turned up respectively.
Marked `browser` (uses the `page` fixture) and `slow`, so it is outside
`pytest tests -n auto -m "not slow and not browser"`; run it with `pytest
tests/test_shinylive_build.py -m slow` (needs `pip install shinylive` and
`playwright install chromium`).

## Publishing

`.github/workflows/shinylive.yml` exports every bundle on each push and pull
request (catching a build break before a user does), and on a push to `main`
publishes them to GitHub Pages. GitHub Pages itself needs a one-time repo
setting: **Settings → Pages → Build and deployment → Source: GitHub
Actions**.
