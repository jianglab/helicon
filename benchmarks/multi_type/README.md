# Several helical types in one Class2D job

Studies of how well AbInitio3D's grouping of classes into types agrees with 3D
reconstructions that were solved by hand from the same Class2D job (several types,
each to better than 4 A). They need a RELION project whose Class2D job and refined 3D
sets are on disk; nothing here is specific to one computer, the paths are arguments.
Intermediate results (similarity matrices, the reference table) are written to
`types_out/` in the current directory, or to the folder named by `TYPES_OUT`.

| Script | What it does |
|---|---|
| `build_reference.py` | Finds every Class2D segment in each refined set (micrograph, tube and position along the tube) and writes which sets it ended in |
| `types_eval.py` | Scores the Types card's groups against the sets, class by class |
| `cut_study.py` | How filament consistency (the share of a filament's segments in its majority group) changes with the cut, on any Class2D job |
| `pair_rings.py`, `pair_mixture.py` | Two sets that share their classes: do their filaments differ in repeat, and does the two-ring fit separate them |
| `two_ring_test.py`, `two_ring_null.py` | The gain of two rings over one, on real groups and on synthetic ones of one type and of two |
| `segment_hmm.py` | A type along each filament (hidden state, switch rate, class probabilities by EM), anchored on the classes the pixels make clear |

## What the reference is

Each set went through class selection and then `images2star --recoverFullFilaments
minFraction=0.5` (in RELION's job records), which brings back every segment of a
filament in which half or more were in the selected classes. The labels are therefore
**filament-level**: a segment can carry its filament's type while sitting in a class of
the other type. Scoring by class majority is misleading; score per filament
(filaments that belong to exactly one set).

## Findings

1. **Average linkage cannot tell two thick types apart, whatever the cut.** For two
   such types, balanced accuracy of telling their filaments apart from the groups
   their segments fall in was 0.52 to 0.56 at every cut. Spectral clustering of the
   same pixel similarity (the baseline of the pairs removed) gave 0.82 to 0.87 for 3
   or 4 groups. A thin straight type separated from every other type with either
   method (0.97 to 1.0). The spectral code is not in the app: the types are easy to
   tell apart by eye, and the app asks for a number of groups (`pixel_groups_n`).
2. **The repeat separates types that share their classes.** Two types whose class
   averages are the same crossover views (twists 0.6 deg apart) are not separable by
   any grouping of classes; the per-filament repeats are (medians 500 and 417 A,
   AUC 0.81), and the two-ring fit finds two repeats (522 and 396 A) from every start,
   with 75% agreement with the reference (85% for the longest filaments).
3. **The gain of the second ring is not a threshold.** It is 0.19 for that pair and
   0.02 to 0.04 for groups of one type, but on synthetic data it depends on filament
   length (0.007 for 400 A filaments of two types 10% apart, 0.30 for 1600 A and 20%),
   so a fixed cut-off would miss short-filament mixtures. A held-out version
   (`helical_pitch_mixture.heldout_gain`) is the same on data sets of thousands of
   filaments, and guards small groups.
4. **Neighbour counts and filament usage do not help here.** The two thick types
   share filaments (the labels switch about 2.2 times along a mixed filament, against
   4.6 if shuffled), so classes of one type are used by the filaments of the other.
   Spectral clustering of the class-neighbour graph cuts the rings of classes into
   arcs (ARI about 0).
5. **The segment-wise model adds nothing over the pixel groups on these data** (87.6%
   for the pixel split, 88.1% with the model); unanchored it drifts to 76%.
