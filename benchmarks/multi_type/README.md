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
| `filament_alignment.py` | Filaments as sequences of class labels: an ungapped alignment of two filaments' class sequences (both directions, a substitution matrix from the class averages' correlation), scored against class usage and class bigrams |
| `filament_combined.py` | The alignment together with a repeat that each filament gets from its own class sequence |
| `two_stage.py` | Filaments grouped by the alignment, then split by pitch inside each group with the ring fit |
| `compare_stage1.py` | The Types card's first stage (class groups, then a vote by each filament's segments) against the alignment, on the same filaments |
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
6. **An ungapped alignment of the class sequences beats class usage for types that
   differ in their classes.** A filament is read as the sequence of its segments'
   classes. Two filaments of one type walk the same cyclic path begun at different
   points, so they match at equal distances along their lengths, in one direction or
   the other; a gap would stretch one to fit the other and hide a pitch difference, so
   the alignment has none. Against the cosine of class usage it gave AUC 0.78 against
   0.73 over all pairs, and 0.93 to 1.00 against 0.83 to 0.91 for the pairs of types
   whose classes differ. For the two types that share their classes it stayed at
   0.53 to 0.69 and did not improve with filament length: a pitch difference only makes
   the two walks drift apart, after about 150 A here, and a local alignment takes the
   best run over all shifts.
7. **A repeat from a filament's own class sequence is not recoverable.** The lag at
   which the sequence returns, after it has first decorrelated, exists for 3.6% of
   the filaments (median length 54 segments, about one repeat, and a third of the
   segments in a wrong class) and is inconsistent. Subtracting the difference of
   these repeats from the alignment score changed nothing.
8. **Alignment then ring fit does not beat the first stage alone.** Groups from the
   alignment (adjusted Rand index 0.31 with the reference sets) and then a two-ring
   split inside each group (0.29): the large group of types that share their repeat
   had no second repeat to find (the gain was 0.03, the two rings 2% apart), and the one
   group that split did so inside a single type.
9. **Against what the Types card does, the alignment is equal, not better.** The card
   groups the classes by their averages and gives each filament the group most of its
   segments are in: adjusted Rand index 0.29 to 0.33 for 3 to 8 groups, against 0.25
   to 0.31 for alignment clustering and 0.31 to 0.34 for the two mixed. The balanced
   accuracy for each pair of types differs by 0.01 to 0.14 in either direction. The
   alignment needs every pair of filaments (hundreds of millions for a data set of
   thousands), the vote one pass over the segments, so the card's first stage stays.
