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
| `filament_context.py` | The type of each segment from the classes along its filament: a hidden Markov chain over the positions of a filament, with a type that stays and a class that follows the type's own transitions (`helical_pitch_context`, a port of RELION's `helix_filament_context`), against the Types card's marker splits and its first stage |

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
10. **Filament context, a hidden Markov chain with a sticky type above the class, reaches the
    purity of the marker splits with fewer groups, once the picking directions are dealt
    with.** On every filament of the reference sets (about 17,000, of five sets; 1.65 million
    segments, 199 classes), with the probability of each segment's class from the
    classification: purity (share of filaments in the majority set of their group) and mean
    balanced accuracy of telling two sets apart (worst pair in brackets). Marker splits: 0.689
    and 0.818 (0.521) at 3 to 5 groups, 0.757 and 0.851 (0.723) at 6, 0.770 and 0.872 (0.731)
    at 8. Context, with the classes that are 180-degree counterparts merged and filaments
    picked backwards read the other way (the views follow the molecule, not the picking
    direction): 0.781 and 0.871 (0.720) at 4 types, 0.763 and 0.862 (0.716) at 5, 0.755 and
    0.869 (0.713) at 6, 0.781 and 0.861 (0.715) at 8; adjusted Rand index 0.535 at 4 types
    against 0.43 to 0.48 for the markers at 3 to 8 groups. **Without that handling the
    context model was worse** (0.719 and 0.864 (0.684) at 5 types, 0.741 at 8): a filament
    picked backwards runs its class sequence the other way and uses the counterpart classes,
    and a sequence model reads that as a second type. On the default data of the app (a
    nearly pure set) two of the "types" were largely the two directions (47% of the filaments
    of one had most of their segments in backward classes, 6% of the other). The first stage
    (class groups, then a vote) was lower than both (0.68 purity at any number of groups).
    Tempering the log-likelihoods (the segments overlap) did not help, and the held-out
    log-likelihood is nearly flat across the number of types, so it chooses the fewest (the
    tolerance of RELION, 0.01 nats per segment) and the number is best set by the slider. A
    fit of the model on all segments took 2 to 3 minutes on 8 CPUs; learning it from 400,000
    segments in whole filaments and decoding all of them took 25 seconds, with the same
    purity. AbInitio3D offers it as the "Filament context" method of the Types card (2 to 8
    types, starting at the number chosen by the held-out likelihood), the marker splits
    staying the default: it is seconds against minutes, and was measured on one set of
    several types.

    The port was checked against the C++ program it follows (RELION's `helix_filament_context`,
    built from source) on simulated filaments, with 20% of the segments missing, the input
    shuffled and the tempering and the largest gap changed: the log-likelihood agrees to
    0.5 nats of 15,000 (the two programs start from different random numbers; with no
    missing segments, to the printed digits), as do the types of the segments (all but the 0.3% on a
    position already taken, which RELION leaves without a type), the context probabilities
    (mean difference 1e-5 with no such segments), the transition table (1e-5), the sections
    (numbered alike when the input is in order along the filaments), the periods and their
    scores, and the held-out log-likelihoods by number of
    types. The check found one error, since corrected: of several segments on one grid
    position, the one that counts is the first along the filament, not the first in the
    input.
11. **A type of the filament context is not a structure: it can be a part of the cycle of
    views, and no rule from the class averages told them apart.** On the nearly pure
    default data of the app (98,000 segments, 3,700 filaments) the held-out log-likelihood
    rises with the number of types (+0.007 nats per segment at 2 types, +0.025 at 3,
    +0.039 at 7 over one type, every fold and shuffle the same sign), so the gain is
    reproducible, and the types learnt from two halves of the filaments agree on 91% of
    the segments. But the median filament spans 271 A and 90% span less than 700 A, the
    period of the class sequence, so a filament shows only part of the cycle of views, and
    the classes it uses depend on where in the cycle it lies. The types also differ in the
    mean class probability of their filaments (0.35, 0.49, 0.52) and not in the
    micrographs they are on (0.516 against 0.497 by chance). The app therefore says when
    the median filament shows less than half of the classes. A rule that merged types whose
    class averages look alike (the average linkage of `pixel_groups`, between the types)
    gave one group on that data, as it should if it is one structure, **but also one group
    on the multi-type data above**: the types of different sets had a mean correlation of
    0.83 to 0.91 between their classes, those of one set 0.88 to 0.92. It is not used; the
    number is shown for each group, and the held-out choice may now be one type.
