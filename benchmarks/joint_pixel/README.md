# Joint ring: segment metadata and class-average pixels

Prototype and synthetic benchmark of `helicon.webApps.lib.helical_pitch_pixel`,
which adds the pixels of the class averages to the class ring of AbInitio3D
(`helical_pitch_phase`). Not wired into the app yet.

## What is compared

| Name | What it is |
|---|---|
| `meta_cal` | the AbInitio3D estimate as shipped: metadata ring, then `calibrate_period` |
| `gated_meta` | the metadata's repeat, kept from taking a sub-multiple by the self-similarity of the averages; class phases from the metadata and pixels together (`estimate_period_gated`) |
| `adapt2` | joint scan; the metadata's sharp value if it agrees, else the joint estimate (`estimate_period_adaptive`) |

## Run

```sh
python run.py --out train.csv   --period 400 --classes 12 --structure-seed 0
python run.py --out heldout.csv --period 700 --classes 20 --structure-seed 1 --lengths-in-periods
```

80 conditions (filament length 0.25 to 4 repeats, 0 to 60% of the segments given a
random class, 30 or 200 filaments, 0 or 0.05 repeats of position noise) times 5
seeds. A different helix and different numbers of classes in the second run, which
was not used to choose any setting. Every truth is known. Job time: a few minutes
on 8 CPUs.

## What was found (5 seeds, 400 runs per method and data set)

Share of runs with the repeat within 5% of the truth:

| Filament length (repeats) | train: meta_cal | gated_meta | adapt2 | held-out: meta_cal | gated_meta | adapt2 |
|---|---|---|---|---|---|---|
| 0.25 | 0.02 | 0.04 | 0.84 | 0.01 | 0.04 | 0.12 |
| 0.5 | 0.30 | 0.40 | 0.64 | 0.05 | 0.35 | 0.25 |
| 1 | 0.98 | 0.99 | 0.99 | 0.89 | 1.00 | 0.79 |
| 2 | 0.98 | 1.00 | 1.00 | 0.95 | 1.00 | 0.98 |
| 4 | 0.99 | 1.00 | 1.00 | 0.96 | 1.00 | 1.00 |

RMS error of the class phases, as a fraction of the repeat (median, filaments of
at least 1 repeat), by share of wrong labels:

| Wrong labels | train: meta_cal | gated_meta | held-out: meta_cal | gated_meta |
|---|---|---|---|---|
| 0 | 0.0018 | 0.0011 | 0.0013 | 0.0049 |
| 0.2 | 0.0040 | 0.0019 | 0.0044 | 0.0051 |
| 0.4 | 0.0081 | 0.0029 | 0.0076 | 0.0069 |
| 0.6 | 0.0138 | 0.0036 | 0.0161 | 0.0085 |

Findings:

1. **Pairs of registered class averages are as good as pairs of segments for the
   class phases**, and do not depend on the labels. With 60% wrong labels the
   phase error falls 2 to 4 times. Where the labels are all right the pixels can be
   a little worse (held-out, 0.0049 against 0.0013 of the repeat, about 3 A), as
   its images were binned by 2.
2. **The repeat from the pixels alone is not precise and is biased**, by an amount
   that depends on the images (+2% in the first set, -11% in the second): the
   offsets reach only a fraction of a repeat. So the metadata decides the repeat
   wherever it can, and the pixels do not pull it.
3. **A pair of segments or of registered averages cannot tell the repeat from its
   half** when filaments are shorter than a repeat; what is consistent mod P is
   consistent mod P/2 (the pixel term has exactly the same blind spot). The
   *self-similarity* of a class average at a lag (does the average repeat at half
   the repeat?) can, when the lag is shorter than the image; used as a veto on the
   coarse scan it fixes the "half the repeat" failures of the metadata.
4. **Filaments shorter than a repeat stay unreliable** for every method. The
   joint fallback of `adapt2` gets the right basin but not a precise repeat (error
   2 to 7%), and was worse in the held-out set. The user should be warned
   (`support` low), not told a number.

## Real data: EMPIAR-10940 (tau filaments, 98,287 segments, RELION Class2D job010)

`real_10940.py` runs the AbInitio3D pipeline as the tab does (counterpart merge,
ring, filament directions, calibration), then the pixel-assisted estimates on the
same pairs. Truth, used only to score: the deposited twist, 1.21 deg per 4.75 A
rise, so a projection repeat of 706.6 A. Selection rules, fixed in advance: the 20
most populated classes (13 after merging turned copies) and all 42 non-blank
classes (26 after merging).

| | top 20 classes | all classes |
|---|---|---|
| Production repeat (metadata ring, calibrated) | 707.2 A (+0.08%) | 705.5 A (-0.16%) |
| `gated_meta`, `adaptive` | 707.2 A | 705.5 A |
| pixels in the fine scan (`joint_gated`) | 687.1 A (-2.8%) | 674.7 A (-4.5%) |

Stress, repeat within 5% of the deposited one (6 random draws each):

| Stress | meta_cal | gated_meta | adaptive |
|---|---|---|---|
| 20 filaments | 0.50 | 0.50 | 0.50 |
| 50 filaments | 0.83 | 0.83 | 0.50-0.67 |
| 150 filaments | 1.00 | 1.00 | 1.00 |
| 20 / 40 / 60% of segments in a random class | 1.00 | 1.00 | 1.00 / 1.00 / 0.67 (top 20), 0.00 (all) at 60% |
| filaments cut to 350 A (half a repeat) | 0.00 | 0.00 | 0.00 |
| filaments cut to 700 A and 1400 A | 1.00 | 1.00 | 1.00 |

1. **The repeat is not improved on this data set**, and not harmed by `gated_meta`
   (identical to the shipped estimate in every condition). The metadata already
   gives it to within 0.2%; the self-similarity veto never had a half repeat to
   reject. `adaptive` is worse than the shipped estimate in a few conditions (60%
   wrong labels, 350 A filaments), so it should not be used. The pixels used for
   the fine scan are biased by 3 to 8%, as in the synthetic runs.
2. **The pixel offsets agree with the metadata ring, and flag the same classes.**
   For the classes whose metadata fit is at least 0.5 the registered offsets lie a
   median 28 A (top 20) and 24 A (all) from where the ring puts them (3 to 4% of
   the repeat); for classes with a fit below 0.5 the median is 206 A and 112 A.
   Spearman correlation of fit with the residual -0.75 (top 20, p = 0.003) and
   -0.62 (all, p = 0.001); AUC for telling a poorly fitting class from a good one
   0.90 and 0.94. The two measurements are independent (segment positions and
   labels against image content). This is the level 2 diagnostic of the design.
3. **Some classes have a poor metadata fit but a pixel offset that agrees with the
   ring** (class 5 of the top 20: fit 0.23, residual 28 A; class 20 of all: fit 0.12,
   residual 24 A). Their averages look like proper views at the ring's phase while
   their segments' labels are inconsistent. These are the candidates for a
   segment-level repair (reading each segment's phase from its own pixels), which
   is not built.
4. **Class-phase accuracy on real data is not shown.** The synthetic gain from the
   pixel phases (2 to 4 times at 60% wrong labels) could not be checked here: the
   class phases were compared with the placement of each class on the deposited map
   (EMD-14046), but that placement agreed with the ring for only 6 of 8 well fitting
   classes within 50 A (two were 250 to 300 A away, near the half-repeat ambiguity),
   so it is not a clean truth. It needs a better reference (for example the
   per-class placement from a refined 3D reconstruction).
5. **Label degradation is optimistic for the pixels**: the class averages stay as
   they are. A real poor assignment also blurs the averages, which needs the
   segment stacks to reproduce.

## Real data, a mixture: EMPIAR-10230 (AD tau, RELION Class2D job684, 102,027 segments)

`real_mixed.py`. A data set of two helical types (PHF and SF) in one
classification, where the pooled ring is a blend. Same analysis as the tab; then
the repeat is re-estimated from the classes the pixels agree with and from those
they do not. 6.9 A/pixel averages.

| Selection | Classes (after merging) | Pooled repeat | Classes the pixels agree with | Classes they flag |
|---|---|---|---|---|
| 20 largest | 15 | 766.3 A | 13 classes: 766.4 A | 2 classes (too few for a ring) |
| all non-blank | 35 | 768.0 A | 24 classes: 768.1 A | 11 classes: 317.8 A, score 0.70, median fit 0.27 |

1. **The pooled repeat equals that of the classes the pixels agree with** (766 and
   768 A). In this classification one type dominates; the metadata found a single
   group of classes in both selections.
2. **The flagged classes are mostly poor ones, not a second polymorph.** Of the 11
   flagged (all classes), most are tiny (3 to 135 segments) with fits below 0.25,
   and re-estimating from them gives a spurious 318 A ring. Two large classes are
   flagged in both selections (class 1: 4,252 segments, fit 0.23 to 0.30, residual
   273 to 290 A; class 31: 4,295 segments, fit 0.51 to 0.54, residual 111 to 115 A).
   Whether they are another type cannot be told from these data (no labels), and
   they are too few to form a ring on their own.
3. **So this data set does not test the detection of a second polymorph.** The
   earlier notes (cryoSPARC classes of the same data) found 762 and 840 A for the
   two types; here the SF classes are not present in strength. The test needs a
   classification whose classes are known to contain both types.
4. **Agreement of the two measurements as a diagnostic** held again: classes with a
   metadata fit above 0.7 had pixel residuals of 6 to 24 A (all under 4% of the
   repeat).

## Two rings: filaments of two types that share their classes

`helical_pitch_mixture.py`, `mixture_synthetic.py`, `real_two_rings.py`,
`real_two_rings_validation.py`. When two helical types (tau PHF and SF) fall into the
same 2D classes, the classes cannot be split by type (`class_groups` finds one
group), and the pooled ring is a blend. What differs is, per filament, how the
class label advances along it. The fit gives each of two rings a repeat and a phase
per class, and each filament to the ring it follows better (either way round),
alternating until nothing changes.

**Synthetic mixtures** (repeat 400 A and 400 x ratio, classes shared, 200 filaments,
5 seeds, 6 random starts; both repeats within 3% of the truth):

| Filament length (repeats) | ratio 1.03-1.20, no wrong labels | 30% wrong labels | 60% wrong labels |
|---|---|---|---|
| 0.5 | never | never | rarely |
| 1 | always (assignments 75-100% right at 3%, 100% from 6%) | 40% at 3%, 80-100% from 6% | 0-40% |
| 2 | always | always | 80% |
| 4 | always | always | always |

A minor type of 20% is found as well as a 50/50 mixture. In data of one type (the
null) the fit gives two rings of the same repeat (separation 0.1% or less) and a gain
of 0.001, **except for short filaments (0.5 repeat) or many wrong labels with
filaments of 1 repeat**, where it invents a second ring with a gain of 0.02 to 0.13 and
separation up to 81%. So a gain alone does not show a second type.

**EMPIAR-10230** (AD tau, RELION job684; 35 classes after merging, 3,895
filaments, 6.9 A/pixel; PHF and SF):

| | one ring | two rings |
|---|---|---|
| Repeat | 767.1 A | **724.0 A and 881.7 A** (ratio 1.22) |
| Filaments | 3,895 | 1,799 and 2,096 |
| Mean agreement of the pairs with their ring | 0.743 | 0.802 (gain +0.059) |

* All nine starts (by each filament's own repeat, and eight at random) end at the same
  two rings (723.6 to 724.2 and 879.5 to 881.7 A).
* Split half (four random halves of the filaments, fitted separately): the first ring
  at 721.9 to 727.1 A, the second at 870.4 to 897.5 A; the filaments of one half are
  given the same ring by the other half's rings as by that half's own in 74 to 85%.
* Against a single perfect ring laid on the data's own segment positions and filament
  lengths (24 surrogates, noise 0.2): gain at most 0.0035, so the real 0.059 is
  larger than all 24 (p < 0.04, the most 24 replicates can show). The surrogates were
  noisier than the real data (one-ring objective 0.63 against 0.74); the noise grid
  did not go lower.
* The class averages follow the first ring better than the second (median class
  residual 39 A with 8 of 35 classes disagreeing, against 69 A and 16 of 35), as
  expected if the shared averages are dominated by one type.

**But the control also finds a second ring.** EMPIAR-10940 (one type) gives rings of
593 and 715 A, gain +0.017, also above its surrogates (p < 0.04). Its second ring is
unstable (587 to 643 A in eight half fits, against 721 to 727 for the main one),
holds short filaments (median span 287 A), and its class averages agree with it as
well as with the main ring (residuals 21 and 26 A). So "better than one perfect
ring" does not separate a second structure from a catch-all for filaments that fit
no ring well; EMPIAR-10230's second ring is more stable (870 to 898 A, spread 1%
against 3.5%), has more filaments (54%) and a gain 3.5 times larger. No ground-truth
type labels were used at this point.

**Against the manual labels: the rings are not PHF and SF.** The segments of the job
come from two extractions: `Extract/job670` (77,911 segments; 86% found by
coordinate in the manually labelled `PHF_particles.star`, 0.4% in the SF one) and
`Extract/job671` (24,116; 86% in `SF_particles.star`, 2% in the PHF one). The source
is the type (`real_10230_truth.py`; labels used only to score, and they are
"not quite perfect"). Of 3,895 filaments, 2,883 are PHF only, 984 SF only, 28 mixed.

| | repeat | score of its ring |
|---|---|---|
| PHF filaments fitted on their own (oracle) | 766.4 A | 0.819 |
| SF filaments fitted on their own (oracle) | 799.6 A | 0.608 |
| all filaments, one ring | 767.1 A | |
| unlabelled two-ring fit | 724.0 A and 881.7 A | |

* The unlabelled rings do not follow the types: the ring a filament is given to
  predicts its type for 51% of the filaments (59% for filaments over 1,600 A); always
  answering "PHF" gives 75%. The class phases of the rings do not match the oracle's.
* With the **true** types and their own rings, two rings fit the data only 0.0055
  better than one (mean agreement 0.7483 against 0.7428), while the unlabelled
  partition fits 0.059 better. The fit found a better-fitting partition that is not
  the types, so reproducibility (nine starts, split halves) and a gain over a
  single-ring surrogate are not evidence of a second type.
* The signal is not in these data: the two oracle repeats differ by 4%, the SF ring
  is weak, and the median filament is about one repeat long. Two rings whose repeats
  differ by a fraction d predict nearly the same labels over a short distance (they
  drift apart by d of the distance), so telling them apart needs filaments of many
  repeats: at 4% a filament of 800 A moves the two predictions only 32 A apart.
* The class averages do not help either: the true PHF ring agrees with them no better
  (median residual 51 A, 10 of 35 classes disagree) than the unlabelled 724 A ring
  (39 A, 8 of 35); the true SF ring (70 A, 11) no better than the 882 A ring (68 A, 16).
  The shared averages are blends.

**Sections instead of whole filaments** (`segment_scores`, `assign_segments`): each
segment is given a ring, or a junk state, along its filament by a Viterbi path that
charges for each change of state. A switch cost large against a filament's length gives
one state to the whole filament, so the two models are two ends of one setting. On
synthetic filaments of two known rings (400 and 440 A, no wrong labels) of which half
switch type midway, the switching filaments were 54% right as whole filaments and at
best 81% right as sections (window 300 A, cost 0.5), the others 100%: a local window
can only tell the rings apart over a stretch of a couple of repeats, so frequent
switching cannot be resolved however it is modelled.


## Types from the class averages and the classes each filament uses

`helical_pitch_types.py`, `real_10230_types.py`, `real_types_survey.py`,
`similarity_study.py`. The repeat cannot separate helical types (they can have nearly
the same one, and in EMPIAR-10230 the filament-level two-ring fit did not find them).
Their class averages differ in appearance, and filaments of one type use the same
classes.

**EMPIAR-10230, against the manual labels.** The classes are type-specific: of 38
classes with at least 100 segments, 70% of the segments are in classes that are
under 10% SF, 21% in classes that are 78 to 90% SF, 9% in mixed ones; calling each
class by its majority type labels 90.8% of the segments right (always PHF: 76.4%).

* The registered correlation of two class averages separates the types: PHF-PHF
  0.965, SF-SF 0.942, PHF-SF 0.773 (AUC 0.997 for same-type against different-type
  pairs). Spectral clustering with two groups failed (a few junk classes dominate the
  embedding); **average-linkage clustering of one minus the correlation** gives a
  24-class group that is 93% PHF (79,354 segments), a 7-class group that is 85% SF
  (21,170) and a 7-class junk group, from any cut of 0.10 to 0.20 (0.25 joins them),
  and a class-call accuracy of 0.908, the most any class-level rule can reach.
* A filament goes to the group most of its segments are in (left out below 60%):
  accuracy 0.91 with 97% of the filaments called (PHF 0.96, SF 0.77); 0.93 with 88%
  called at 80%; by filament span 0.83 (under 400 A), 0.92 (400 to 800), 0.97 (800
  to 1,600), 0.98 (over 1,600). Always PHF: 0.746. The class usage alone (filament
  co-usage of classes) reaches the same 0.91 (SF 0.82); the pixels alone were better
  at the class level and the usage at the filament level, and mixing them gave no
  more.
* The earlier "two rings" (724 and 882 A) were not these types; the groups'
  own rings are: PHF group 764.8 A (median class fit 0.75, 0 class averages
  disagreeing), SF group 794.2 A (fit 0.55), against the oracle's 766.4 and 799.6 A.

**Overlap, and what to correlate** (`similarity_study.py`; 28 classes of a clear type).
The correlation is normalised over the columns the two averages overlap in (at least
35% of the width). A short overlap raises the correlation of unrelated averages by
chance (different types 0.81 at an overlap of at most 40%, 0.73 at 90% or more; same
type 0.92 and 0.98) but the same-type pairs scored above all different-type pairs in
every overlap bin (AUC 1.0 per bin). The body of the filament correlates between any
two averages. Removing the mean along the axis first (`modulation=True`) widens the
gap (0.55 against 0.92, AUC 0.988), and a profile across the filament alone (no
overlap at all) reaches AUC 0.93; **but with the modulation the grouping was worse**:
the SF classes were not grouped on EMPIAR-10230, and two groups of 650 and 710 A on
EMPIAR-10944 were joined. The straighter averages vary little along the axis, and the
cross-section shape the plain correlation keeps tells the types apart. So the plain
correlation and a cut of 0.15 are the defaults. The largest gap in the tree
(`suggest_cut`) suggested 0.28 to 0.63 and joined PHF and SF on EMPIAR-10230 (0.41),
so it is not used.

**Without labels: does each group's own ring come out coherent?** Cut 0.15, plain
correlation, the tab's analysis on each group. The pooled ring of a mixture is broken
(for example median class fit 0.19 on EMPIAR-10943's mixed classification, 0.34 on
EMPIAR-10944).

| Data set | Pooled | Groups | Group rings |
|---|---|---|---|
| EMPIAR-10940 (one type) | 705.7 A, fit 0.81 | 1 | the same, 0 averages disagree |
| EMPIAR-10943, authors' type A (A1) | 499.7 A, fit 0.87 | 1 | the same |
| EMPIAR-10943, mixed (1,000 micrographs, 100 classes) | fit 0.19 | A-like 499.3 A (fit 0.86, 80k segments) and another 1,071 A (fit 0.30, incoherent) | the first matches A1 |
| EMPIAR-10944 (all particles) | 666 A, fit 0.34, 22 of 57 averages disagree | 3 | 650.0 A (fit 0.73), 710.0 A (0.70), 431.2 A (0.58): all coherent |
| EMPIAR-10944 (autopicked, 338k segments) | fit 0.08 to 0.11 | 10 | one coherent group of 21 classes (663 A, fit 0.58) and one of 3 (444 A, 0.64); the rest incoherent: noisy classes |

On the type-A subset and the single-type data sets the grouping returns one group, as
it should. Whether the groups of 10943 and 10944 are the types was not checked against
labels (the authors' later selections could provide them).

**What the filament neighbours of a class pair add** (`neighbour_study.py`;
EMPIAR-10230, 28 classes of a clear type, 231 same-type and 147 different-type
pairs, 1.54 million neighbouring segment pairs, pooled repeat 773 A). For classes a
and b, the segments of a and b near each other on one filament give how often they
are neighbours and the distances d = position(b) - position(a) between them.

* **Frequency.** Observed over expected neighbour pairs (from the class sizes): same
  type median 1.1 to 1.2, different type 0.07 to 0.11. AUC for same-type against
  different-type pairs 0.87 (window 300 A), 0.91 (800 A and 1,500 A); the pixel
  correlation alone 0.997; the two z-scored and added 0.999.
* **Distances as a starting offset.** The distances mod the repeat of a same-type pair
  are concentrated (R, the length of their mean unit vector: 0.84 against 0.57 for
  different types, AUC 0.88), and their circular mean predicts the axial offset the
  registration finds, **for the pairs registered without a flip: median |difference|
  9 A, 90% within 40 A (n = 108)**. For pairs registered with a 180 degree rotation
  or a mirror (n = 121) it does not (median 115 to 133 A, 17% within 40 A): the
  segments of a class and of its turned counterpart are of opposite polarity and the
  pairs of opposite polarity are not collected, so the relation is not there to be
  read. Overall within 20 A 45%, within 100 A 70%. The disagreements are not low
  correlations (median correlation 0.95 to 0.97 in every bin of disagreement).
* So the neighbour statistics are independent evidence of the same relation the
  pixels measure, and a good starting offset for the same-orientation pairs; for the
  turned pairs they need the pairs of opposite polarity, which `prepare_pairs`
  discards. Not tested: restricting the registration to the predicted offset (speed,
  fewer chance maxima), and adding the frequency to the affinity of the grouping.

**Neighbour frequency added to the grouping: not a clear gain**
(`types_lift_study.py`, `types_lift_survey.py`; affinity `0.7 * correlation + 0.3 *
min(lift, 1.5) / 1.5`, lift as above, window 800 A). On EMPIAR-10230 (labelled) the
grouping was right (class calls 0.908, filaments 0.91 to 0.92) for every cut from 0.20
to 0.40, where the correlation alone worked only from 0.10 to 0.20 and joined PHF
and SF from 0.25. But:

* a single-type data set was split: EMPIAR-10940 into 4 groups at cuts 0.15 to 0.20
  (3 at 0.25) and the type-A subset of EMPIAR-10943 into 3 groups (cuts 0.15 and 0.20),
  where the correlation alone gave one; the neighbour frequency of two classes of one
  type depends on how far apart they are on the repeat, so same-type pairs far apart on
  it look unrelated. They were one group again from cut 0.30 (0.25 for the subset).
* on EMPIAR-10944 (all particles) the groups with coherent rings that the correlation
  alone gave (650.0 A fit 0.73, 710.0 A fit 0.70, 431.2 A fit 0.58) became 677.5 A and
  659.9 A (fit 0.66 each, less distinct) and an incoherent group of 12 classes (1,016
  A, 11 of 12 class averages disagreeing); on EMPIAR-10943's mixed classification the
  type-A group lost 3 of its 17 classes (the same ring, 498.7 A).

So it is not in the Types card; `neighbour_lift` and `neighbour_affinity` stay as
documented options. A better use of the neighbours would take the distance into
account (compare the frequency at the distance the registered offset predicts), not
the frequency in a fixed window.

## What each segment's alignment (shift, score) and its run of classes say

`segment_info_study.py`, `shift_phase_study.py`. EMPIAR-10230, RELION job684 (102,027
segments, 3,895 filaments, spacing 28 A); "wrong type" = the segment's true type
differs from the majority type of its class (5.6% of the segments in clear-type
classes); "off the line" = a segment more than 90 degrees from its filament's fitted
phase line (15%, in the PHF group of 23 classes).

* **The numbers.** Shift along the filament: sd 11.5 A, kurtosis 4.5 (heavier tails
  than a normal), skew 0.12; max probability median 0.33; log-likelihood sd 468;
  none of them related to each other as a Gaussian profile would suggest: the
  correlation of max probability with |shift| is +0.01 and of the log-likelihood
  +0.03; the confidence is almost flat against the distance to the nearest end of the
  filament (0.36 within 50 A, 0.39 at 200 to 400 A).
* **The shift is already used, and helps.** The tab's positions are the box
  coordinate minus the origin ("refined path"). On the PHF group: with raw track
  positions the ring score is 0.759 and the median class fit 0.67; with the refined
  positions 0.847 and 0.75. Adding the shift once more as a fine phase on top of the
  class phase does not reduce the scatter around the filament's phase line (the
  regression slope is 0.00 degrees per A, against the 0.47 a shift that was the
  phase would give), with either kind of positions.
* **Run length.** The run of consecutive segments of one class has a pooled
  distribution of 42% at 1, 25% at 2, 15% at 3, 8% at 4 (median 3 for the segments, 1
  for the runs), not peaked at the sector of a class (about 1.2 segments). It is
  informative at both ends: runs of 1 are enriched in wrong-type segments (9.6%
  against 2.4% for runs of 10 or more), and runs of 10 or more in segments off the
  line (34.8% against 12 to 17%).
* **As detectors.** For the segments of the wrong type, AUC 0.60 for a short run, 0.58
  for low max probability, 0.57 for neighbours of another type, 0.47 for the
  log-likelihood and 0.48 for |shift|. For the segments off the line: 0.73 for
  |shift| (a shift that moves the position moves the segment off the line), 0.55 for
  low max probability, 0.50 for the log-likelihood, 0.44 for a short run (long runs
  are the suspect ones).
* **Per filament** (2,639 filaments with at least 10 segments; 279 with 20% or more
  wrong-type segments): AUC 0.69 for a low mean max probability, 0.62 for a long mean
  run, 0.57 for the log-likelihood of its run lengths under the pooled run-length
  distribution, 0.51 for the share of runs of 1 and for |shift|. The mean off-the-line
  residual of a filament correlates +0.34 with its mean run length and -0.25 with its
  mean confidence. These are modest, and independent of the correlation of the class
  averages, which already identifies the type.

**Run lengths in other data sets** (`run_length_survey.py`; per run, the share of
lengths 1 / 2 / 3 / 4 / 5; mean run; neighbouring segments overlap by 93 to 98%):

| Data set | Segment spacing | Classes | Runs of 1 / 2 / 3 / 4 / 5 | Mean run | Consecutive pairs in one class |
|---|---|---|---|---|---|
| EMPIAR-10230 (job684) | 28.2 A | 50 | 42 / 25 / 15 / 8 / 4% | 2.3 | 0.59 |
| EMPIAR-10940 (job010) | 14.2 A | 80 | 38 / 18 / 11 / 8 / 7% | 3.2 | 0.72 |
| EMPIAR-10943, mixed | 14.2 A | 100 | 37 / 18 / 15 / 12 / 7% | 3.0 | 0.70 |
| EMPIAR-10944 (all) | 14.2 A | 100 | 37 / 20 / 16 / 12 / 8% | 2.7 | 0.63 |
| EMPIAR-10943, type A (A1) | irregular (a selected subset) | 100 | 82 / 16 / 2 / 0 / 0% | 1.2 | 0.23 |

By segment, the median run is 5 segments on EMPIAR-10940 (13% of segments are in runs
over 10, 45% in runs of 6 or more) and 3 on EMPIAR-10230 (2% over 10). The spacing
sets the scale (the same classes cover about twice as many segments at 14 A as at 28
A); the A1 job was run on a selection from the full set, so its neighbours are
sparse. The run-length figures quoted for the labelled analysis are those of 10230.

**The detectors do not transfer between data sets** (`shift_phase_study.py` on
EMPIAR-10940, all classes: 3,700 filaments, repeat 705.5 A). Off the line (more than
90 degrees from the filament's fitted phase line): 12.5%. Here short runs are the
suspect ones, the reverse of EMPIAR-10230: off the line by run length 1: 23.6%, 2 to
3: 16.5%, 4 to 9: 11.7%, 10 or more: 3.5% (EMPIAR-10230: 14.8, 12.1, 17.0, 34.8%).
AUC for those segments: 0.66 for a short run (0.44 on EMPIAR-10230), 0.61 for low max
probability (0.55), 0.49 for |shift| (0.73), 0.39 for the log-likelihood (0.50). By
confidence quartile 17.1% (lowest) to 5.8% (highest) off the line (18.1% to 13.1%).
Per filament the mean off-the-line residual correlates -0.36 with the mean confidence
(-0.25) and +0.00 with the mean run length (+0.34). The shift is again not a phase
(slope -0.011 degrees per A, against the 0.51 expected). So run length and confidence
carry information about the assignment, but which way and how much depends on the
spacing and on how wide the classes are, and a score built on them needs to be
calibrated on each data set.

**Gaps left by a 2D class selection** (`gap_study.py`). The jobs used above have none:
the spacing is the same at every quantile (14.2 A on EMPIAR-10940, 28.2 A on
EMPIAR-10230). The type-A subset of EMPIAR-10943 does (spacings 14 to 157 A, median
43 A). To see what gaps do, whole classes of EMPIAR-10940 were removed at random
(five draws each) until a share of the segments was gone, and the run lengths
counted on what was left, naively (consecutive rows) and gap-aware (a filament cut
into chains where the spacing is over 1.5 times the nominal one, only complete runs
counted, neither end at a gap):

| Segments removed | Neighbours that still are | Naive: mean run, distance from the full data | Gap-aware: complete runs, mean, distance |
|---|---|---|---|
| 0% | 100% | 3.2 | 23,322, 2.98 |
| 12% | 97% | 3.34, 0.022 | 14,990, 3.02, 0.037 |
| 25% | 94% | 3.34, 0.033 | 10,204, 2.81, 0.042 |
| 34% | 92% | 3.38, 0.044 | 7,199, 2.75, 0.051 |
| 52% | 87% | 3.56, 0.077 | 2,479, 2.62, 0.086 |

(Total variation distance of the run-length distribution.) Removing classes at random
changes the distribution little even at half the segments, and it makes the naive runs
longer (a run of one class is joined across the gap left by a removed class). The
gap-aware count is no closer to the full data, because requiring a run to be complete
drops the longer ones; the right treatment is to count a run cut by a gap as censored
(at least that long), not to discard it. Not done.

## Gaps and missing views in the ring fit

`gap_ring_study.py`: EMPIAR-10940 (full data: 26 classes after merging, 98,287 segments,
repeat 705.5 A) with a share of the segments removed in four ways, 5 draws each at
shares of about 10, 20, 30 and 50%, the pitch estimate (ring, filament directions,
calibration) run on what is left.

| Removal | Failures (more than 2% off) | Median error of the others |
|---|---|---|
| segments at random (gaps, all classes present) | 0 of 20 | 0.09% |
| the worst-fitting classes first (junk) | 0 of 20 (and the median class fit rises from 0.74 to 0.81-0.82) | 0.07% |
| classes at random (views missing) | 3 of 20, all at half the repeat (49.8% off), from 38% removed | 0.32% |
| a contiguous arc of the ring | 5 of 20, at half the repeat from 51% removed | 0.42% |

Gaps alone do no harm; missing views do, by leaving the classes that remain to fit a
ring of half the size. The calibration bias stayed at -0.3 to -0.6% everywhere. The
tab's warning "2x fits almost as well" (double ratio at or above 0.8) fired in 5 of
the 8 failing runs and in none of the 72 good ones (0.7: 7 of 8 and still none; the
good runs peaked at 0.59); the "ring empty over 120 degrees" warning did not tell them
apart (12% of failing, 11% of good runs).

**The class averages rule the half repeat out** (`estimate_period(gate=...)`,
`analyze(class_images=...)`). The self-similarity of an average (its correlation with
itself shifted along the axis, the filament body removed) at half the repeat of
EMPIAR-10940 is -0.29 (at 353 A); a candidate repeat whose lag the averages cover and
at which they do not repeat gets a factor of 0. Of the 8 failures, the 3 with classes
removed at random were put right (about 1% off), the arc ones were not (49.7% became
45% off), and no good run got worse (median error 0.135% with and without it). It is
used by the tab's pitch estimate (and said in a warning when it changed the repeat).

**The class plot.** The ranked fits became a plot of the fit (y) against the
agreement of the registered class averages with the ring (x, 0 to 1:
`pixel_agreement`, the mean of cos(2 pi r / P) over the class's registered pairs, r
the difference between the offset and the ring's, clipped at 0). On EMPIAR-10940 the 16
classes with a fit of at least 0.5 have an agreement of 0.72 to 0.92 (median 0.87) and
the others a median of 0.21 (one, class 20, fit 0.12 and agreement 0.90: a class that
looks right with segments that do not fit); on EMPIAR-10230 0.54 to 0.87 (median 0.76)
against a median of 0.54.

## Caveats

* Synthetic only, one pixel size, one noise level, class averages that are windows
  of the long filament (the true model). Real class averages are averages over
  many filaments, centred where the classification put them (see
  `denovo3d_register`: a stitched composite of real averages put the twist at 1.30,
  not 1.20). The next test is on EMPIAR-10940 and EMPIAR-10230.
* A few constants (the 5% fallback window, the weight of the pixels in the
  phases, the self-similarity gate) were set on the first set and checked on the
  second. The second set was not used to choose them, but it shows they are not
  tuned to be optimal for both.
* The weights of the two terms are a convex combination of two normalised
  matrices; they could depend on the evidence of each.

The studies that save intermediate results (similarity matrices, tables) write them
to `joint_pixel_out/` in the current directory, or to the folder named by the
environment variable `JOINT_PIXEL_OUT`.
