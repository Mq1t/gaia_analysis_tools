JD_offset = 2455197.5

# Used in input
SPTYPE_TEFF_RANGES = {
    "O": (30000, 60000),
    "B": (10000, 30000),
    "A": (7500, 10000),
    "F": (6000, 7500),
    "G": (5200, 6000),
    "K": (3700, 5200),
    "M": (2400, 3700),
}

DEFAULT_RELEASE = "dr3"

#Sample data used for generating the main sequence line in the HR Diagram.
# Coarse 8-point subsample of the full Pecaut & Mamajek dwarf sequence
# (spaced every ~10th spectral subtype). Verified against the full ~70-point
# table before applying: the interpolated curve barely shifts (well under
# 0.1 mag through most of the BP-RP range), since it's already a smooth,
# well-determined locus rather than something more sample points refine.
MAIN_SEQUENCE_TABLE = [
    # (BP-RP, M_G), spectral type
    (-0.620, -3.44),   # O9V
    (-0.330, 0.19),    # B8V
    (-0.020, 1.80),    # A7V
    (0.320, 2.85),     # F7V
    (0.880, 5.006),    # G7V
    (1.700, 7.57),     # K7V
    (2.780, 10.87),    # M3.5V
    (5.100, 15.90),    # M8.5V
]

# Approximate (BP-RP, M_G) anchor points for labelling broad stellar
# populations on plot_hr_diagram() when underlay=True. These are illustrative
# region labels only, not precise boundaries or a classification model, based
# on typical locations reported for Gaia (BP-RP, M_G) diagrams:
#   - White dwarfs: Gentile Fusillo et al. 2019 (arXiv:1904.02022) place the
#     DA/non-DA white dwarf sequence at roughly BP-RP 0.0-0.8, M_G +8 to +16.
#   - Giants: the red clump / red giant branch sits at roughly BP-RP 1.0-1.5,
#     M_G 0 to +1 in the Gaia DR2 HRD (Babusiaux et al. 2018, A&A 616, A10).
#   - Supergiants: very luminous evolved stars, roughly M_G -5 to -8, spanning
#     a wide colour range from blue to red supergiants.
# Label positions are clipped to the plot's actual xlim/ylim at draw time, so
# a label for a population that sits outside the current view (e.g.
# supergiants under the default ylim=[0, 20]) is pinned to the nearest edge
# rather than disappearing off-plot.
HRD_REGION_LABELS = [
    # (label, BP-RP, M_G)
    ("White Dwarfs", 0.2, 12.5),
    ("Giants", 1.3, 0.5),
    ("Supergiants", 0.5, -6.0),
]