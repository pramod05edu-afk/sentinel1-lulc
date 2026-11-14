"""15-class LULC scheme (must match training label order 0..14) + colors.

Label order = sorted kept IGBP ids [1,2,4,5,6,7,8,9,10,11,12,13,14,16,17].
Also defines coarse super-classes used to sanity-check against ESA WorldCover,
whose taxonomy differs from IGBP.
"""
import numpy as np

CLASS_NAMES = [
    "Evergreen Needleleaf Forest",   # 0
    "Evergreen Broadleaf Forest",    # 1
    "Deciduous Broadleaf Forest",    # 2
    "Mixed Forest",                  # 3
    "Closed Shrublands",             # 4
    "Open Shrublands",               # 5
    "Woody Savannas",                # 6
    "Savannas",                      # 7
    "Grasslands",                    # 8
    "Permanent Wetlands",            # 9
    "Croplands",                     # 10
    "Urban/Built-up",                # 11
    "Cropland/Natural Veg. Mosaic",  # 12
    "Barren",                        # 13
    "Water",                         # 14
]
N_CLASSES = len(CLASS_NAMES)

CLASS_COLORS = [
    "#006400", "#228B22", "#32CD32", "#6B8E23",   # forests
    "#BDB76B", "#D2B48C",                          # shrublands
    "#808000", "#BDC25B", "#9ACD32",               # savannas, grassland
    "#00CED1",                                     # wetlands
    "#FFD700",                                     # croplands
    "#E31A1C",                                     # urban
    "#FFA500",                                     # cropland/nat-veg mosaic
    "#DEB887",                                     # barren
    "#1F78B4",                                     # water
]

# ---- coarse super-classes for cross-taxonomy comparison ----
SUPER = ["Forest", "Shrubland", "Grassland", "Cropland",
         "Urban", "Barren", "Wetland", "Water"]

# our 15 labels -> super-class index
LABEL2SUPER = np.array([
    0, 0, 0, 0,   # forests
    1, 1,         # shrublands
    2, 2, 2,      # woody savanna, savanna, grassland -> Grassland
    6,            # wetlands
    3,            # croplands
    4,            # urban
    3,            # cropland/nat-veg -> Cropland
    5,            # barren
    7,            # water
], dtype=np.int64)

# ESA WorldCover v200 code -> super-class index (None = not mapped)
ESA2SUPER = {
    10: 0,   # Tree cover
    20: 1,   # Shrubland
    30: 2,   # Grassland
    40: 3,   # Cropland
    50: 4,   # Built-up
    60: 5,   # Bare / sparse vegetation
    70: None,  # Snow and ice
    80: 7,   # Permanent water bodies
    90: 6,   # Herbaceous wetland
    95: 6,   # Mangroves -> Wetland
    100: 2,  # Moss and lichen -> Grassland
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
