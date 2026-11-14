"""Build the classic-academic Word report (Times, navy headings)."""
import re
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

NAVY = RGBColor(0x1F, 0x38, 0x64)
GREY = RGBColor(0x44, 0x44, 0x44)
FONT = "Times New Roman"

doc = Document()

# base style
normal = doc.styles["Normal"]
normal.font.name = FONT
normal.font.size = Pt(11)
normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

# page = US Letter, 1in margins
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
for m in ("top_margin", "bottom_margin", "left_margin", "right_margin"):
    setattr(sec, m, Inches(1))


def _runs(p, text):
    """add text with **bold** / *italic* parsing."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not tok:
            continue
        r = p.add_run(tok[2:-2] if tok.startswith("**") else tok[1:-1] if tok.startswith("*") else tok)
        r.font.name = FONT
        if tok.startswith("**"):
            r.bold = True
        elif tok.startswith("*"):
            r.italic = True


def body(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.line_spacing = 1.15
    _runs(p, text)
    return p


def heading(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = FONT; r.font.size = Pt(15); r.bold = True; r.font.color.rgb = NAVY
    return p


def title_block():
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(18); p.paragraph_format.space_after = Pt(3)
    r = p.add_run("Sentinel-1-Only LULC Semantic Segmentation")
    r.font.name = FONT; r.font.size = Pt(20); r.bold = True; r.font.color.rgb = NAVY
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(2)
    r = p.add_run("Predictive Analytics and Deep Learning (Data Analytics 3) — SRH Hochschule Heidelberg")
    r.font.name = FONT; r.font.size = Pt(11); r.italic = True
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.space_after = Pt(12)
    r = p.add_run("Pramodkumar Shivanna · November 2026")
    r.font.name = FONT; r.font.size = Pt(11); r.font.color.rgb = GREY


def caption(text):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3); p.paragraph_format.space_after = Pt(10)
    r = p.add_run(text); r.font.name = FONT; r.font.size = Pt(9.5); r.italic = True; r.font.color.rgb = GREY


def figure(path, width_in):
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    doc.add_picture(path, width=Inches(width_in))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def _shade(cell, fill):
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:fill"), fill)
    cell._tc.get_or_add_tcPr().append(sh)


def _table_borders(tbl):
    tblPr = tbl._tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge in ("top", "bottom", "insideH"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6" if edge != "insideH" else "4")
        e.set(qn("w:space"), "0"); e.set(qn("w:color"), "AAAAAA" if edge != "insideH" else "DDDDDD")
        borders.append(e)
    for edge in ("left", "right", "insideV"):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "none"); borders.append(e)
    tblPr.append(borders)


def table(headers, rows, widths_in, aligns):
    t = doc.add_table(rows=1, cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    _table_borders(t)
    hdr = t.rows[0].cells
    for i, htext in enumerate(headers):
        hdr[i].width = Inches(widths_in[i])
        _shade(hdr[i], "EDF1F7")
        pr = hdr[i].paragraphs[0]
        pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT if aligns[i] == "r" else WD_ALIGN_PARAGRAPH.LEFT
        run = pr.add_run(htext); run.bold = True; run.font.name = FONT; run.font.size = Pt(10); run.font.color.rgb = NAVY
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].width = Inches(widths_in[i])
            pr = cells[i].paragraphs[0]
            pr.alignment = WD_ALIGN_PARAGRAPH.RIGHT if aligns[i] == "r" else WD_ALIGN_PARAGRAPH.LEFT
            run = pr.add_run(str(val)); run.font.name = FONT; run.font.size = Pt(10)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


# ================= content =================
title_block()

heading("1. Introduction")
body("This project trains a deep-learning model that labels every pixel of a satellite scene with its land-use / land-cover (LULC) class, using **Sentinel-1 radar as the only input**, and wraps that model in a prototype that maps how land cover at a chosen location changes across twelve months.")
body("Sentinel-1 carries a C-band synthetic-aperture radar (SAR) that measures surface backscatter in two polarizations (VV and VH). Unlike optical sensors, it sees through cloud and works day or night, which makes it attractive for consistent monthly monitoring. The cost is interpretability: a SAR scene encodes surface roughness, geometry and moisture rather than colour, so classes that are easy to separate in optical imagery (cropland versus grassland, or built-up versus forest) are far more ambiguous in backscatter. Working from SAR alone is the defining constraint and difficulty of this task.")
body("The work has two deliverables. The first is the segmentation **model**: data collection, a class scheme, a training pipeline and a quantitative evaluation. The second is the **prototype**: given a latitude/longitude and a start month, it retrieves twelve monthly Sentinel-1 composites, runs the model, and reports the land-cover map, the percentage of area in each class, and the month-to-month and total change, sanity-checked against an independent reference map.")

heading("2. Dataset")
body("The model is trained on **SEN12MS** (Schmitt et al., 2019), a global benchmark of georeferenced patch triplets of Sentinel-1 SAR, Sentinel-2 optical and MODIS land cover. Only the **Sentinel-1** patches (input) and the **MODIS land-cover** patches (labels) are used; the optical band is deliberately excluded to respect the SAR-only constraint. Each patch is a 256×256 GeoTIFF at 10 m ground sampling: Sentinel-1 holds two bands (VV, VH sigma-nought backscatter in dB), and the label is the MODIS IGBP classification.")
body("**Regions and time period.** All imagery was acquired in 2017. Two of the four seasonal collections were used, covering 102 globally distributed scenes: ROIs1158_spring (56 scenes) and ROIs2017_winter (46 scenes). Two seasons give the model exposure to seasonal backscatter variation, which matters because the prototype operates on a monthly time series.")
body("**Class scheme.** The IGBP legend has 17 classes. A scan of all 72,708 patch pairs showed two classes effectively absent (Deciduous Needleleaf Forest, 14 patches; Snow & Ice, 10 patches); both were dropped to an ignore label, leaving a **15-class** scheme. A richer scheme than the common 10-class simplification was kept deliberately, for a more detailed segmentation and a more informative per-class evaluation.")
body("**Class distribution.** SEN12MS is heavily imbalanced, so the training subset was sampled to flatten the distribution while preserving the natural rarity order. The resulting training-split pixel shares are:")
table(["Class", "Train pixel %"], [
    ["Evergreen Needleleaf Forest", "4.9"], ["Evergreen Broadleaf Forest", "6.1"], ["Deciduous Broadleaf Forest", "7.0"],
    ["Mixed Forest", "6.4"], ["Closed Shrublands", "0.8"], ["Open Shrublands", "6.7"], ["Woody Savannas", "8.7"],
    ["Savannas", "11.7"], ["Grasslands", "9.8"], ["Permanent Wetlands", "3.3"], ["Croplands", "8.1"],
    ["Urban / Built-up", "7.5"], ["Cropland / Natural Veg. Mosaic", "4.5"], ["Barren", "7.1"], ["Water", "7.5"],
], [4.0, 2.5], ["l", "r"])
body("**Data splits.** The split is by **scene**, not by patch, so no scene contributes to more than one split and adjacent patches cannot leak between training and test. From the 102 scenes, patches were drawn into 8,000 training / 1,500 validation / 2,000 test patches (11,500 total), class-balanced within each split. A consequence of honest scene-level splitting is that a few classes concentrated in a handful of scenes (notably Closed Shrublands, and Water in the test split) have low pixel support; their per-class metrics should be read with that in mind.")

heading("3. Preprocessing")
body("**SAR processing pipeline.** A full Sentinel-1 pipeline involves radiometric calibration, conversion to decibels, speckle filtering and terrain correction. SEN12MS ships the Sentinel-1 patches already calibrated, terrain-corrected and expressed as sigma-nought in dB, so these steps are performed upstream by the dataset providers. The same pipeline is reproduced in the prototype for live data, where Google Earth Engine delivers Sentinel-1 GRD already in dB.")
body("**Channel construction.** The two native bands (VV, VH) are expanded to three channels [VV, VH, VV−VH]. The difference channel adds cross-/co-polarization contrast, and three channels let the model load an **ImageNet-pretrained encoder** instead of training the first layer from scratch, which consistently improves convergence on small SAR datasets.")
body("**Normalization.** Each channel is standardized to zero mean and unit variance using statistics estimated from the training split. The exact per-channel mean and standard deviation are stored with the trained weights and used unchanged for test-set evaluation. On live Earth Engine data the prototype instead standardizes each scene by its own statistics, which is needed to bridge the SEN12MS-to-live domain shift (Section 6).")
body("**Patching and labels.** Imagery already arrives as 256×256 patches. Labels come from the co-registered MODIS IGBP layer of the same patch, so image and label are matched by construction; the 17 IGBP codes are remapped to the 15 training labels, the two absent classes set to an ignore value excluded from the loss and metrics.")
body("**Augmentation.** During training, patches are augmented with random horizontal/vertical flips and 90° rotations, which preserve the pixel-to-label correspondence and the backscatter statistics (unlike photometric augmentations, which would distort the physical dB values).")

heading("4. Model")
body("**Architecture.** The network is a **U-Net** with a **ResNet-34 encoder pretrained on ImageNet**, implemented with segmentation-models-pytorch. U-Net's encoder–decoder with skip connections is a strong default for dense pixel labelling and recovers fine spatial detail. A pretrained encoder was chosen because the training set is small; the low-level filters transfer usefully to SAR once the three-channel input is in place. A DeepLabV3+ variant with the same encoder was trained as a comparison (Section 5).")
body("**Loss.** Training minimizes a sum of **class-weighted cross-entropy and Dice loss**, with the ignore label excluded. Cross-entropy is weighted by inverse square-root class frequency to counter imbalance; the Dice term directly optimizes region overlap, which stabilizes learning on the rarer classes.")
body("**Optimization.** The model is trained with **AdamW** (learning rate 1.5×10⁻⁴, weight decay 10⁻⁴), a cosine-annealing schedule, batch size 16, **gradient clipping** (max-norm 1.0, which keeps training stable) and mixed-precision (AMP) on a single GPU. Training runs for up to 40 epochs with **early stopping on validation mIoU** (patience 10), and the best checkpoint by validation mIoU is kept. The checkpoint stores the architecture configuration and the channel normalization statistics so that inference is reproducible.")

heading("5. Experimental Results")
body("**Headline metrics (test set).** Overall accuracy (OA) = **41.2 %**, mean IoU (mIoU) = **0.250**. These are modest but expected: a 15-class scheme learned from Sentinel-1 alone, on a small subset with coarse MODIS labels, is a hard problem. The strongest classes are the ones physically distinct in backscatter — Evergreen Broadleaf Forest (IoU 0.72, F1 0.84) and Urban/Built-up (0.65 / 0.79) — while Closed Shrublands collapses to near zero for lack of test support.")
body("**Per-class results.** IoU and F1 are reported for every class, with test pixel support, since classes with little support give noisy scores.")
table(["Class", "IoU", "F1", "Support (% px)"], [
    ["Evergreen Needleleaf F.", "0.106", "0.192", "4.98"], ["Evergreen Broadleaf F.", "0.722", "0.838", "8.63"],
    ["Deciduous Broadleaf F.", "0.184", "0.311", "7.72"], ["Mixed Forest", "0.179", "0.304", "8.27"],
    ["Closed Shrublands", "0.000", "0.000", "0.22"], ["Open Shrublands", "0.229", "0.372", "7.34"],
    ["Woody Savannas", "0.250", "0.400", "9.89"], ["Savannas", "0.138", "0.243", "10.85"],
    ["Grasslands", "0.181", "0.307", "10.78"], ["Permanent Wetlands", "0.338", "0.506", "1.52"],
    ["Croplands", "0.306", "0.468", "9.78"], ["Urban / Built-up", "0.646", "0.785", "9.07"],
    ["Cropland / Nat. Veg.", "0.057", "0.108", "0.66"], ["Barren", "0.213", "0.351", "9.82"],
    ["Water", "0.200", "0.333", "0.48"],
], [3.2, 1.1, 1.1, 1.1], ["l", "r", "r", "r"])
body("**Training curves.** With gradient clipping and a lower peak learning rate, training is stable across all 40 epochs: the training loss descends smoothly with no instability spike, validation OA climbs to about 0.46 and validation mIoU to about 0.31 before early stopping at epoch 28.")
figure("v2_figs/curves.png", 6.2)
caption("Figure 1. Training/validation loss, validation OA and validation mIoU per epoch.")
body("**Confusion matrix.** The matrix shows strong diagonals for the classes physically distinct in backscatter — Evergreen Broadleaf Forest, Urban/Built-up, Permanent Wetlands and Water. The errors are systematic: the forest sub-types are confused with one another, and the cropland / grassland / savanna group confuses internally — the behaviour expected when only backscatter is available.")
figure("v2_figs/confusion.png", 4.5)
caption("Figure 2. Row-normalized confusion matrix over the 15 classes.")
body("**Qualitative predictions.** On test patches containing well-separated classes, the model produces coherent regions that track the reference rather than collapsing to a single label: water and permanent wetlands are delineated cleanly, and urban and forest blocks are recovered.")
figure("v2_figs/predictions.png", 4.3)
caption("Figure 3. VV input, ground truth and model prediction for four test patches.")
body("**Model-variant comparison.** Two architectures with the same ResNet-34 encoder were trained with the same stable recipe. The U-Net gives the higher OA and mIoU and is used for the prototype; the DeepLabV3+ result confirms U-Net was the better choice for this SAR segmentation task.")
table(["Variant", "OA", "mIoU"], [
    ["U-Net (ResNet-34)", "0.412", "0.250"], ["DeepLabV3+ (ResNet-34)", "0.387", "0.230"],
], [3.6, 1.45, 1.45], ["l", "r", "r"])
body("**Coarse-class (land-cover group) evaluation.** Re-scored at the level of eight coarse land-cover groups — which collapses the many within-group confusions of the 15-class scheme — overall accuracy rises from 41.2 % to **58.0 %** and mean IoU from 0.250 to **0.358**. This lines up almost exactly with the 58 % agreement against ESA WorldCover in the case study (Section 6): two independent evaluations agreeing that the model is reliable at the land-cover-group level.")
table(["Super-class", "IoU", "F1"], [
    ["Urban", "0.65", "0.79"], ["Forest", "0.58", "0.73"], ["Grassland", "0.35", "0.51"], ["Wetland", "0.34", "0.51"],
    ["Cropland", "0.33", "0.50"], ["Shrubland", "0.22", "0.36"], ["Barren", "0.21", "0.35"], ["Water", "0.20", "0.33"],
], [3.6, 1.45, 1.45], ["l", "r", "r"])

heading("6. Prototype")
body("The prototype turns the trained model into an end-to-end tool: it takes a **latitude/longitude and a start month** and produces twelve monthly land-cover maps with per-class area and change statistics.")
body("**System architecture.** The prototype is a small Python package with one module per stage, orchestrated by a command-line runner (run.py): gee_fetch.py (Earth Engine access, monthly compositing, tile download); model_infer.py (checkpoint loading, normalization, tiled inference); analyze.py (percentages and change tables); visualize.py (map grid and class-share chart); sanity_check.py (comparison against a reference map). A browser UI (app_gradio.py) and a demo notebook drive the same pipeline interactively.")
body("**Data retrieval and monthly compositing.** For each month, the prototype queries COPERNICUS/S1_GRD over a window around the point (IW mode, VV and VH). Compositing converts dB to linear power, averages over the month, and converts back to dB — more physically correct than averaging decibels. Months with no acquisition are carried forward and flagged.")
body("**Percentage and change calculations.** For each monthly map, the area share of every class is the pixel count divided by the total, in percent. The **month-to-month change** is the difference in each class's share between consecutive months; the **total change** is the difference between the first and last month — both in **percentage points**.")
body("**Case study: Munich (urban), 2023.** The prototype was run for central Munich (48.14 N, 11.58 E) over a 5.1 km window for the twelve months of 2023, using the stable v2 U-Net. Because a live composite does not match the SEN12MS statistics, the model is applied with **per-scene standardization** — each scene normalized by its own channel mean and standard deviation — which prevents the out-of-distribution collapse that the stored training statistics otherwise cause. The result is an urban-dominated series: Urban/Built-up covers roughly 70–80 % of the window, with the vegetated fringes split between cropland, grassland and wetland.")
figure("munich_v2/maps.png", 6.2)
caption("Figure 4. Twelve monthly LULC maps for Munich, 2023 (v2 model).")
body("The month-to-month variation is pronounced: the first-to-last change is **+8.5 pp for urban**, with croplands (−5.7 pp) and grasslands (−2.9 pp) shrinking over the year. Much of this is seasonal backscatter change rather than genuine land-use change (Section 7), which the change percentages cannot separate.")
figure("munich_v2/share.png", 6.2)
caption("Figure 5. Class share across the twelve months.")
body("**Sanity check against ESA WorldCover.** The first month was compared at the super-class level against ESA WorldCover (2021), not used in training. Overall per-pixel agreement is **58 %**. Agreement is strong on Urban (72.8 % vs 69.4 %), the class Sentinel-1 identifies best, and breaks down on vegetation structure: the model maps no Forest where WorldCover has 26.5 % tree cover, folding it into Cropland and Grassland — the SAR vegetation confusion made visible against independent ground truth.")
table(["Super-class", "Model %", "WorldCover %", "Diff (pp)"], [
    ["Forest", "0.0", "26.5", "−26.5"], ["Cropland", "18.2", "0.0", "+18.2"], ["Grassland", "8.7", "2.2", "+6.5"],
    ["Urban", "72.8", "69.4", "+3.4"], ["Water", "0.0", "1.6", "−1.6"], ["Wetland", "0.3", "0.0", "+0.3"],
    ["Barren", "0.0", "0.3", "−0.3"], ["Shrubland", "0.0", "0.0", "0.0"],
], [3.0, 1.4, 1.6, 1.2], ["l", "r", "r", "r"])
figure("munich_v2/cmp.png", 6.2)
caption("Figure 6. Model vs ESA WorldCover, reduced to super-classes (month 1).")

heading("7. Critical Discussion")
body("**Generalization to regions unlike the training data.** The model is trained on two seasons of 2017 SEN12MS scenes. Landscapes, crop calendars and acquisition geometries under-represented in that sample — tropical, arid or mountainous regions — will be segmented less reliably, and the prototype can be run anywhere. Reported accuracy is an in-distribution estimate and should not be assumed to hold globally.")
body("**SAR-specific class confusion.** Several classes are intrinsically ambiguous in C-band backscatter. Urban and forest can both appear bright through double-bounce and volume scattering; cropland, grassland and savanna overlap because their roughness is similar; smooth bare soil, calm water and tarmac can all appear dark. These confusions are a property of the sensor, not only of the model.")
body("**Seasonal effects that mimic land-use change.** Backscatter depends on soil moisture, vegetation water content and surface roughness, all of which vary through the year. A ploughed-then-grown field, a flooded-then-dry wetland, or snow cover can shift a pixel's class across months without any real change in land use. The prototype's month-to-month differences therefore mix genuine change with seasonal phenology, and should be read as apparent change.")
body("**Label noise.** The training labels are MODIS-derived at roughly 500 m and resampled to 10 m, so boundaries are coarse and small features mislabelled. This caps the achievable accuracy and biases the model toward the dominant class in each neighbourhood.")
body("**How classification errors inflate the change percentages.** The change statistics are differences of per-class pixel counts, so every misclassified pixel contributes. Per-pixel errors that are random in space do not cancel in a difference the way they would in a single-month average: if a class is over-predicted one month and under-predicted the next, the change absorbs the sum of both errors. Rare classes are the most affected. The reported changes should be read as upper bounds on real change, most trustworthy for the well-separated, high-support classes (water and urban).")

heading("8. Conclusion")
body("This project delivered a complete Sentinel-1-only LULC pipeline: a 15-class semantic-segmentation model — a stable U-Net that outperformed a DeepLabV3+ baseline — trained on a scene-disjoint, class-balanced subset of SEN12MS, and a prototype that maps twelve monthly land-cover composites for any location and reports per-class area and change, sanity-checked against an independent product. On the test set the model reaches **41.2 % overall accuracy and 0.250 mIoU** over 15 classes, rising to **58 % accuracy at the land-cover-group level** — corroborated by the 58 % agreement against ESA WorldCover in the Munich case study.")
body("The main limitations are inherent to the sensor and the labels: SAR class ambiguity, seasonal backscatter that mimics change, coarse MODIS labels, and the amplification of per-pixel errors in the change statistics. Future work would add more seasons and regions for robustness, use a higher-quality label source such as ESA WorldCover for training, and add temporal smoothing so the monthly series separates genuine land-use change from phenology.")

heading("9. References")
refs = [
    "Schmitt, M., Hughes, L. H., Qiu, C., Zhu, X. X. (2019). SEN12MS — A curated dataset of georeferenced multi-spectral Sentinel-1/2 imagery for deep learning and data fusion. ISPRS Annals IV-2/W7, 153–160.",
    "Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.",
    "Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation (DeepLabV3+). ECCV.",
    "He, K., Zhang, X., Ren, S., Sun, J. (2016). Deep Residual Learning for Image Recognition (ResNet). CVPR.",
    "Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. European Space Agency.",
    "European Space Agency. Sentinel-1 SAR User Guide / Level-1 GRD products. Copernicus Programme.",
    "Iakubovskii, P. (2019). Segmentation Models PyTorch. GitHub software library.",
]
for i, r in enumerate(refs, 1):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.3)
    p.paragraph_format.first_line_indent = Inches(-0.3)
    p.paragraph_format.space_after = Pt(4)
    run = p.add_run(f"[{i}] {r}"); run.font.name = FONT; run.font.size = Pt(10.5)

doc.save("report_academic.docx")
print("wrote report_academic.docx")
