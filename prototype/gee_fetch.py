"""Google Earth Engine access: monthly Sentinel-1 composites for a location.

COPERNICUS/S1_GRD is provided by GEE already as sigma-nought backscatter in dB
(the same quantity SEN12MS stores), so no calibration/terrain-correction is
needed here - we only build a monthly composite and download a tile.

Monthly composite: dB -> linear -> mean -> dB (averaging in linear power is more
physically correct than averaging in dB).
"""
import io
import calendar
import requests
import numpy as np
import rasterio
import ee


def init_ee(project):
    """Authenticate (first run opens a browser) and initialize EE."""
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def region_around(lat, lon, size_px, scale=10):
    """Square region (ee.Geometry) of size_px*scale metres centred on point."""
    half = size_px * scale / 2.0
    return ee.Geometry.Point([lon, lat]).buffer(half).bounds()


def monthly_s1_db(region, year, month):
    """Mean VV/VH composite (dB) for one calendar month, or (0, None) if empty."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    col = (ee.ImageCollection("COPERNICUS/S1_GRD")
           .filterBounds(region)
           .filterDate(start, end)
           .filter(ee.Filter.eq("instrumentMode", "IW"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
           .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
           .select(["VV", "VH"]))
    n = col.size().getInfo()
    if n == 0:
        return 0, None
    lin = col.map(lambda im: ee.Image(10.0).pow(im.divide(10.0)))
    db = lin.mean().log10().multiply(10.0)
    return n, db


def download_db_tile(db_img, region, size_px):
    """Download a size_px x size_px, 2-band (VV,VH) dB array as float32."""
    url = db_img.getDownloadURL({
        "bands": ["VV", "VH"],
        "region": region,
        "dimensions": f"{size_px}x{size_px}",
        "format": "GEO_TIFF",
    })
    content = requests.get(url, timeout=180).content
    with rasterio.io.MemoryFile(content) as mf, mf.open() as ds:
        arr = ds.read().astype(np.float32)          # (2, H, W)
    # guard exact size
    arr = arr[:, :size_px, :size_px]
    if arr.shape[1] != size_px or arr.shape[2] != size_px:
        pad = ((0, 0),
               (0, size_px - arr.shape[1]),
               (0, size_px - arr.shape[2]))
        arr = np.pad(arr, pad, mode="edge")
    return arr


def fetch_year(lat, lon, start_year, start_month, size_px):
    """Return (months, stack) for 12 consecutive months from the start month.

    months : list of (year, month)
    stack  : float32 (12, 2, size_px, size_px) dB; months with no imagery are
             carried forward from the previous available month (NaN if none yet).
    """
    region = region_around(lat, lon, size_px)
    months, arrays = [], []
    last = None
    y, m = start_year, start_month
    for _ in range(12):
        n, db = monthly_s1_db(region, y, m)
        if db is None:
            arr = last if last is not None else np.full((2, size_px, size_px), np.nan, np.float32)
            status = "carried-forward" if last is not None else "MISSING"
        else:
            arr = download_db_tile(db, region, size_px)
            last = arr
            status = f"{n} scenes"
        print(f"  {y}-{m:02d}: {status}")
        months.append((y, m))
        arrays.append(arr)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months, np.stack(arrays).astype(np.float32)


def fetch_esa_worldcover(lat, lon, size_px):
    """ESA WorldCover v200 (2021) class codes for the same region, as uint8."""
    region = region_around(lat, lon, size_px)
    img = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    url = img.getDownloadURL({
        "region": region, "dimensions": f"{size_px}x{size_px}", "format": "GEO_TIFF",
    })
    content = requests.get(url, timeout=180).content
    with rasterio.io.MemoryFile(content) as mf, mf.open() as ds:
        arr = ds.read(1).astype(np.uint8)
    return arr[:size_px, :size_px]
