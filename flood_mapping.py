#!/usr/bin/env python3
"""
Sentinel-1 SAR Flood Inundation Mapping — Permanent Standalone Web Map
=======================================================================
WHY THIS APPROACH
  geemap.addLayer() writes GEE tile-server URLs into the HTML.  Those URLs
  expire after ~2 days, causing layers to go blank.  This script avoids that
  by downloading every layer as a GeoTIFF, converting it to a base64-encoded
  PNG, and embedding the PNG data URI directly inside the folium HTML.
  The resulting file is fully self-contained and never expires.

METHOD
  Change-detection: VV backscatter difference (post - pre, dB).
  Flooded open water causes specular reflection away from the sensor,
  producing a sharp drop in VV backscatter.

DATA      Copernicus Sentinel-1 GRD IW (via Google Earth Engine)
AOI       Sagaing Region / Kachin State, upper Myanmar
Pre-flood  2026-06-11  (+/-DATE_WINDOW days)
Post-flood 2026-06-23  (+/-DATE_WINDOW days)
Output     flood_inundation_map.html  -- self-contained, never-expiring HTML

REQUIREMENTS
    pip install earthengine-api geemap folium rasterio numpy Pillow

FIRST-TIME GEE SETUP
    1. Sign up at https://earthengine.google.com
    2. Enable the Earth Engine API in a Google Cloud project
    3. Run:  earthengine authenticate
       (or let this script call ee.Authenticate() automatically)
"""

import os
import base64
import shutil
from io import BytesIO

from rasterio.merge import merge as rio_merge

import numpy as np
import ee
import geemap
import folium
from folium.plugins import MiniMap, Fullscreen, MousePosition
import rasterio
from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

AOI_COORDS = [
    [97.708969, 25.428393],
    [96.168823, 25.671236],
    [95.526123, 22.907803],
    [97.102661, 22.654572],
    [97.708969, 25.428393],
]

PRE_FLOOD_DATE  = "2026-06-11"
POST_FLOOD_DATE = "2026-06-23"
DATE_WINDOW     = 1       # +/- days to search for imagery around each target date

FLOOD_DB_THRESH = -3.0    # VV drop (dB) below which pixel is classified flooded
SPECKLE_RADIUS  = 30      # focal-mean kernel radius in metres

# Your GEE Cloud project ID -- leave "" to let GEE auto-detect from credentials
GEE_PROJECT = "gee-python-419405"

# Resolution for local download (metres/pixel).
# This AOI is moderate-large; 100 m avoids GEE's 48 MB single-request limit.
EXPORT_SCALE = 100

OUTPUT_HTML = "flood_inundation_map.html"
TEMP_DIR    = "_gee_tmp"      # deleted automatically after the map is built

# Max tile size in degrees for tiled GEE downloads.
# 1.5° × 1.5° tiles stay well under GEE's 48 MB per-request limit at 100 m.
TILE_DEG = 1.5

BBOX = {
    "west":  min(c[0] for c in AOI_COORDS),
    "east":  max(c[0] for c in AOI_COORDS),
    "south": min(c[1] for c in AOI_COORDS),
    "north": max(c[1] for c in AOI_COORDS),
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. GEE INITIALISATION
# ──────────────────────────────────────────────────────────────────────────────

def init_gee() -> None:
    kwargs = {"project": GEE_PROJECT} if GEE_PROJECT else {}
    try:
        ee.Initialize(**kwargs)
        print("[OK] GEE initialised.")
    except ee.EEException:
        print("[!] Not authenticated -- running ee.Authenticate() ...")
        ee.Authenticate()
        ee.Initialize(**kwargs)
        print("[OK] GEE initialised.")

# ──────────────────────────────────────────────────────────────────────────────
# 2. SENTINEL-1 RETRIEVAL
# ──────────────────────────────────────────────────────────────────────────────

def load_s1(aoi: ee.Geometry, center_date: str, window: int,
            orbit_pass: str = "DESCENDING") -> ee.Image:
    """Mean composite of Sentinel-1 IW GRD scenes within +/-window days."""
    d_start = ee.Date(center_date).advance(-window, "day")
    d_end   = ee.Date(center_date).advance( window, "day")

    col = (
        ee.ImageCollection("COPERNICUS/S1_GRD")
        .filterBounds(aoi)
        .filterDate(d_start, d_end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass))
        .select(["VV", "VH"])
    )
    n = col.size().getInfo()

    if n == 0:
        print(f"  No {orbit_pass} images -- trying both orbit passes ...")
        col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(aoi)
            .filterDate(d_start, d_end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .select(["VV", "VH"])
        )
        n = col.size().getInfo()

    if n == 0:
        raise RuntimeError(
            f"No Sentinel-1 data within +/-{window} days of {center_date}. "
            "Try widening DATE_WINDOW."
        )

    print(f"  [{center_date}] {n} scene(s) found -> mean composite.")
    return col.mean().clip(aoi)

# ──────────────────────────────────────────────────────────────────────────────
# 3. SPECKLE FILTERING
# ──────────────────────────────────────────────────────────────────────────────

def speckle_filter(image: ee.Image, radius: int = SPECKLE_RADIUS) -> ee.Image:
    """Boxcar (focal-mean) speckle suppression."""
    return image.focal_mean(radius=radius, kernelType="circle", units="meters")

# ──────────────────────────────────────────────────────────────────────────────
# 4. FLOOD DETECTION
# ──────────────────────────────────────────────────────────────────────────────

def detect_floods(pre: ee.Image, post: ee.Image,
                  aoi: ee.Geometry,
                  thresh: float = FLOOD_DB_THRESH):
    """
    Returns
    -------
    flood_mask  binary (1 = newly flooded)
    perm_water  binary (1 = JRC permanent/seasonal water)
    diff_db     continuous VV change image (dB)
    """
    diff_db = post.select("VV").subtract(pre.select("VV")).rename("VV_change")

    # JRC Global Surface Water v1.4 -- seasonality >= 4 months/year
    perm_water = (
        ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
        .select("seasonality").gte(4).clip(aoi)
    )

    # HydroSHEDS terrain slope -- exclude steep pixels (unlikely to flood)
    slope = ee.Terrain.slope(ee.Image("WWF/HydroSHEDS/03VFDEM")).clip(aoi)

    flood_mask = (
        diff_db.lt(thresh)          # large VV decrease
        .And(perm_water.Not())      # not already open water
        .And(slope.lt(5))           # flat terrain only
        .rename("flood")
    )
    return flood_mask, perm_water, diff_db

# ──────────────────────────────────────────────────────────────────────────────
# 5. FLOOD AREA STATISTICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_flood_area_km2(flood_mask: ee.Image, aoi: ee.Geometry) -> float:
    area = (
        flood_mask
        .multiply(ee.Image.pixelArea())
        .reduceRegion(
            reducer=ee.Reducer.sum(), geometry=aoi,
            scale=10, maxPixels=1e10,
        )
    )
    return ee.Number(area.get("flood")).divide(1e6).getInfo()

# ──────────────────────────────────────────────────────────────────────────────
# 6. DOWNLOAD GEE IMAGE -> LOCAL GEOTIFF
# ──────────────────────────────────────────────────────────────────────────────

def download_as_geotiff(image: ee.Image, name: str,
                        aoi: ee.Geometry, scale: int) -> str:
    """
    Download a GEE image to TEMP_DIR/<name> (EPSG:4326).

    GEE's getDownloadURL API rejects requests larger than 48 MB.
    For large AOIs this function tiles the BBOX into TILE_DEG x TILE_DEG
    degree cells, downloads each tile separately, then merges them into a
    single GeoTIFF with rasterio.merge -- no size limit, any resolution.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    final_path = os.path.join(TEMP_DIR, name)

    if os.path.exists(final_path):
        print(f"  Using cached {name}.")
        return final_path

    # Build tile grid over the full BBOX
    tile_defs, tile_paths = [], []
    lat = BBOX["south"]
    while lat < BBOX["north"]:
        lat_end = min(lat + TILE_DEG, BBOX["north"])
        lon = BBOX["west"]
        while lon < BBOX["east"]:
            lon_end = min(lon + TILE_DEG, BBOX["east"])
            tile_defs.append((lon, lat, lon_end, lat_end))
            lon = lon_end
        lat = lat_end

    n_tiles = len(tile_defs)
    print(f"  Downloading {name} in {n_tiles} tile(s) at {scale} m/px ...")

    for i, (w, s, e, n) in enumerate(tile_defs, 1):
        tile_region = ee.Geometry.Rectangle([w, s, e, n])
        tile_path   = os.path.join(TEMP_DIR, f"_tile_{i:03d}_{name}")
        tile_paths.append(tile_path)

        if not os.path.exists(tile_path):
            print(f"    Tile {i}/{n_tiles}  [{w:.3f},{s:.3f} -> {e:.3f},{n:.3f}] ...")
            geemap.ee_export_image(
                image.clip(tile_region),
                filename=tile_path,
                scale=scale,
                region=tile_region,
                crs="EPSG:4326",
                file_per_band=False,
            )
            if not os.path.exists(tile_path):
                raise RuntimeError(
                    f"Tile {i}/{n_tiles} download failed for {name}.\n"
                    f"Try reducing TILE_DEG (currently {TILE_DEG}) or "
                    f"increasing EXPORT_SCALE (currently {scale} m)."
                )

    if n_tiles == 1:
        shutil.move(tile_paths[0], final_path)
    else:
        print(f"  Merging {n_tiles} tiles -> {name} ...")
        datasets = [rasterio.open(p) for p in tile_paths]
        mosaic, transform = rio_merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update({
            "height":    mosaic.shape[1],
            "width":     mosaic.shape[2],
            "transform": transform,
        })
        for ds in datasets:
            ds.close()
        with rasterio.open(final_path, "w", **profile) as dst:
            dst.write(mosaic)
        for p in tile_paths:
            if os.path.exists(p):
                os.remove(p)

    print(f"  [OK] {name} saved.")
    return final_path

# ──────────────────────────────────────────────────────────────────────────────
# 7. GEOTIFF -> BASE64 PNG  (embedded data URI -- never expires)
# ──────────────────────────────────────────────────────────────────────────────

def geotiff_to_overlay(path: str, colorize) -> tuple:
    """
    Read a GeoTIFF, apply colorize(array)->RGBA, return:
      (data_uri_string, [[south, west], [north, east]])

    The data URI embeds the PNG bytes directly in the HTML string --
    no external server involved, no expiry date.
    """
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        b    = src.bounds           # assumes EPSG:4326 from download step

    rgba = colorize(data).astype(np.uint8)
    img  = Image.fromarray(rgba, "RGBA")
    buf  = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()

    folium_bounds = [[b.bottom, b.left], [b.top, b.right]]
    return f"data:image/png;base64,{b64}", folium_bounds


# ── Colour functions ──────────────────────────────────────────────────────────

def colorize_flood(data: np.ndarray) -> np.ndarray:
    """Flooded pixels -> semi-transparent red; background -> fully transparent."""
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[data == 1] = [215, 48, 39, 210]
    return rgba


def colorize_water(data: np.ndarray) -> np.ndarray:
    """Permanent water -> semi-transparent blue; background -> transparent."""
    rgba = np.zeros((*data.shape, 4), dtype=np.uint8)
    rgba[data == 1] = [33, 102, 172, 185]
    return rgba


def colorize_sar(data: np.ndarray,
                 vmin: float = -25.0, vmax: float = 0.0) -> np.ndarray:
    """SAR VV (dB) -> grayscale RGBA; nodata stays transparent."""
    rgba  = np.zeros((*data.shape, 4), dtype=np.uint8)
    valid = np.isfinite(data)
    gray  = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    g8    = (gray * 255).astype(np.uint8)
    rgba[valid, 0] = g8[valid]
    rgba[valid, 1] = g8[valid]
    rgba[valid, 2] = g8[valid]
    rgba[valid, 3] = 210
    return rgba

# ──────────────────────────────────────────────────────────────────────────────
# 8. BUILD PERMANENT STANDALONE FOLIUM MAP
# ──────────────────────────────────────────────────────────────────────────────

def build_map(overlays: dict, flood_km2: float, output: str) -> None:
    """
    Build a folium map where every raster layer is a base64 data URI
    embedded in the HTML.  No GEE tile links -- the file works forever.

    overlays keys: flood, water, pre_sar, post_sar
    Each value:    (data_uri_string, [[S, W], [N, E]] bounds)
    """
    cx = (BBOX["south"] + BBOX["north"]) / 2
    cy = (BBOX["west"]  + BBOX["east"])  / 2

    m = folium.Map(location=[cx, cy], zoom_start=10, tiles=None)

    # ── Basemaps ──────────────────────────────────────────────────────────────
    folium.TileLayer(
        tiles=("https://server.arcgisonline.com/ArcGIS/rest/services/"
               "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
        attr="Esri World Imagery",
        name="Satellite (Esri)",
        overlay=False, control=True,
    ).add_to(m)
    folium.TileLayer(
        "OpenStreetMap", name="OpenStreetMap",
        overlay=False, control=True,
    ).add_to(m)

    # ── Raster overlays (base64-embedded -- permanent) ────────────────────────
    # Each image is a PNG stored as a data URI inside the HTML file.
    # There are no URLs pointing to GEE servers, so nothing ever expires.
    pre_uri,   pre_bounds   = overlays["pre_sar"]
    post_uri,  post_bounds  = overlays["post_sar"]
    water_uri, water_bounds = overlays["water"]
    flood_uri, flood_bounds = overlays["flood"]

    folium.raster_layers.ImageOverlay(
        image=pre_uri, bounds=pre_bounds, opacity=0.85,
        name=f"Pre-flood VV SAR ({PRE_FLOOD_DATE})",
        overlay=True, control=True, show=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=post_uri, bounds=post_bounds, opacity=0.85,
        name=f"Post-flood VV SAR ({POST_FLOOD_DATE})",
        overlay=True, control=True, show=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=water_uri, bounds=water_bounds, opacity=1.0,
        name="Permanent Water (JRC)",
        overlay=True, control=True, show=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=flood_uri, bounds=flood_bounds, opacity=1.0,
        name="Flood Inundation",
        overlay=True, control=True, show=True,
    ).add_to(m)

    # ── AOI boundary (GeoJSON literal embedded in HTML) ───────────────────────
    folium.GeoJson(
        {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [AOI_COORDS]},
                "properties": {},
            }],
        },
        name="AOI Boundary",
        style_function=lambda _: {"color": "yellow", "weight": 2,
                                   "fillOpacity": 0},
        tooltip="Area of Interest",
    ).add_to(m)

    # ── Controls ──────────────────────────────────────────────────────────────
    folium.LayerControl(collapsed=False).add_to(m)
    Fullscreen(position="topright").add_to(m)
    MiniMap(toggle_display=True).add_to(m)
    MousePosition(position="bottomleft", separator=" | ",
                  prefix="Lat / Lon:").add_to(m)

    # ── Legend + info panel (fixed-position HTML, no external dependencies) ───
    legend_html = f"""
    <div style="
        position:fixed; bottom:36px; right:10px; z-index:1000;
        background:rgba(255,255,255,0.93);
        padding:14px 18px; border-radius:10px;
        border:1px solid #bbb;
        box-shadow:2px 2px 8px rgba(0,0,0,0.22);
        font-family:'Segoe UI',Arial,sans-serif;
        font-size:13px; line-height:1.85; min-width:230px;
    ">
      <b style="font-size:14px">Sentinel-1 Flood Map</b><br>
      <span style="color:#666;font-size:11px">Myanmar &mdash; Sagaing / Kachin</span>
      <hr style="margin:7px 0;border-color:#ddd">
      Pre-flood &nbsp;&nbsp;: <b>{PRE_FLOOD_DATE}</b><br>
      Post-flood : <b>{POST_FLOOD_DATE}</b>
      <hr style="margin:7px 0;border-color:#ddd">
      <span style="display:inline-block;width:13px;height:13px;
        background:#d73027;border-radius:2px;vertical-align:middle;
        margin-right:6px;opacity:0.85"></span>Flood Inundation<br>
      <span style="display:inline-block;width:13px;height:13px;
        background:#2166ac;border-radius:2px;vertical-align:middle;
        margin-right:6px;opacity:0.75"></span>Permanent Water (JRC)
      <hr style="margin:7px 0;border-color:#ddd">
      <b>Flood extent &approx; {flood_km2:.1f} km&sup2;</b><br>
      <span style="color:#999;font-size:11px">
        Threshold : VV &lt; {FLOOD_DB_THRESH} dB<br>
        Speckle filter : {SPECKLE_RADIUS} m radius<br>
        Permanent water excluded (JRC)
      </span>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save(output)
    print(f"[OK] Permanent standalone map saved -> {output}")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    init_gee()
    aoi = ee.Geometry.Polygon([AOI_COORDS])

    # ── GEE server-side processing ────────────────────────────────────────────
    print(f"\n> Loading S1 pre-flood  (+/-{DATE_WINDOW}d of {PRE_FLOOD_DATE}) ...")
    pre_raw  = load_s1(aoi, PRE_FLOOD_DATE,  DATE_WINDOW)

    print(f"> Loading S1 post-flood (+/-{DATE_WINDOW}d of {POST_FLOOD_DATE}) ...")
    post_raw = load_s1(aoi, POST_FLOOD_DATE, DATE_WINDOW)

    print("> Applying speckle filter ...")
    pre  = speckle_filter(pre_raw)
    post = speckle_filter(post_raw)

    print("> Detecting flooded pixels ...")
    flood_mask, perm_water, _ = detect_floods(pre, post, aoi)

    print("> Computing flood area ...")
    flood_km2 = compute_flood_area_km2(flood_mask, aoi)
    print(f"  => Estimated newly inundated area : {flood_km2:.2f} km2")

    # ── Download GeoTIFFs locally (EPSG:4326 for correct lat/lon bounds) ──────
    print("\n> Downloading rasters from GEE ...")
    flood_path = download_as_geotiff(flood_mask.toFloat(), "flood.tif",   aoi, EXPORT_SCALE)
    water_path = download_as_geotiff(perm_water.toFloat(), "water.tif",   aoi, EXPORT_SCALE)
    pre_path   = download_as_geotiff(pre.select("VV"),     "pre_vv.tif",  aoi, EXPORT_SCALE)
    post_path  = download_as_geotiff(post.select("VV"),    "post_vv.tif", aoi, EXPORT_SCALE)

    # ── Convert to base64 PNGs and embed in HTML ──────────────────────────────
    # data:image/png;base64,<bytes> URIs are stored inside the HTML -- no
    # external tile server, no token expiry, the map works indefinitely.
    print("> Encoding rasters as base64 PNGs ...")
    overlays = {
        "flood":    geotiff_to_overlay(flood_path, colorize_flood),
        "water":    geotiff_to_overlay(water_path, colorize_water),
        "pre_sar":  geotiff_to_overlay(pre_path,   colorize_sar),
        "post_sar": geotiff_to_overlay(post_path,  colorize_sar),
    }

    # ── Build and save the map ────────────────────────────────────────────────
    print("> Building permanent standalone HTML map ...")
    build_map(overlays, flood_km2, OUTPUT_HTML)

    # ── Clean up temp GeoTIFFs ────────────────────────────────────────────────
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    print("[OK] Temporary GeoTIFFs removed.")

    print(f"\n{'-' * 55}")
    print(f"  Done!  Open  '{OUTPUT_HTML}'  in any web browser.")
    print(f"  The map is self-contained and never expires.")
    print(f"{'-' * 55}")


if __name__ == "__main__":
    main()
