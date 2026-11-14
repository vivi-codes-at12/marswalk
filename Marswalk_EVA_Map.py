#!/usr/bin/env python3
"""
Marswalk EVA map — Jezero Crater prototype.

Converted from the Colab notebook ``Marswalk_EVA_Map.ipynb``. Each notebook
cell (or group of cells) is now a named *step*; steps are run in notebook
order by default, or individually with ``--steps``.

Project folder layout (same as the notebook's Google Drive folder)::

    Marswalk/
        Documentation/
        Notebooks/
        Outputs/
        Raw data/        <- M20_JezeroCrater_CTXDEM_20m.tif, HiRISE .IMG, MEDA .csv
        Processed data/

Usage examples::

    python Marswalk_EVA_Map.py --list-steps
    python Marswalk_EVA_Map.py --project-root ~/Marswalk
    python Marswalk_EVA_Map.py --steps crop,slope,static-demo
    python Marswalk_EVA_Map.py --steps plan-route --start-row 20 --start-col 20 \\
        --slope-cutoff 8 --layer Slope
    python Marswalk_EVA_Map.py --steps gradio --share

Required packages (previously pip-installed inside the notebook; install them
yourself before running)::

    pip install numpy scipy pandas matplotlib pillow rasterio requests folium \\
        geopandas fiona shapely pyproj pymupdf gradio

Only the packages needed by the steps you run are imported (folium, geopandas,
fiona, shapely, pyproj, pymupdf and gradio are imported lazily).
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LightSource, Normalize
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window
from scipy.ndimage import convolve, minimum_filter
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

CTX_DEM_NAME = "M20_JezeroCrater_CTXDEM_20m.tif"
HIRISE_DTM_NAME = "DTEEC_023524_1985_023379_1985_U01.IMG"
MEDA_FILES = {
    "PS": "WE__0001___________DER_PS__________________P02.csv",
    "TIRS": "WE__0001___________DER_TIRS________________P04.csv",
    "RHS": "WE__0001___________DER_RHS_________________P03.csv",
}

USGS_URL = "https://pubs.usgs.gov/sim/3464/sim3464_database.zip"
NASA_IMAGE_URL = "https://images-assets.nasa.gov/image/PIA25917/PIA25917~medium.jpg"
NASA_METADATA_URL = "https://images-assets.nasa.gov/image/PIA25917/metadata.json"
MEDA_SIS_URL = (
    "https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/"
    "mars2020_meda/document/meda_env_rdr_sis.pdf"
)

# Overview rasters are read at no more than this many pixels along the long side.
MAX_OVERVIEW_DIMENSION = 1200

# 8-connected neighbourhood used by the route planner.
NEIGHBOURS = (
    (-1, 0), (1, 0), (0, -1), (0, 1),
    (-1, -1), (-1, 1), (1, -1), (1, 1),
)

# Source notes: these describe intended use, not proof that data has been downloaded.
SOURCE_REGISTRY = [
    {
        "source": "JPL Mars Trek",
        "intended_use": "Mars map and mission-data discovery; record the specific products used.",
        "caveat": "The portal is a source/discovery tool; cite each actual dataset or map product in the project.",
    },
    {
        "source": "USGS-NASA Planetary Geologic Mapping Program",
        "intended_use": "Geologic context for the selected Mars location.",
        "caveat": "Record the map title, scale, publication, and source when a specific map is selected.",
    },
    {
        "source": "Curiosity proposed-route animation",
        "intended_use": "Visual reference for route-map presentation.",
        "caveat": "Curiosity/Mount Sharp example; do not present it as route data for Jezero.",
    },
    {
        "source": "NASA Worldview",
        "intended_use": "Possible interface inspiration.",
        "caveat": "Do not use as a Mars satellite-data layer unless a specific Mars product is verified.",
    },
    {
        "source": "Mars Perseverance Raw Images Gallery",
        "intended_use": "Surface photos and visual context near the selected exploration area.",
        "caveat": "Keep available captions, sol/camera information, and image links with each selected image.",
    },
]


# --------------------------------------------------------------------------
# Project context and shared helpers
# --------------------------------------------------------------------------

@dataclass
class Context:
    """Project paths, command-line options, and state shared between steps."""

    root: Path
    args: argparse.Namespace
    state: dict = field(default_factory=dict)  # e.g. the Folium map shared by map steps

    # Project folders
    @property
    def documentation(self) -> Path:
        return self.root / "Documentation"

    @property
    def notebooks(self) -> Path:
        return self.root / "Notebooks"

    @property
    def outputs(self) -> Path:
        return self.root / "Outputs"

    @property
    def raw(self) -> Path:
        return self.root / "Raw data"

    @property
    def processed(self) -> Path:
        return self.root / "Processed data"

    # Frequently used files
    @property
    def ctx_dem(self) -> Path:
        return self.raw / CTX_DEM_NAME

    @property
    def crop_dem(self) -> Path:
        return self.processed / "jezero_test_corridor_dem.tif"

    @property
    def crop_slope(self) -> Path:
        return self.processed / "jezero_test_corridor_slope.tif"

    @property
    def geology_raster(self) -> Path:
        return self.processed / "jezero_geology_units.tif"

    @property
    def geology_legend(self) -> Path:
        return self.outputs / "jezero_geology_legend.csv"

    @property
    def gdb(self) -> Path:
        return (
            self.raw / "USGS_SIM3464" / "sim3464_JezeroNili_FINAL"
            / "Geodatabase" / "Nili.gdb"
        )

    @property
    def hirise_dtm(self) -> Path:
        return self.raw / HIRISE_DTM_NAME

    @property
    def hirise_aligned(self) -> Path:
        return self.outputs / "HiRISE_Jezero_DTM_in_CTX_CRS_5m.tif"

    @property
    def final_map_html(self) -> Path:
        return self.outputs / "marswalk_jezero_final.html"

    @property
    def nasa_image(self) -> Path:
        return self.raw / "PIA25917_hollowed_rock.jpg"

    @property
    def meda_merged(self) -> Path:
        return self.outputs / "MEDA_sol1_merged_by_SCLK.csv"

    @property
    def meda_verified_png(self) -> Path:
        return self.outputs / "MEDA_sol1_presentation_verified_units.png"

    def meda_path(self, key: str) -> Path:
        return self.raw / MEDA_FILES[key]


def finish_figure(ctx: Context, fig) -> None:
    """Replacement for the notebook's inline ``plt.show()``.

    Figures are always saved by the caller. They are only displayed when the
    script is run with ``--show``; otherwise they are closed to free memory.
    """
    if ctx.args.show:
        plt.show()
    plt.close(fig)


def raster_extent(transform, width: int, height: int):
    """(left, right, bottom, top) for ``imshow`` of an unrotated north-up raster."""
    return (
        transform.c,
        transform.c + width * transform.a,
        transform.f + height * transform.e,
        transform.f,
    )


def valid_mask(*pairs) -> np.ndarray:
    """Finite-and-not-nodata mask for one or more ``(array, nodata)`` pairs."""
    mask = None
    for array, nodata in pairs:
        this = np.isfinite(array)
        if nodata is not None:
            this &= array != nodata
        mask = this if mask is None else (mask & this)
    return mask


def read_overview(src, max_dimension: int = MAX_OVERVIEW_DIMENSION, *, masked=True,
                  resampling=Resampling.average):
    """Read band 1 of ``src`` downsampled to at most ``max_dimension`` pixels."""
    scale = min(1.0, max_dimension / max(src.height, src.width))
    height = max(1, round(src.height * scale))
    width = max(1, round(src.width * scale))
    data = src.read(1, out_shape=(height, width), masked=masked, resampling=resampling)
    return data, height, width


def unit_name(row, fallback: str) -> str:
    """Geologic unit label: symbol, then unit, then name."""
    for key in ("symbol", "unit", "name"):
        value = row.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return fallback


# ---- map helpers (Folium, local pixel coordinates) -------------------------

def new_pixel_map(center, height: int):
    import folium

    return folium.Map(
        location=center,
        crs="Simple",
        tiles=None,
        zoom_start=0,
        min_zoom=-2,
        max_zoom=4,
        width="100%",
        height=height,
        control_scale=False,
    )


def add_notice(m, html_block: str) -> None:
    import folium

    folium.Element(html_block).add_to(m.get_root().html)


# ---- route planning (shared by static demo, planner, Leaflet map, Gradio) ---

class RouteError(RuntimeError):
    """Raised when no route can be found or reconstructed."""


@dataclass
class DemoRasters:
    """The 150 x 150 test-crop DEM and its provisional slope raster."""

    dem: np.ndarray
    slope: np.ndarray
    valid: np.ndarray
    transform: object
    crs: str
    pixel_size: float

    @property
    def rows(self) -> int:
        return self.dem.shape[0]

    @property
    def cols(self) -> int:
        return self.dem.shape[1]

    @property
    def extent(self):
        return raster_extent(self.transform, self.cols, self.rows)


def load_demo_rasters(ctx: Context) -> DemoRasters:
    for path in (ctx.crop_dem, ctx.crop_slope):
        if not path.is_file():
            raise FileNotFoundError(
                f"{path} not found. Run the 'crop' and 'slope' steps first."
            )

    with rasterio.open(ctx.crop_dem) as src:
        dem = src.read(1).astype(float)
        transform = src.transform
        dem_nodata = src.nodata
        crs = str(src.crs)
        pixel_size = abs(src.transform.a)

    with rasterio.open(ctx.crop_slope) as src:
        slope = src.read(1).astype(float)
        slope_nodata = src.nodata

    valid = valid_mask((dem, dem_nodata), (slope, slope_nodata))
    if not valid.any():
        raise ValueError("No valid pixels found in the test rasters.")
    return DemoRasters(dem, slope, valid, transform, crs, pixel_size)


def snap_to(mask: np.ndarray, row, col):
    """Nearest True pixel in ``mask`` to (row, col)."""
    points = np.argwhere(mask)
    d2 = (points[:, 0] - int(row)) ** 2 + (points[:, 1] - int(col)) ** 2
    return tuple(points[np.argmin(d2)])


def find_route(allowed: np.ndarray, slope: np.ndarray, start, goal, pixel_size: float,
               *, no_route_msg: str = "No route found.",
               broken_msg: str = "Could not reconstruct the route."):
    """Dijkstra over an 8-connected grid; returns (path_rows, path_cols).

    Illustrative cost: distance x (1 + mean slope / 20 deg). This is not a
    validated route-planning or safety model.
    """
    rows, cols = allowed.shape
    ids = np.arange(rows * cols).reshape(rows, cols)
    graph = lil_matrix((rows * cols, rows * cols), dtype=np.float32)

    for dr, dc in NEIGHBOURS:
        r0, r1 = max(0, -dr), min(rows, rows - dr)
        c0, c1 = max(0, -dc), min(cols, cols - dc)
        rr, cc = np.mgrid[r0:r1, c0:c1]
        nr, nc = rr + dr, cc + dc
        ok = allowed[rr, cc] & allowed[nr, nc]

        step = pixel_size * (np.sqrt(2) if dr and dc else 1.0)
        cost = step * (1 + (slope[rr, cc] + slope[nr, nc]) / 40.0)
        graph[ids[rr[ok], cc[ok]], ids[nr[ok], nc[ok]]] = cost[ok]

    start_id, goal_id = ids[start], ids[goal]
    distances, predecessors = dijkstra(
        graph.tocsr(), directed=True, indices=start_id, return_predecessors=True
    )
    if not np.isfinite(distances[goal_id]):
        raise RouteError(no_route_msg)

    path = [goal_id]
    while path[-1] != start_id:
        previous = predecessors[path[-1]]
        if previous < 0:
            raise RouteError(broken_msg)
        path.append(previous)
    path.reverse()
    return np.unravel_index(path, (rows, cols))


# --------------------------------------------------------------------------
# Steps
# --------------------------------------------------------------------------

# ---- Cell 0: project folders -------------------------------------------------

def step_setup(ctx: Context) -> None:
    """Check (or with --create-dirs, create) the project folder structure."""
    expected_dirs = {
        "Documentation": ctx.documentation,
        "Notebooks": ctx.notebooks,
        "Outputs": ctx.outputs,
        "Raw data": ctx.raw,
        "Processed data": ctx.processed,
    }

    if ctx.args.create_dirs:
        for path in expected_dirs.values():
            path.mkdir(parents=True, exist_ok=True)

    if not ctx.root.exists():
        raise FileNotFoundError(
            f"Could not find {ctx.root}. Pass --project-root, set MARSWALK_ROOT, "
            "or use --create-dirs to create the folder structure."
        )

    missing = [name for name, path in expected_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "These expected project folders were not found: " + ", ".join(missing)
            + " (use --create-dirs to create them)"
        )

    print(f"Project folder: {ctx.root}")
    print("Expected folders found:")
    for name, path in expected_dirs.items():
        print(f"  ✓ {name}: {path}")


# ---- Cell 1: source registry + input inventory -------------------------------

def step_registry(ctx: Context) -> None:
    """Write the source registry JSON and an inventory CSV of the data folders."""
    ctx.documentation.mkdir(parents=True, exist_ok=True)
    ctx.outputs.mkdir(parents=True, exist_ok=True)

    registry_path = ctx.documentation / "source_registry.json"
    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(SOURCE_REGISTRY, f, indent=2, ensure_ascii=False)

    inventory = []
    for data_dir in (ctx.raw, ctx.processed):
        for path in sorted(data_dir.rglob("*")):
            if path.is_file():
                stat = path.stat()
                inventory.append({
                    "folder": data_dir.name,
                    "relative_path": str(path.relative_to(ctx.root)),
                    "size_bytes": stat.st_size,
                    "modified_utc": datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                })

    inventory_path = ctx.outputs / "input_inventory.csv"
    with inventory_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["folder", "relative_path", "size_bytes", "modified_utc"]
        )
        writer.writeheader()
        writer.writerows(inventory)

    print(f"Saved source notes: {registry_path}")
    print(f"Saved file inventory: {inventory_path}")
    print(f"Files found: {len(inventory)}")

    if inventory:
        for item in inventory:
            print(f"- [{item['folder']}] {item['relative_path']} ({item['size_bytes']:,} bytes)")
    else:
        print("No files found yet in Raw data or Processed data.")


# ---- Cell 2: inspect the CTX DEM ----------------------------------------------

def step_dem_inspect(ctx: Context) -> None:
    """Read DEM metadata, save it as JSON, and save an overview preview."""
    dem_path = ctx.ctx_dem
    if not dem_path.is_file():
        raise FileNotFoundError(f"DEM not found: {dem_path}")

    metadata_path = ctx.outputs / "dem_metadata.json"
    preview_path = ctx.outputs / "dem_preview.png"

    with rasterio.open(dem_path) as src:
        band = 1

        # Read a smaller overview for inspection and plotting.
        preview_height = min(src.height, 1200)
        preview_width = min(src.width, 1200)
        preview = src.read(
            band,
            out_shape=(preview_height, preview_width),
            resampling=Resampling.average,
            masked=True,
        )

        valid = preview.compressed()
        if valid.size == 0:
            raise ValueError("The preview contains no valid pixels; check the raster's nodata metadata.")

        band_tags = src.tags(band)
        metadata = {
            "file": str(dem_path),
            "driver": src.driver,
            "width_pixels": src.width,
            "height_pixels": src.height,
            "band_count": src.count,
            "data_type": src.dtypes[band - 1],
            "crs": str(src.crs) if src.crs else None,
            "bounds_in_raster_crs": {
                "left": src.bounds.left,
                "bottom": src.bounds.bottom,
                "right": src.bounds.right,
                "top": src.bounds.top,
            },
            "pixel_size_in_raster_units": list(src.res),
            "nodata": src.nodata,
            "scale": src.scales[band - 1],
            "offset": src.offsets[band - 1],
            "dataset_tags": src.tags(),
            "band_tags": band_tags,
            "preview_valid_pixel_count": int(valid.size),
            "preview_min": float(valid.min()),
            "preview_max": float(valid.max()),
            "preview_mean": float(valid.mean()),
        }

        # Save the preview with axes in the raster's native coordinate system.
        fig, ax = plt.subplots(figsize=(9, 7))
        image = ax.imshow(
            preview,
            extent=(src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top),
            origin="upper",
            cmap="terrain",
        )
        ax.set_title("Jezero CTX DEM — overview preview")
        ax.set_xlabel("Raster X coordinate")
        ax.set_ylabel("Raster Y coordinate")
        fig.colorbar(image, ax=ax, label="Raster pixel value (units to verify)")
        fig.tight_layout()
        fig.savefig(preview_path, dpi=160)
        finish_figure(ctx, fig)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Saved metadata: {metadata_path}")
    print(f"Saved preview:  {preview_path}")
    print("\nRaster summary:")
    for key in (
        "width_pixels", "height_pixels", "crs", "bounds_in_raster_crs",
        "pixel_size_in_raster_units", "nodata", "scale", "offset",
        "preview_min", "preview_max",
    ):
        print(f"{key}: {metadata[key]}")
    print("\nBand tags:", band_tags or "(none)")


# ---- Cell 3: 3 km test crop ----------------------------------------------------

def step_crop(ctx: Context) -> None:
    """Cut a centered 150 x 150 pixel (3 km) test crop out of the CTX DEM."""
    dem_path = ctx.ctx_dem
    crop_path = ctx.crop_dem
    preview_path = ctx.outputs / "jezero_test_corridor_dem.png"
    crop_path.parent.mkdir(parents=True, exist_ok=True)

    crop_pixels = 150  # 150 pixels × 20 m/pixel = 3 km

    with rasterio.open(dem_path) as src:
        row = (src.height - crop_pixels) // 2
        col = (src.width - crop_pixels) // 2
        window = Window(col, row, crop_pixels, crop_pixels)

        data = src.read(1, window=window)
        transform = src.window_transform(window)

        profile = src.profile.copy()
        profile.update(
            height=crop_pixels,
            width=crop_pixels,
            transform=transform,
            compress="deflate",
        )

        with rasterio.open(crop_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(
                source_file=dem_path.name,
                note="Center test crop; not a selected science site.",
            )

        nodata = src.nodata
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata

    left, right, bottom, top = raster_extent(transform, crop_pixels, crop_pixels)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        np.ma.array(data, mask=~valid),
        extent=(left, right, bottom, top),
        origin="upper",
        cmap="terrain",
    )
    ax.set_title("Jezero DEM — 3 km test crop")
    ax.set_xlabel("Easting in DEM coordinate system (m)")
    ax.set_ylabel("Northing in DEM coordinate system (m)")
    fig.colorbar(image, ax=ax, label="DEM value (units to verify)")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=160)
    finish_figure(ctx, fig)

    print(f"Saved crop:    {crop_path}")
    print(f"Saved preview: {preview_path}")
    print(f"Crop bounds:   left={left:.0f}, bottom={bottom:.0f}, right={right:.0f}, top={top:.0f}")


# ---- Cell 4: provisional slope of the crop ---------------------------------------

def step_slope(ctx: Context) -> None:
    """Horn 3x3 slope of the test crop, saved as a GeoTIFF + preview PNG."""
    dem_path = ctx.crop_dem
    slope_path = ctx.crop_slope
    preview_path = ctx.outputs / "jezero_test_corridor_slope.png"

    with rasterio.open(dem_path) as src:
        elevation = src.read(1).astype(np.float32)
        nodata = src.nodata
        dx = abs(src.transform.a)
        dy = abs(src.transform.e)
        # (In the notebook these came from the crop cell's globals.)
        crop_transform = src.transform
        crop_height, crop_width = src.height, src.width
        valid = np.isfinite(elevation)
        if nodata is not None:
            valid &= elevation != nodata

        # Apply the raster's declared scale and offset.
        elevation = elevation * src.scales[0] + src.offsets[0]

        # Horn 3x3 terrain-gradient kernels.
        kernel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1],
        ], dtype=np.float32) / (8 * dx)

        kernel_y = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1],
        ], dtype=np.float32) / (8 * dy)

        # Ignore invalid pixels while calculating, then mask any 3x3
        # neighborhood that touches nodata or the raster edge.
        filled = np.where(valid, elevation, 0)
        dz_dx = convolve(filled, kernel_x, mode="constant", cval=0)
        dz_dy = convolve(filled, kernel_y, mode="constant", cval=0)
        slope_degrees = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

        complete_neighborhood = minimum_filter(
            valid.astype(np.uint8), size=3, mode="constant", cval=0
        ).astype(bool)
        slope_valid = complete_neighborhood & np.isfinite(slope_degrees)

        slope_nodata = -9999.0
        slope_output = np.where(slope_valid, slope_degrees, slope_nodata).astype(np.float32)

        # Don't carry over source tiling/block settings into this small output.
        profile = src.profile.copy()
        for key in ("blockxsize", "blockysize", "tiled", "predictor"):
            profile.pop(key, None)
        profile.update(
            dtype="float32",
            count=1,
            nodata=slope_nodata,
            compress="deflate",
        )

        with rasterio.open(slope_path, "w", **profile) as dst:
            dst.write(slope_output, 1)
            dst.set_band_description(1, "Provisional slope in degrees")
            dst.update_tags(
                method="Horn 3x3 gradient",
                horizontal_units="metres, from raster CRS",
                vertical_units="unverified; confirm before interpreting slope degrees",
                source_file=dem_path.name,
            )

    left, right, bottom, top = raster_extent(crop_transform, crop_width, crop_height)

    values = slope_output[slope_valid]
    if values.size == 0:
        raise ValueError("No valid slope pixels were calculated.")

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(
        np.ma.array(slope_output, mask=~slope_valid),
        extent=(left, right, bottom, top),
        origin="upper",
        cmap="magma",
        vmin=0,
        vmax=np.percentile(values, 98),
    )
    ax.set_title("Jezero test crop — provisional slope")
    ax.set_xlabel("Easting in DEM coordinate system (m)")
    ax.set_ylabel("Northing in DEM coordinate system (m)")
    fig.colorbar(image, ax=ax, label="Slope (degrees; vertical units unverified)")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=160)
    finish_figure(ctx, fig)

    print(f"Saved slope raster: {slope_path}")
    print(f"Saved slope preview: {preview_path}")
    print(
        "Provisional slope percentiles (degrees):",
        np.percentile(values, [0, 50, 90, 98, 100]).round(2).tolist(),
    )
    print("Reminder: verify the DEM's vertical units before using slope values as mission constraints.")


# ---- Cell 5: static demo map -------------------------------------------------------

def step_static_demo(ctx: Context) -> None:
    """Static two-panel demo map (elevation + slope) with an illustrative route."""
    r = load_demo_rasters(ctx)
    map_path = ctx.outputs / "marswalk_static_demo.png"

    # Arbitrary test-crop endpoints; snap each to the nearest valid pixel.
    start = snap_to(r.valid, 8, 8)
    goal = snap_to(r.valid, r.rows - 9, r.cols - 9)

    path_rows, path_cols = find_route(
        r.valid, r.slope, start, goal, r.pixel_size,
        no_route_msg="No path found across valid DEM pixels; try different endpoints.",
        broken_msg="Could not reconstruct the example path.",
    )

    # Map pixel centers into the DEM's projected coordinates.
    xs = r.transform.c + (path_cols + 0.5) * r.transform.a
    ys = r.transform.f + (path_rows + 0.5) * r.transform.e

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    dem_plot = axes[0].imshow(
        np.ma.masked_where(~r.valid, r.dem),
        extent=r.extent, origin="upper", cmap="terrain",
    )
    axes[0].plot(xs, ys, color="red", linewidth=2, label="Illustrative route")
    axes[0].scatter([xs[0], xs[-1]], [ys[0], ys[-1]], c=["lime", "red"],
                    edgecolor="black", s=55, zorder=3)
    axes[0].set_title("Elevation values + example route")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")
    axes[0].legend(loc="lower right")
    fig.colorbar(dem_plot, ax=axes[0], shrink=0.75, label="DEM value (units unverified)")

    slope_plot = axes[1].imshow(
        np.ma.masked_where(~r.valid, r.slope),
        extent=r.extent, origin="upper", cmap="magma", vmin=0, vmax=8,
    )
    axes[1].plot(xs, ys, color="cyan", linewidth=2)
    axes[1].set_title("Provisional slope + same route")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")
    fig.colorbar(slope_plot, ax=axes[1], shrink=0.75,
                 label="Slope (degrees; vertical units unverified)")

    fig.suptitle(
        "MARSWALK — Jezero test crop\n"
        "Illustrative only: arbitrary endpoints; obstacles and safety not modeled",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(map_path, dpi=180, bbox_inches="tight")
    finish_figure(ctx, fig)

    print(f"Saved static demo map: {map_path}")
    print(f"Example path length: {len(path_rows)} grid cells")


# ---- Cell 6: route planner (was an ipywidgets UI; now command-line options) ---------

def plan_route_png(ctx: Context, r: DemoRasters, start_row, start_col, end_row, end_col,
                   max_slope: float, layer_choice: str) -> bool:
    """Plan an illustrative route and save ``marswalk_prototype.png``.

    Returns True on success; prints a message and returns False if there is no route.
    """
    map_path = ctx.outputs / "marswalk_prototype.png"

    start = snap_to(r.valid, start_row, start_col)
    goal = snap_to(r.valid, end_row, end_col)

    # Demo-only routing cost. The cutoff is a user-selected display
    # parameter, not a validated astronaut/suit safety limit.
    allowed = r.valid & (r.slope <= max_slope)
    if not allowed[start] or not allowed[goal]:
        print("Start or destination exceeds the selected slope cutoff. "
              "Increase --slope-cutoff and re-plan.")
        return False

    try:
        pr, pc = find_route(
            allowed, r.slope, start, goal, r.pixel_size,
            no_route_msg="No route found under this slope cutoff. Try increasing it.",
            broken_msg="Could not reconstruct route.",
        )
    except RouteError as exc:
        print(exc)
        return False

    # Approximate map coordinates from the raster transform.
    px = r.transform.c + (pc + 0.5) * r.transform.a
    py = r.transform.f + (pr + 0.5) * r.transform.e

    if layer_choice == "Elevation":
        layer, cmap = r.dem, "terrain"
        label = "DEM value (units unverified)"
    else:
        layer, cmap = r.slope, "magma"
        label = "Slope (degrees; vertical units unverified)"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(
        np.ma.masked_where(~r.valid, layer),
        extent=r.extent, origin="upper", cmap=cmap,
    )
    ax.plot(px, py, color="cyan", linewidth=2.5, label="Illustrative route")
    ax.scatter([px[0], px[-1]], [py[0], py[-1]],
               c=["lime", "red"], edgecolor="black", s=65, zorder=3)
    ax.set_title(f"Marswalk prototype — {layer_choice}")
    ax.set_xlabel("Easting in DEM coordinate system (m)")
    ax.set_ylabel("Northing in DEM coordinate system (m)")
    ax.legend()
    fig.colorbar(im, ax=ax, label=label)
    fig.suptitle("Test crop · illustrative route · not a safety assessment")
    fig.tight_layout()
    fig.savefig(map_path, dpi=160, bbox_inches="tight")
    finish_figure(ctx, fig)

    step_lengths = np.hypot(np.diff(pr), np.diff(pc)) * r.pixel_size
    route_length = float(step_lengths.sum())
    route_max_slope = float(np.max(r.slope[pr, pc]))
    print(f"Approximate route length: {route_length:.0f} m")
    print(f"Maximum sampled slope on route: {route_max_slope:.1f}° (provisional)")
    print(f"Map saved to: {map_path}")
    print(f"Raster CRS: {r.crs}")
    return True


def step_plan_route(ctx: Context) -> None:
    """Plan a route between two test-crop pixels (see --start-row, --slope-cutoff, ...)."""
    a = ctx.args
    r = load_demo_rasters(ctx)
    # Controls use pixel positions in the 150 x 150 test crop.
    end_row = a.end_row if a.end_row is not None else r.rows - 9
    end_col = a.end_col if a.end_col is not None else r.cols - 9
    plan_route_png(ctx, r, a.start_row, a.start_col, end_row, end_col,
                   a.slope_cutoff, a.layer)


# ---- Cell 7: Leaflet route map -------------------------------------------------------

def step_leaflet_route(ctx: Context) -> None:
    """Interactive Leaflet map of the test crop with an illustrative route."""
    import folium
    from folium.plugins import Fullscreen

    r = load_demo_rasters(ctx)
    map_path = ctx.outputs / "marswalk_leaflet_map.html"
    rows, cols, valid = r.rows, r.cols, r.valid

    # Convert data to embedded RGBA images; transparent pixels represent nodata.
    def rgba_layer(data, cmap_name, vmin=None, vmax=None):
        vals = data[valid]
        norm = Normalize(
            vmin=float(np.min(vals)) if vmin is None else vmin,
            vmax=float(np.max(vals)) if vmax is None else vmax,
        )
        rgba = plt.get_cmap(cmap_name)(norm(np.where(valid, data, 0)))
        rgba[..., 3] = valid.astype(float) * 0.88
        return (rgba * 255).astype(np.uint8)

    elevation_image = rgba_layer(r.dem, "terrain")
    slope_image = rgba_layer(r.slope, "magma", vmin=0, vmax=8)

    # Find a demonstration route, restricted by a provisional slope cutoff.
    max_slope_demo = 10.0
    allowed = valid & (r.slope <= max_slope_demo)
    if not allowed.any():
        raise ValueError("No pixels meet the provisional slope cutoff.")

    start = snap_to(allowed, 8, 8)
    goal = snap_to(allowed, rows - 9, cols - 9)

    # pixel_size=1.0: costs are in pixel units here, as in the notebook.
    path_rows, path_cols = find_route(
        allowed, r.slope, start, goal, 1.0,
        no_route_msg="No route found; try a higher provisional slope cutoff.",
    )

    # Leaflet CRS.Simple coordinates use pixel units here.
    # Negative row values keep image rows aligned with the overlay.
    bounds = [[-rows, 0], [0, cols]]
    route_points = [
        [-float(row + 0.5), float(col + 0.5)]
        for row, col in zip(path_rows, path_cols)
    ]

    m = new_pixel_map([-rows / 2, cols / 2], 650)

    folium.raster_layers.ImageOverlay(
        image=elevation_image, bounds=bounds, name="Elevation",
        opacity=0.9, interactive=True, cross_origin=False, zindex=1, show=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=slope_image, bounds=bounds, name="Provisional slope",
        opacity=0.85, interactive=True, cross_origin=False, zindex=2, show=False,
    ).add_to(m)

    folium.PolyLine(
        route_points, color="#00e5ff", weight=4,
        tooltip="Illustrative route — not safety guidance",
    ).add_to(m)

    folium.Marker(
        route_points[0],
        tooltip="Example start",
        popup="Illustrative start point in the test crop",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)

    folium.Marker(
        route_points[-1],
        tooltip="Example destination",
        popup="Illustrative destination in the test crop",
        icon=folium.Icon(color="red", icon="flag"),
    ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    m.fit_bounds(bounds, padding=[25, 25], max_zoom=1)

    add_notice(m, """
<div style="position:fixed;bottom:24px;left:50px;z-index:9999;
background:white;padding:10px 14px;border:1px solid #777;
border-radius:6px;max-width:340px;font:14px/1.4 Arial,sans-serif">
<b>Marswalk prototype</b><br>
Test crop and illustrative route. Slope is provisional; DEM vertical units
are unverified. Not mission or safety guidance.
</div>
""")

    m.save(str(map_path))
    print(f"Saved interactive map: {map_path}")


# ---- Cell 8: package the demo as a standalone HTML + README ---------------------------

def step_package_demo(ctx: Context) -> None:
    """Embed the planner PNG in a standalone HTML page and write a short README."""
    map_path = ctx.outputs / "marswalk_prototype.png"
    html_path = ctx.outputs / "marswalk_demo.html"
    readme_path = ctx.documentation / "README_demo.md"

    if not map_path.is_file():
        raise FileNotFoundError(
            f"{map_path} not found. Run the 'plan-route' step first to generate a map."
        )

    image_data = base64.b64encode(map_path.read_bytes()).decode("ascii")
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Marswalk — Jezero prototype</title>
<style>
  body {{ margin: 0; background: #eceee8; color: #1d2733;
          font: 16px/1.6 system-ui, sans-serif; }}
  main {{ max-width: 1000px; margin: auto; padding: 28px 20px 56px; }}
  h1 {{ margin-bottom: 4px; }}
  .note {{ padding: 14px 18px; border-left: 5px solid #c0367a;
           background: #fff; margin: 18px 0; }}
  img {{ display: block; width: 100%; height: auto; background: white; }}
  .muted {{ color: #58636f; }}
</style>
</head>
<body>
<main>
<h1>Marswalk: Jezero route-planning prototype</h1>
<p class="muted">A script-based prototype using a 20 m CTX DEM test crop. Generated {escape(generated)}.</p>
<img src="data:image/png;base64,{image_data}" alt="Elevation or slope map with an illustrative route">
<div class="note">
<strong>Prototype, not mission guidance.</strong>
The displayed crop is a centered test area, not a verified science site.
The example route uses a simplified, user-selected slope cutoff. The DEM's
vertical units have not been verified. This prototype does not model obstacles,
radiation, weather, suit mobility, or astronaut safety.
</div>
<h2>Data and provenance</h2>
<p>The input raster is
<code>M20_JezeroCrater_CTXDEM_20m.tif</code> from this project's Raw data folder.
Record its original product citation, provider, download URL, and license in the
project documentation before presenting it as a sourced NASA product. Other
challenge references (Mars Trek, USGS–NASA geologic maps, Perseverance images,
and the Curiosity route animation) are planned research sources or design
references; they are not represented as layers in this map.</p>
<h2>What the prototype demonstrates</h2>
<ul>
<li>Loads a georeferenced Mars DEM crop and derived provisional slope raster.</li>
<li>Lets the user select endpoints, a displayed layer, and a slope cutoff (command-line options or the Gradio app).</li>
<li>Computes and visualizes an illustrative route over valid raster cells.</li>
</ul>
</main>
</body>
</html>
"""

    html_path.write_text(html_content, encoding="utf-8")

    readme_content = """# Marswalk — prototype demo

## Run
Run the raster-processing and planner steps of `Marswalk_EVA_Map.py`, e.g.
`python Marswalk_EVA_Map.py --project-root <Marswalk folder> --steps crop,slope,plan-route`.
Adjust `--start-row`, `--start-col`, `--end-row`, `--end-col`, `--slope-cutoff` and
`--layer` to change the route. Then run `--steps package-demo` to refresh
`Outputs/marswalk_demo.html`.

## Current prototype
- Input: `Raw data/M20_JezeroCrater_CTXDEM_20m.tif`
- Test crop: `Processed data/jezero_test_corridor_dem.tif`
- Provisional slope: `Processed data/jezero_test_corridor_slope.tif`
- Interactive map image: `Outputs/marswalk_prototype.png`
- Shareable demo: `Outputs/marswalk_demo.html`

## Important limitations
The crop is centered for testing and is not a verified science site. Vertical
units have not been confirmed. The route and slope cutoff are illustrative;
this is not a validated safety or mission-planning tool. Obstacles, radiation,
weather, and suit mobility are not modeled.

## Source follow-up
Document the original DEM product citation, provider, download URL, and license.
Do not imply that reference portals or example videos are data layers unless
specific products from them are actually included.
"""
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(readme_content, encoding="utf-8")

    print(f"Shareable demo: {html_path}")
    print(f"Project notes:  {readme_path}")
    print("Open marswalk_demo.html in a browser, or send the file, to share the standalone demo.")


# ---- Cell 9: Gradio frontend (blocking; not part of the default run) -------------------

def step_gradio(ctx: Context) -> None:
    """Launch the Gradio route-planner app (runs until you stop it with Ctrl+C)."""
    import gradio as gr

    r = load_demo_rasters(ctx)
    rows, cols = r.rows, r.cols

    # Hillshade is derived from the DEM; it is a visualization layer,
    # not an independent dataset. Vertical units are unverified.
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
        np.ma.array(r.dem, mask=~r.valid),
        vert_exag=1,
        dx=r.pixel_size,
        dy=r.pixel_size,
    )

    def make_map(start_row, start_col, end_row, end_col, max_slope, layer):
        allowed = r.valid & (r.slope <= max_slope)
        if not allowed.any():
            return None, "No valid terrain at this cutoff. Increase the slope cutoff."

        start = snap_to(allowed, start_row, start_col)
        goal = snap_to(allowed, end_row, end_col)

        try:
            path_rows, path_cols = find_route(
                allowed, r.slope, start, goal, r.pixel_size,
                no_route_msg="No route found. Try increasing the slope cutoff.",
                broken_msg="Could not reconstruct a route. Try different endpoints.",
            )
        except RouteError as exc:
            return None, str(exc)

        x = r.transform.c + (path_cols + 0.5) * r.transform.a
        y = r.transform.f + (path_rows + 0.5) * r.transform.e

        if layer == "Elevation":
            image_data = np.ma.array(r.dem, mask=~r.valid)
            cmap = "terrain"
            colorbar_label = "DEM value (units unverified)"
        elif layer == "Slope":
            image_data = np.ma.array(r.slope, mask=~r.valid)
            cmap = "magma"
            colorbar_label = "Slope (degrees; vertical units unverified)"
        else:
            image_data = np.ma.array(hillshade, mask=~r.valid)
            cmap = "gray"
            colorbar_label = "Relative illumination (qualitative)"

        fig, ax = plt.subplots(figsize=(8, 6))
        plotted = ax.imshow(image_data, extent=r.extent, origin="upper", cmap=cmap)
        ax.plot(x, y, color="cyan", linewidth=2.5, label="Illustrative route")
        ax.scatter(
            [x[0], x[-1]], [y[0], y[-1]],
            c=["lime", "red"], edgecolor="black", s=60, zorder=3,
        )
        ax.set_title(f"Jezero test crop — {layer}")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.legend()

        fig.colorbar(plotted, ax=ax, label=colorbar_label)
        fig.suptitle("Prototype only — not a safety assessment")
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
        plt.close(fig)
        buffer.seek(0)
        result_image = Image.open(buffer).copy()

        route_length_m = np.hypot(np.diff(path_rows), np.diff(path_cols)).sum() * r.pixel_size
        max_route_slope = float(np.max(r.slope[path_rows, path_cols]))

        note = (
            f"Approximate route length: **{route_length_m:.0f} m**  \n"
            f"Maximum sampled slope: **{max_route_slope:.1f}° (provisional)**  \n\n"
            "**Limitations:** centered test crop; vertical units unverified; "
            "obstacles, radiation, weather, suit mobility, and astronaut safety "
            "are not modeled. Hillshade is derived from the DEM."
        )
        return result_image, note

    with gr.Blocks(title="Marswalk — Jezero prototype") as app:
        gr.Markdown(
            "# 🪐 Marswalk — Jezero prototype\n"
            "Select a map layer and generate an illustrative route. "
            "**This is not mission guidance or a safety assessment.**"
        )

        with gr.Row():
            layer = gr.Dropdown(
                ["Elevation", "Slope", "Hillshade"], value="Elevation", label="Map layer",
            )
            slope_limit = gr.Slider(
                1, 20, value=10, step=0.5, label="Illustrative slope cutoff (°)",
            )

        with gr.Row():
            start_row = gr.Slider(0, rows - 1, value=8, step=1, label="Start row")
            start_col = gr.Slider(0, cols - 1, value=8, step=1, label="Start column")

        with gr.Row():
            end_row = gr.Slider(0, rows - 1, value=rows - 9, step=1, label="Destination row")
            end_col = gr.Slider(0, cols - 1, value=cols - 9, step=1, label="Destination column")

        button = gr.Button("Plan / update route", variant="primary")
        map_output = gr.Image(label="Prototype map")
        note_output = gr.Markdown()

        button.click(
            fn=make_map,
            inputs=[start_row, start_col, end_row, end_col, slope_limit, layer],
            outputs=[map_output, note_output],
        )

    # share=True creates a public gradio.live link (needed in Colab, optional locally).
    app.launch(share=ctx.args.share)


# ---- Cell 10: movable overview map of the full DEM -------------------------------------

def step_overview_map(ctx: Context) -> None:
    """Leaflet overview of the full DEM (elevation + hillshade)."""
    import folium
    from folium.plugins import Fullscreen

    map_path = ctx.outputs / "marswalk_jezero_movable_map.html"

    # Read a reduced-resolution overview of the full DEM for a faster web map.
    with rasterio.open(ctx.ctx_dem) as src:
        dem_small, height, width = read_overview(src)
        dem_small = dem_small.astype(np.float32)
        crs_text = str(src.crs)

    valid = ~np.ma.getmaskarray(dem_small) & np.isfinite(dem_small.data)
    values = dem_small.data[valid]
    if values.size == 0:
        raise ValueError("No valid DEM pixels found.")

    # Use the full raster's downsampled pixel coordinates for the web map.
    # This is a local Jezero view, not a global Mars coordinate system.
    bounds = [[-height, 0], [0, width]]
    center = [-height / 2, width / 2]

    # Colorized elevation layer.
    norm = Normalize(vmin=float(np.percentile(values, 2)),
                     vmax=float(np.percentile(values, 98)))
    elevation_rgba = plt.get_cmap("terrain")(norm(np.where(valid, dem_small.data, values.min())))
    elevation_rgba[..., 3] = valid.astype(float) * 0.95
    elevation_rgba = (elevation_rgba * 255).astype(np.uint8)

    # Shaded relief, derived from this same DEM.
    filled = np.where(valid, dem_small.data, np.nanmedian(values))
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(filled, vert_exag=1, dx=1, dy=1)
    hill_rgba = plt.get_cmap("gray")(hillshade)
    hill_rgba[..., 3] = valid.astype(float) * 0.9
    hill_rgba = (hill_rgba * 255).astype(np.uint8)

    m = new_pixel_map(center, 700)

    folium.raster_layers.ImageOverlay(
        image=elevation_rgba, bounds=bounds, name="Elevation",
        opacity=0.95, interactive=True, cross_origin=False, show=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=hill_rgba, bounds=bounds, name="Hillshade (derived from DEM)",
        opacity=0.9, interactive=True, cross_origin=False, show=False,
    ).add_to(m)

    Fullscreen(position="topright").add_to(m)
    folium.LatLngPopup().add_to(m)
    m.fit_bounds(bounds, padding=[20, 20], max_zoom=1)

    add_notice(m, """
<div style="position:fixed;bottom:24px;left:50px;z-index:9999;
background:white;padding:10px 14px;border:1px solid #777;
border-radius:6px;max-width:380px;font:14px/1.4 Arial,sans-serif">
<b>Marswalk — Jezero DEM overview</b><br>
Pan and zoom within the coverage of the input DEM. Click the map to inspect
local display coordinates. Hillshade is derived from the same DEM.
DEM vertical units are unverified; this is not mission or safety guidance.
</div>
""")

    m.save(str(map_path))
    print(f"Saved movable map: {map_path}")
    print(f"Showing downsampled overview: {width} × {height} pixels")
    print(f"Source CRS: {crs_text}")


# ---- Cell 11: download + extract the USGS geology archive ----------------------------------

def step_download_usgs(ctx: Context) -> None:
    """Download (if missing) and safely extract the USGS SIM 3464 geology GIS archive."""
    zip_path = ctx.raw / "usgs_jezero_geology_sim3464.zip"
    extract_dir = ctx.raw / "USGS_SIM3464"
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    # Download only if the archive isn't already present.
    if not zip_path.exists():
        print("Downloading the USGS Jezero geology GIS archive (~349 MB)...")
        request = Request(USGS_URL, headers={"User-Agent": "MarswalkPrototype/1.0"})
        with urlopen(request, timeout=60) as response, zip_path.open("wb") as out:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0

            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    print(f"\rDownloaded {downloaded / total:.0%}", end="")

        print("\nDownload complete.")
    else:
        print(f"Using existing archive: {zip_path}")

    # Extract once, and prevent archive paths from writing outside the target folder.
    extract_dir.mkdir(parents=True, exist_ok=True)
    root = extract_dir.resolve()

    try:
        with ZipFile(zip_path) as archive:
            for member in archive.infolist():
                destination = (extract_dir / member.filename).resolve()
                if destination != root and root not in destination.parents:
                    raise ValueError(f"Unsafe archive path: {member.filename}")
            archive.extractall(extract_dir)
    except BadZipFile as exc:
        raise RuntimeError(f"The downloaded file is not a valid ZIP: {zip_path}") from exc

    vector_files = []
    for pattern in ("*.shp", "*.gpkg", "*.geojson", "*.gdb"):
        vector_files.extend(extract_dir.rglob(pattern))

    print(f"\nExtracted to: {extract_dir}")
    if vector_files:
        print("Vector datasets found:")
        for path in vector_files:
            print(f"- {path.relative_to(extract_dir)}")
    else:
        print("No common vector files found. Check the extracted file listing.")


# ---- Cell 12: list geodatabase layers ---------------------------------------------------------

def step_inspect_gdb(ctx: Context) -> None:
    """List the layers in Nili.gdb."""
    import fiona

    # Search within the Marswalk project folder for the geodatabase directory.
    gdb_matches = list(ctx.root.rglob("Nili.gdb"))
    if not gdb_matches:
        raise FileNotFoundError(
            f"Could not find Nili.gdb under {ctx.root}. "
            "Make sure the geodatabase folder is inside your Marswalk project folder."
        )

    gdb_path = gdb_matches[0]
    print(f"Geodatabase found: {gdb_path}")
    print("\nLayers:")

    for layer_name in fiona.listlayers(str(gdb_path)):
        with fiona.open(str(gdb_path), layer=layer_name) as layer:
            print(f"\n- {layer_name}")
            print(f"  Feature count: {len(layer)}")
            print(f"  CRS: {layer.crs_wkt or layer.crs}")
            print(f"  Bounds: {layer.bounds}")
            print(f"  Geometry: {layer.schema.get('geometry')}")
            print(f"  Fields: {list(layer.schema.get('properties', {}).keys())}")


# ---- Cell 13: rasterize the geology onto the DEM grid -------------------------------------------

def step_geology(ctx: Context) -> None:
    """Rasterize GeoUnits_v1 onto the CTX DEM grid; save GeoTIFF, overlay PNG, legend CSV."""
    import geopandas as gpd
    from rasterio.features import rasterize
    from shapely.geometry import box

    gdb_path = ctx.gdb
    geology_raster_path = ctx.geology_raster
    overlay_path = ctx.outputs / "jezero_geology_overlay.png"
    preview_path = ctx.outputs / "jezero_geology_quicklook.png"
    legend_path = ctx.geology_legend

    if not gdb_path.exists():
        raise FileNotFoundError(f"Geodatabase not found: {gdb_path}")

    print("Reading GeoUnits_v1 and transforming it to the DEM CRS...")
    geology = gpd.read_file(str(gdb_path), layer="GeoUnits_v1", engine="fiona")

    with rasterio.open(ctx.ctx_dem) as dem_src:
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_bounds = dem_src.bounds
        dem_width, dem_height = dem_src.width, dem_src.height
        dem_profile = dem_src.profile.copy()
        dem_preview, display_height, display_width = read_overview(dem_src)

    if geology.crs is None:
        raise ValueError("GeoUnits_v1 has no CRS metadata; stop rather than guess its projection.")

    geology = geology.to_crs(dem_crs)
    geology = geology[geology.geometry.notna() & ~geology.geometry.is_empty].copy()

    # Confirm that reprojected polygons overlap the DEM before rasterizing.
    dem_box = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    overlapping = geology[geology.geometry.intersects(dem_box)].copy()

    print(f"Features in GeoUnits_v1: {len(geology)}")
    print(f"Features overlapping the DEM: {len(overlapping)}")
    print(f"DEM bounds in its CRS: {dem_bounds}")

    if overlapping.empty:
        raise ValueError(
            "No geology polygons overlap the DEM after reprojection. "
            "Do not add this layer until the CRS definitions are checked."
        )

    # Use the geologic unit symbol as the category; fall back to unit/name.
    overlapping["map_category"] = overlapping.apply(
        lambda row: unit_name(row, "Unlabeled unit"), axis=1
    )
    categories = sorted(overlapping["map_category"].unique())
    category_to_id = {category: i + 1 for i, category in enumerate(categories)}

    shapes = [
        (geom, category_to_id[category])
        for geom, category in zip(overlapping.geometry, overlapping["map_category"])
    ]

    print("Rasterizing geology polygons onto the DEM grid...")
    geology_ids = rasterize(
        shapes=shapes,
        out_shape=(dem_height, dem_width),
        transform=dem_transform,
        fill=0,
        dtype="uint16",
        all_touched=False,
    )

    covered_pixels = int(np.count_nonzero(geology_ids))
    if covered_pixels == 0:
        raise ValueError(
            "Reprojected polygons intersect the DEM bounds but rasterize to no pixels. "
            "Check the coordinate transformation and map coverage."
        )

    # Save a categorical GeoTIFF aligned exactly with the source DEM.
    for key in ("blockxsize", "blockysize", "tiled", "predictor"):
        dem_profile.pop(key, None)
    dem_profile.update(count=1, dtype="uint16", nodata=0, compress="deflate")

    geology_raster_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(geology_raster_path, "w", **dem_profile) as dst:
        dst.write(geology_ids, 1)
        dst.set_band_description(1, "USGS SIM 3464 geologic unit category ID")
        dst.update_tags(
            source="USGS SIM 3464, GeoUnits_v1 layer",
            alignment="Reprojected to and rasterized on the CTX DEM grid",
            category_field="symbol, with unit/name fallback",
        )

    # Make a transparent PNG the same reduced size as the full-DEM Leaflet overview.
    small_ids = np.asarray(
        Image.fromarray(geology_ids).resize(
            (display_width, display_height), resample=Image.Resampling.NEAREST,
        )
    )

    palette = plt.get_cmap("tab20")
    rgba = np.zeros((display_height, display_width, 4), dtype=np.uint8)
    for category, category_id in category_to_id.items():
        color = np.array(palette((category_id - 1) % 20)) * 255
        rgba[small_ids == category_id] = color.astype(np.uint8)
        rgba[small_ids == category_id, 3] = 190  # partial transparency

    ctx.outputs.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(overlay_path)  # (H, W, 4) uint8 array -> RGBA

    # Save a category legend so map colors can be explained.
    with open(legend_path, "w", encoding="utf-8") as f:
        f.write("category_id,geologic_unit\n")
        for category, category_id in category_to_id.items():
            safe_category = category.replace('"', '""')
            f.write(f'{category_id},"{safe_category}"\n')

    # Show the geology overlay on the downsampled DEM for an alignment check.
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(dem_preview, extent=(0, dem_width, dem_height, 0), cmap="terrain")
    ax.imshow(rgba, extent=(0, dem_width, dem_height, 0))
    ax.set_title("USGS SIM 3464 geology over Jezero CTX DEM")
    ax.set_xlabel("DEM pixel column (overview)")
    ax.set_ylabel("DEM pixel row (overview)")
    fig.tight_layout()
    fig.savefig(preview_path, dpi=150)
    finish_figure(ctx, fig)

    print(f"Geology raster: {geology_raster_path}")
    print(f"Leaflet overlay PNG: {overlay_path}")
    print(f"Legend: {legend_path}")
    print(f"Quicklook: {preview_path}")
    print(f"Geology-covered DEM pixels: {covered_pixels:,}")


# ---- Cell 15: DEM corner longitude / latitude ---------------------------------------------------

def step_corners(ctx: Context) -> None:
    """Print the DEM's corner coordinates as Mars longitude/latitude."""
    from pyproj import CRS, Transformer

    with rasterio.open(ctx.ctx_dem) as src:
        mars_crs = CRS.from_wkt(src.crs.to_wkt())
        geographic_crs = mars_crs.geodetic_crs
        if geographic_crs is None:
            raise ValueError("Could not identify the DEM's Mars geographic CRS.")

        to_mars_lonlat = Transformer.from_crs(mars_crs, geographic_crs, always_xy=True)

        b = src.bounds
        corners = {
            "lower-left": (b.left, b.bottom),
            "lower-right": (b.right, b.bottom),
            "upper-left": (b.left, b.top),
            "upper-right": (b.right, b.top),
        }

        print("Geographic CRS from DEM:", geographic_crs.name)
        print("DEM corner coordinates (longitude, latitude):")
        for name, (x, y) in corners.items():
            lon, lat = to_mars_lonlat.transform(x, y)
            print(f"  {name}: {lon:.4f}°, {lat:.4f}°")

    print("\nThese are transformed DEM coordinates, not the Leaflet click-popup values.")


# ---- Cell 17: final terrain explorer map (replaces notebook cells 14 and 16) ---------------------

def build_final_map(ctx: Context):
    """Build the final Leaflet terrain explorer and store it in ``ctx.state['map']``."""
    import folium
    import geopandas as gpd
    from folium.plugins import Fullscreen
    from shapely.geometry import box, mapping
    from shapely.ops import transform as transform_geometry

    dem_path = ctx.ctx_dem
    gdb_path = ctx.gdb
    html_path = ctx.final_map_html

    # Load a downsampled full-DEM view so the browser map stays responsive.
    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        dem_transform = src.transform
        dem_bounds = src.bounds
        dem_width, dem_height = src.width, src.height
        dem_nodata = src.nodata

        dem_masked, view_height, view_width = read_overview(src)
        dem_masked = dem_masked.astype(np.float32)

    dem = dem_masked.filled(np.nan)
    valid = np.isfinite(dem)
    if dem_nodata is not None:
        valid &= dem != dem_nodata
    if not valid.any():
        raise ValueError("No valid DEM pixels found.")
    dem_values = dem[valid]

    # Read geology categories from the raster already aligned to the DEM.
    if not ctx.geology_raster.is_file():
        raise FileNotFoundError(
            f"{ctx.geology_raster} not found. Run the 'geology' step first."
        )
    with rasterio.open(ctx.geology_raster) as src:
        geology_ids = src.read(
            1, out_shape=(view_height, view_width), resampling=Resampling.nearest,
        )

    # Leaflet Simple CRS coordinates are local display pixels, not longitude/latitude.
    map_bounds = [[-view_height, 0], [0, view_width]]
    map_center = [-view_height / 2, view_width / 2]

    # Elevation image.
    elev_norm = Normalize(
        vmin=float(np.percentile(dem_values, 2)),
        vmax=float(np.percentile(dem_values, 98)),
    )
    elev_rgba = plt.get_cmap("terrain")(
        elev_norm(np.where(valid, dem, float(np.nanmin(dem_values))))
    )
    elev_rgba[..., 3] = valid.astype(float) * 0.96
    elevation_array = (elev_rgba * 255).astype(np.uint8)

    # Hillshade derived from this DEM.
    filled_dem = np.where(valid, dem, float(np.nanmedian(dem_values)))
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(
        filled_dem, vert_exag=1, dx=1, dy=1
    )
    hill_rgba = plt.get_cmap("gray")(hillshade)
    hill_rgba[..., 3] = valid.astype(float) * 0.95
    hillshade_array = (hill_rgba * 255).astype(np.uint8)

    # Calculate provisional slope for visual screening only.
    # Vertical units are unverified, so this is not a safety rating.
    dx = abs(dem_transform.a) * dem_width / view_width
    dy = abs(dem_transform.e) * dem_height / view_height
    grad_y, grad_x = np.gradient(filled_dem, dy, dx)
    slope_degrees = np.degrees(np.arctan(np.hypot(grad_x, grad_y)))

    slope_rgba = np.zeros((view_height, view_width, 4), dtype=np.uint8)
    slope_rgba[valid & (slope_degrees <= 5)] = (46, 160, 90, 200)
    slope_rgba[valid & (slope_degrees > 5) & (slope_degrees <= 10)] = (242, 180, 55, 205)
    slope_rgba[valid & (slope_degrees > 10)] = (205, 65, 65, 215)

    # Categorical geology image, transparent where no geology polygon was rasterized.
    geo_rgba = np.zeros((view_height, view_width, 4), dtype=np.uint8)
    for unit_id in sorted(int(v) for v in np.unique(geology_ids) if v > 0):
        color = (np.array(plt.get_cmap("tab20")((unit_id - 1) % 20)) * 255).astype(np.uint8)
        mask = (geology_ids == unit_id) & valid
        geo_rgba[mask, :3] = color[:3]
        geo_rgba[mask, 3] = 190

    # Build the map and add every toggleable layer. (A single LayerControl is
    # added by the 'meda-panel' step, after all layers are present.)
    m = new_pixel_map(map_center, 720)

    folium.raster_layers.ImageOverlay(
        image=elevation_array, bounds=map_bounds, name="Elevation",
        opacity=0.95, show=True,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=hillshade_array, bounds=map_bounds, name="Hillshade (derived)",
        opacity=0.85, show=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=geo_rgba, bounds=map_bounds, name="USGS geology units",
        opacity=0.80, show=False,
    ).add_to(m)

    folium.raster_layers.ImageOverlay(
        image=slope_rgba, bounds=map_bounds,
        name="Provisional slope screening — not safety",
        opacity=0.85, show=False,
    ).add_to(m)

    # Add clickable geology polygons and unit labels.
    geology = gpd.read_file(str(gdb_path), layer="GeoUnits_v1", engine="fiona")
    if geology.crs is None:
        raise ValueError("GeoUnits_v1 has no CRS metadata; cannot align it safely.")
    geology = geology.to_crs(dem_crs)

    dem_box = box(dem_bounds.left, dem_bounds.bottom, dem_bounds.right, dem_bounds.top)
    geology = geology[
        geology.geometry.notna()
        & ~geology.geometry.is_empty
        & geology.geometry.intersects(dem_box)
    ].copy()

    inverse_transform = ~dem_transform
    scale_x = view_width / dem_width
    scale_y = view_height / dem_height

    def to_map_pixels(x, y, z=None):
        col, row = inverse_transform * (x, y)
        px, py = col * scale_x, -row * scale_y
        return (px, py) if z is None else (px, py, z)

    features = []
    for _, row in geology.iterrows():
        geom = row.geometry.intersection(dem_box)
        if geom.is_empty:
            continue
        geom = geom.simplify(100, preserve_topology=True)
        geom = transform_geometry(to_map_pixels, geom)
        features.append({
            "type": "Feature",
            "geometry": mapping(geom),
            "properties": {"unit": unit_name(row, "Unlabeled geologic unit")},
        })

    if features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": features},
            name="Clickable geology boundaries",
            style_function=lambda feature: {
                "color": "#273444",
                "weight": 1,
                "opacity": 0.7,
                "fillOpacity": 0,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["unit"], aliases=["Geologic unit:"], sticky=True
            ),
            popup=folium.GeoJsonPopup(
                fields=["unit"], aliases=["Geologic unit:"], labels=True
            ),
            show=False,
        ).add_to(m)

    Fullscreen(position="topright").add_to(m)

    add_notice(m, """
<div style="position:fixed;bottom:22px;left:50px;z-index:9999;
background:#fff;padding:10px 14px;border:1px solid #777;border-radius:6px;
max-width:410px;font:14px/1.45 Arial,sans-serif">
<b>Marswalk · Jezero terrain explorer</b><br>
Use the layer menu at top right to toggle elevation, hillshade, geology,
and provisional slope. Click geology boundaries for unit names.<br>
Slope bands are illustrative only—not a safety rating. DEM vertical units
are unverified. Radiation, weather, obstacles, and suit mobility are not modeled.
Map click coordinates are local display pixels, not longitude/latitude.
</div>
""")

    m.fit_bounds(map_bounds, padding=[20, 20], max_zoom=1)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(html_path))

    ctx.state["map"] = m
    print(f"Saved interactive map: {html_path}")
    print(f"Geology boundary features: {len(features)}")
    print("Map layers: Elevation, Hillshade, USGS geology units, provisional slope, clickable geology boundaries")
    return m


def get_map(ctx: Context):
    """The shared Folium map; built first if no earlier step has created it."""
    if "map" not in ctx.state:
        print("(No map in memory yet — building the final map first.)")
        build_final_map(ctx)
    return ctx.state["map"]


def step_final_map(ctx: Context) -> None:
    """Build the final interactive terrain explorer map."""
    build_final_map(ctx)


# ---- Cell 18: download the NASA context image --------------------------------------------------

def step_nasa_image(ctx: Context) -> None:
    """Download the PIA25917 Perseverance image and its NASA metadata."""
    import requests

    # Download and save the image for offline use in the project.
    response = requests.get(NASA_IMAGE_URL, timeout=30)
    response.raise_for_status()

    image_path = ctx.nasa_image
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(response.content)

    # Fetch NASA asset metadata for title, description, and any listed credits.
    meta_response = requests.get(NASA_METADATA_URL, timeout=30)
    meta_response.raise_for_status()
    asset_metadata = meta_response.json()

    ctx.outputs.mkdir(parents=True, exist_ok=True)
    metadata_path = ctx.outputs / "PIA25917_metadata.json"
    metadata_path.write_text(json.dumps(asset_metadata, indent=2), encoding="utf-8")

    print(f"Saved image:    {image_path}")
    print(f"Saved metadata: {metadata_path}")
    print("NASA ID: PIA25917")
    print("Title: Perseverance Finds a Hollowed-Out Rock")
    print("Sol: 836")
    print("Instrument: Mastcam-Z")
    print("Image URL:", NASA_IMAGE_URL)
    print("Metadata URL:", NASA_METADATA_URL)


# ---- Cell 19: add the photo card to the map ------------------------------------------------------

def step_photo_card(ctx: Context) -> None:
    """Embed the NASA photo in the final map as an info card."""
    import folium

    image_path = ctx.nasa_image
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}. Run the 'nasa-image' step first.")

    m = get_map(ctx)
    encoded_image = base64.b64encode(image_path.read_bytes()).decode("ascii")

    title = escape("Perseverance Finds a Hollowed-Out Rock")
    description = escape(
        "Mastcam-Z image from Jezero Crater, sol 836. "
        "Context image only; its location is not plotted on this map."
    )
    source_url = "https://images.nasa.gov/details/PIA25917"

    photo_card = f"""
<div style="
 position:fixed; top:auto; right:12px; bottom:52px; left:auto;
 z-index:9999; width:260px; max-height:40vh; overflow:auto;
 background:#fff; border:1px solid #536273; border-radius:9px;
 padding:10px; box-shadow:0 2px 8px #0005;
 font:13px/1.45 Arial,sans-serif;">
 <b>{title}</b>
 <img src="data:image/jpeg;base64,{encoded_image}"
      alt="{title}, captured by Perseverance"
      style="display:block;width:100%;height:auto;margin:8px 0;border-radius:5px;">
 <div>{description}</div>
 <div style="margin-top:7px">
   <a href="{source_url}" target="_blank" rel="noopener">
     NASA image record · PIA25917
   </a>
 </div>
</div>
"""

    folium.Element(photo_card).add_to(m.get_root().html)
    m.save(str(ctx.final_map_html))
    print(f"Updated interactive map: {ctx.final_map_html}")


# ---- Cell 20: find + inspect the HiRISE DTM and MEDA CSVs -----------------------------------------

def step_hirise_inspect(ctx: Context) -> None:
    """Locate the HiRISE DTM and MEDA CSVs under Raw data and report their metadata."""
    raw_dir = ctx.raw
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Can't find {raw_dir}. Check --project-root points to your Marswalk folder."
        )

    # Find the HiRISE DTM and the MEDA CSVs wherever they are under Raw data.
    dtm_matches = list(raw_dir.rglob(HIRISE_DTM_NAME))
    ps_matches = list(raw_dir.rglob("WE*DER_PS*.csv"))
    tirs_matches = list(raw_dir.rglob("WE*DER_TIRS*.csv"))
    rhs_matches = list(raw_dir.rglob("WE*DER_RHS*.csv"))

    def show_matches(label, matches):
        print(f"\n{label}:")
        if matches:
            for path in matches[:10]:
                print(" ", path)
        else:
            print("  Not found under Raw data.")

    show_matches("HiRISE DTM", dtm_matches)
    show_matches("MEDA pressure (PS)", ps_matches)
    show_matches("MEDA thermal infrared (TIRS)", tirs_matches)
    show_matches("MEDA humidity (RHS)", rhs_matches)

    # Try to open the DTM and report its real raster metadata.
    if dtm_matches:
        dtm_path = dtm_matches[0]
        print(f"\n--- HiRISE DTM inspection: {dtm_path.name} ---")

        try:
            with rasterio.open(dtm_path) as src:
                print("Driver:", src.driver)
                print("Dimensions (width × height):", src.width, "×", src.height)
                print("Band count:", src.count)
                print("Data type:", src.dtypes)
                print("CRS:", src.crs)
                print("Bounds:", src.bounds)
                print("Pixel size:", src.res)
                print("Nodata:", src.nodata)
                print("Raster tags:", src.tags())

                # Read a small sample for a quick validity check, not the full 700 MB raster.
                sample = src.read(
                    1,
                    out_shape=(min(300, src.height), min(300, src.width)),
                    masked=True,
                )
                print("Sample elevation min/max:", sample.min(), sample.max())

        except Exception as exc:
            print("Rasterio could not open this .IMG directly:")
            print(type(exc).__name__ + ":", exc)
            print(
                "Check the download folder for a matching .LBL label file "
                "or a product README."
            )

        # Look for the existing regional CTX DEM so we can compare coverage.
        ctx_matches = list(raw_dir.rglob("*CTXDEM*20m*.tif"))
        if ctx_matches:
            print("\n--- Existing CTX DEM footprint ---")
            try:
                with rasterio.open(ctx_matches[0]) as ctx_src:
                    print("File:", ctx_matches[0].name)
                    print("CRS:", ctx_src.crs)
                    print("Bounds:", ctx_src.bounds)
                    print("Pixel size:", ctx_src.res)
            except Exception as exc:
                print("Could not inspect CTX DEM:", exc)

    # Inspect MEDA CSV columns, time ranges, and a few rows.
    for label, matches in [
        ("PS pressure", ps_matches),
        ("TIRS infrared", tirs_matches),
        ("RHS humidity", rhs_matches),
    ]:
        if not matches:
            continue

        path = matches[0]
        print(f"\n--- MEDA {label}: {path.name} ---")
        try:
            df = pd.read_csv(path)
            print("Rows:", len(df))
            print("Columns:", list(df.columns))

            for time_col in ("LMST", "LTST"):
                if time_col in df.columns and len(df):
                    print(f"{time_col} first/last:", df[time_col].iloc[0], "/", df[time_col].iloc[-1])

            print("First 3 rows:")
            print(df.head(3).to_string(index=False))
        except Exception as exc:
            print("Could not read CSV:", type(exc).__name__ + ":", exc)


# ---- Cell 21: reproject the HiRISE DTM into the CTX CRS --------------------------------------------

def step_hirise_align(ctx: Context) -> None:
    """Reproject/clip the HiRISE DTM to the CTX DEM's CRS at 5 m resolution."""
    from rasterio.transform import from_bounds
    from rasterio.warp import reproject, transform_bounds

    hirise_path = ctx.hirise_dtm
    ctx_path = ctx.ctx_dem
    aligned_path = ctx.hirise_aligned
    preview_path = ctx.outputs / "HiRISE_Jezero_DTM_in_CTX_CRS_5m_preview.png"
    ctx.outputs.mkdir(parents=True, exist_ok=True)

    if not hirise_path.exists():
        raise FileNotFoundError(f"HiRISE DTM not found: {hirise_path}")
    if not ctx_path.exists():
        raise FileNotFoundError(f"CTX DEM not found: {ctx_path}")

    target_resolution_m = 5.0

    with rasterio.open(hirise_path) as hirise, rasterio.open(ctx_path) as ctx_src:
        if hirise.crs is None or ctx_src.crs is None:
            raise ValueError("One of the rasters has no CRS; we must not guess its projection.")

        print("HiRISE source CRS:", hirise.crs)
        print("HiRISE source pixel size:", hirise.res)
        print("CTX target CRS:", ctx_src.crs)

        # Convert the HiRISE footprint into the CTX projection, then clip to CTX coverage.
        transformed_bounds = transform_bounds(
            hirise.crs, ctx_src.crs, *hirise.bounds, densify_pts=41,
        )
        ctx_bounds = ctx_src.bounds

        left = max(transformed_bounds[0], ctx_bounds.left)
        bottom = max(transformed_bounds[1], ctx_bounds.bottom)
        right = min(transformed_bounds[2], ctx_bounds.right)
        top = min(transformed_bounds[3], ctx_bounds.top)

        print("\nHiRISE footprint transformed to CTX CRS:", transformed_bounds)
        print("CTX bounds:", ctx_bounds)

        if left >= right or bottom >= top:
            raise ValueError(
                "The transformed HiRISE DTM footprint does not overlap the CTX DEM. "
                "Check the printed CRS/bounds to diagnose it."
            )

        print("Overlap bounds:", (left, bottom, right, top))

        width = math.ceil((right - left) / target_resolution_m)
        height = math.ceil((top - bottom) / target_resolution_m)

        if width * height > 30_000_000:
            raise MemoryError(
                f"Output would contain {width * height:,} pixels. "
                "Use a coarser target_resolution_m before proceeding."
            )

        dst_transform = from_bounds(left, bottom, right, top, width, height)
        destination = np.full((height, width), np.nan, dtype=np.float32)

        reproject(
            source=rasterio.band(hirise, 1),
            destination=destination,
            src_transform=hirise.transform,
            src_crs=hirise.crs,
            src_nodata=hirise.nodata,
            dst_transform=dst_transform,
            dst_crs=ctx_src.crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )

        valid = np.isfinite(destination)
        if not valid.any():
            raise ValueError("Reprojection produced no valid elevation pixels.")

        profile = {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": "float32",
            "crs": ctx_src.crs,
            "transform": dst_transform,
            "nodata": np.nan,
            "compress": "deflate",
            "predictor": 3,
        }

        with rasterio.open(aligned_path, "w", **profile) as out:
            out.write(destination, 1)
            out.set_band_description(1, "HiRISE stereo DTM elevation")
            out.update_tags(
                source_product="DTEEC_023524_1985_023379_1985_U01",
                source_file=hirise_path.name,
                processing="Reprojected to CTX DEM CRS; bilinear resampling; 5 m output",
                note="Derived visualization/analysis copy; original DTM retained unchanged",
            )

        print("\nSaved aligned DTM:", aligned_path)
        print("Output dimensions:", width, "×", height)
        print("Valid pixel count:", int(valid.sum()))
        print("Elevation min/max (source units):",
              float(np.nanmin(destination)), "/", float(np.nanmax(destination)))

    # Reduced preview of the reprojected DTM footprint (saved, since there is no inline display).
    with rasterio.open(aligned_path) as aligned:
        preview = aligned.read(
            1,
            out_shape=(min(700, aligned.height), min(700, aligned.width)),
            masked=True,
        )
    fig = plt.figure(figsize=(9, 7))
    plt.imshow(preview, cmap="terrain")
    plt.colorbar(label="Elevation (check product label for units)")
    plt.title("HiRISE DTM reprojected to CTX map projection")
    plt.axis("off")
    fig.savefig(preview_path, dpi=150, bbox_inches="tight")
    finish_figure(ctx, fig)
    print("Saved preview:", preview_path)


# ---- Cell 22: look for the HiRISE DTM label ---------------------------------------------------------

def step_hirise_labels(ctx: Context) -> None:
    """Print unit/datum/scale lines from the HiRISE label (or embedded metadata)."""
    dtm_path = ctx.hirise_dtm

    # Look for a matching label or documentation alongside the IMG.
    possible_labels = [
        dtm_path.with_suffix(".LBL"),
        dtm_path.with_suffix(".lbl"),
        dtm_path.with_suffix(".XML"),
        dtm_path.with_suffix(".xml"),
    ]
    found_labels = [p for p in possible_labels if p.exists()]

    print("Files beside the DTM:")
    for p in sorted(ctx.raw.glob("DTEEC_023524_1985_023379_1985_U01*")):
        print(f"  {p.name} ({p.stat().st_size:,} bytes)")

    if found_labels:
        for label_path in found_labels:
            print(f"\n--- Relevant lines from {label_path.name} ---")
            text = label_path.read_text(encoding="utf-8", errors="replace")
            keywords = (
                "UNIT", "ELEVATION", "VERTICAL", "DATUM", "RADIUS",
                "SCALE", "OFFSET", "PIXEL", "PROJECTION", "COORDINATE",
            )
            lines = [
                line.strip()
                for line in text.splitlines()
                if any(word in line.upper() for word in keywords)
            ]
            print("\n".join(lines[:100]) or "No matching metadata lines found.")
    else:
        print("\nNo same-name label found in Raw data.")

        # Show metadata embedded in the raster, if present.
        with rasterio.open(dtm_path) as src:
            print("\nRaster metadata domains:", src.tag_namespaces())
            print("\nDataset tags:")
            print(src.tags())
            print("\nBand tags:")
            print(src.tags(1))


# ---- Cell 23: dump raster metadata namespaces ---------------------------------------------------------

def find_relevant(obj, path=""):
    """Recursively print metadata entries that may describe elevations or map geometry."""
    terms = ("unit", "elev", "vertical", "datum", "radius",
             "scale", "offset", "pixel", "projection", "map")
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if any(term in key.lower() for term in terms):
                print(f"{new_path}: {value}")
            find_relevant(value, new_path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            find_relevant(value, f"{path}[{index}]")


def step_hirise_metadata(ctx: Context) -> None:
    """Print every metadata namespace of the HiRISE DTM."""
    with rasterio.open(ctx.hirise_dtm) as src:
        namespaces = src.tag_namespaces()
        print("Metadata namespaces:", namespaces)

        for namespace in namespaces:
            print(f"\n--- {namespace} ---")
            tags = src.tags(ns=namespace)
            print(json.dumps(tags, indent=2, default=str)[:30000])
            find_relevant(tags)


# ---- Cell 24: slope + hillshade from the aligned HiRISE DTM ----------------------------------------------

def step_hirise_derivatives(ctx: Context) -> None:
    """Slope (degrees) and hillshade GeoTIFFs from the aligned 5 m HiRISE DTM."""
    input_path = ctx.hirise_aligned
    slope_path = ctx.outputs / "HiRISE_Jezero_slope_5m.tif"
    hillshade_path = ctx.outputs / "HiRISE_Jezero_hillshade_5m.tif"

    if not input_path.exists():
        raise FileNotFoundError(
            f"Aligned HiRISE DTM not found: {input_path}. Run the 'hirise-align' step first."
        )

    with rasterio.open(input_path) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        profile = src.profile.copy()
        pixel_y, pixel_x = abs(src.res[1]), abs(src.res[0])

    # Compute terrain gradient and slope in degrees.
    dz_dy, dz_dx = np.gradient(dem, pixel_y, pixel_x)
    slope_degrees = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    # Compute a conventional hillshade for visualization only.
    slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    azimuth_rad = np.radians(315.0)
    altitude_rad = np.radians(45.0)

    illumination = (
        np.sin(altitude_rad) * np.cos(slope_rad)
        + np.cos(altitude_rad) * np.sin(slope_rad)
        * np.cos(azimuth_rad - aspect)
    )
    hillshade = (np.clip(illumination, 0, 1) * 255).astype(np.float32)

    # Keep nodata areas transparent/invalid in both derivatives.
    invalid = ~np.isfinite(dem)
    slope_degrees[invalid] = np.nan
    hillshade[invalid] = np.nan

    profile.update(
        driver="GTiff", dtype="float32", count=1,
        nodata=np.nan, compress="deflate", predictor=3,
    )

    for path, array, description in [
        (slope_path, slope_degrees, "Derived slope in degrees"),
        (hillshade_path, hillshade, "Derived hillshade for visualization"),
    ]:
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(array, 1)
            dst.set_band_description(1, description)
            dst.update_tags(
                source=input_path.name,
                processing="Derived from 5 m reprojected HiRISE DTM",
                note="Visualization/terrain derivative, not a validated safety assessment",
            )
        print(f"Saved: {path}")

    valid_slope = slope_degrees[np.isfinite(slope_degrees)]
    print(f"Slope range: {valid_slope.min():.2f}° to {valid_slope.max():.2f}°")


# ---- Cell 27: MEDA sol-1 plot vs. local time ------------------------------------------------------------------

def add_local_time_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """Extract Mars local-time minutes from strings like '0001 15:28:01'."""
    if "LTST" not in df.columns:
        raise ValueError("Expected an LTST column.")

    def to_minutes(value):
        match = re.search(r"(\d{2}):(\d{2}):(\d{2})", str(value))
        if not match:
            return float("nan")
        hour, minute, second = map(int, match.groups())
        return hour * 60 + minute + second / 60

    result = df.copy()
    result["LOCAL_TIME_MIN"] = result["LTST"].map(to_minutes)
    return result.sort_values("LOCAL_TIME_MIN")


def check_meda_files(ctx: Context, keys=("PS", "TIRS", "RHS")) -> dict:
    files = {key: ctx.meda_path(key) for key in keys}
    for key, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(f"MEDA {key} CSV not found: {path}")
    return files


def step_meda_plot(ctx: Context) -> None:
    """MEDA sol-1 sample plotted against Mars local time (LTST)."""
    ctx.outputs.mkdir(parents=True, exist_ok=True)
    files = check_meda_files(ctx)

    ps = add_local_time_minutes(pd.read_csv(files["PS"]))
    tirs = add_local_time_minutes(pd.read_csv(files["TIRS"]))
    rhs = add_local_time_minutes(pd.read_csv(files["RHS"]))

    print("Coverage:")
    for name, df in [("Pressure", ps), ("TIRS", tirs), ("Humidity", rhs)]:
        print(
            f"{name}: {len(df):,} rows; "
            f"LTST {df['LTST'].iloc[0]} to {df['LTST'].iloc[-1]}"
        )

    rh_valid = rhs["LOCAL_RELATIVE_HUMIDITY"].notna().sum()
    print(f"\nRelative-humidity values present: {rh_valid:,} / {len(rhs):,}")
    print("Humidity product columns:", list(rhs.columns))

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    # Pressure, separated by the reported transducer identifier.
    for transducer, group in ps.groupby("TRANSDUCER", dropna=False):
        axes[0].plot(
            group["LOCAL_TIME_MIN"], group["PRESSURE"],
            linewidth=1, label=f"Transducer {transducer}",
        )
    axes[0].set_title("MEDA pressure — sol 1 sample")
    axes[0].set_ylabel("Reported pressure value\n(units to verify)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(tirs["LOCAL_TIME_MIN"], tirs["DOWNWARD_LW_IRRADIANCE"],
                 label="Downward longwave irradiance")
    axes[1].plot(tirs["LOCAL_TIME_MIN"], tirs["UPWARD_LW_IRRADIANCE"],
                 label="Upward longwave irradiance")
    axes[1].set_title("MEDA TIRS — sol 1 sample")
    axes[1].set_ylabel("Reported irradiance value\n(units to verify)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    axes[2].plot(rhs["LOCAL_TIME_MIN"], rhs["HUMIDITY_LOCAL_TEMP"],
                 color="tab:orange", label="Humidity-local temperature")
    axes[2].set_title(
        "MEDA humidity product — temperature field "
        "(relative-humidity values may be missing)"
    )
    axes[2].set_ylabel("Reported temperature value\n(units to verify)")
    axes[2].set_xlabel("Mars local time (LTST)")
    axes[2].legend()
    axes[2].grid(alpha=0.25)

    # Label x-axis in HH:MM using the sol-local-time minute values.
    tick_positions = sorted(
        set(
            int(round(value / 10) * 10)
            for value in pd.concat(
                [ps["LOCAL_TIME_MIN"], tirs["LOCAL_TIME_MIN"], rhs["LOCAL_TIME_MIN"]]
            ).dropna()
        )
    )
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(
        [f"{minute // 60:02d}:{minute % 60:02d}" for minute in tick_positions],
        rotation=30, ha="right",
    )

    fig.suptitle("Perseverance MEDA sample — sol 1, historical data", fontsize=14)
    fig.tight_layout()

    plot_path = ctx.outputs / "MEDA_sol1_sample.png"
    fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    finish_figure(ctx, fig)

    print(f"\nSaved plot: {plot_path}")


# ---- Cell 28: merge the three MEDA products by SCLK -----------------------------------------------------------------

def step_meda_merge(ctx: Context) -> None:
    """Outer-join the PS, TIRS and RHS products on spacecraft clock (SCLK)."""
    ctx.outputs.mkdir(parents=True, exist_ok=True)
    files = check_meda_files(ctx)

    def load_product(product, path):
        df = pd.read_csv(path)

        if "SCLK" not in df.columns:
            raise ValueError(f"{path.name} has no SCLK column.")

        # Keep one row per clock tick if duplicates exist; report them for review.
        duplicate_count = int(df["SCLK"].duplicated().sum())
        if duplicate_count:
            print(f"Warning: {product} has {duplicate_count} duplicate SCLK rows; keeping the first.")
            df = df.drop_duplicates("SCLK", keep="first")

        # Keep each product's local times so we can detect any clock/time mismatches.
        rename = {}
        if "LTST" in df.columns:
            rename["LTST"] = f"LTST_{product}"
        if "LMST" in df.columns:
            rename["LMST"] = f"LMST_{product}"

        return df.rename(columns=rename)

    ps = load_product("PS", files["PS"])
    tirs = load_product("TIRS", files["TIRS"])
    rhs = load_product("RHS", files["RHS"])

    # Outer joins retain all samples, including times when only one instrument has data.
    merged = ps.merge(tirs, on="SCLK", how="outer", validate="one_to_one")
    merged = merged.merge(rhs, on="SCLK", how="outer", validate="one_to_one")
    merged = merged.sort_values("SCLK").reset_index(drop=True)

    # Convenient display time: use whichever product has a timestamp at that SCLK.
    ltst_cols = [c for c in ("LTST_PS", "LTST_TIRS", "LTST_RHS") if c in merged.columns]
    if ltst_cols:
        merged["LTST"] = merged[ltst_cols].bfill(axis=1).iloc[:, 0]

    # Check whether products report differing local-time strings for shared SCLK values.
    if len(ltst_cols) > 1:
        merged["LTST_MISMATCH"] = merged[ltst_cols].nunique(axis=1, dropna=True) > 1
        mismatch_count = int(merged["LTST_MISMATCH"].sum())
    else:
        mismatch_count = 0

    output_path = ctx.meda_merged
    merged.to_csv(output_path, index=False)

    print(f"Saved merged table: {output_path}")
    print(f"Rows after outer join: {len(merged):,}")
    print(f"SCLK range: {merged['SCLK'].min()} to {merged['SCLK'].max()}")
    print(f"Rows with differing LTST strings across products: {mismatch_count}")

    print("\nPer-product availability in merged table:")
    for product, columns in {
        "PS": ["PRESSURE"],
        "TIRS": ["DOWNWARD_LW_IRRADIANCE", "UPWARD_LW_IRRADIANCE"],
        "RHS": ["HUMIDITY_LOCAL_TEMP", "LOCAL_RELATIVE_HUMIDITY"],
    }.items():
        present_columns = [c for c in columns if c in merged.columns]
        if present_columns:
            print(f"{product}:")
            for column in present_columns:
                print(f"  {column}: {merged[column].notna().sum():,} valid rows")

    print("\nPreview:")
    print(merged.head(10).to_string())


def load_merged_meda(ctx: Context) -> pd.DataFrame:
    if not ctx.meda_merged.exists():
        raise FileNotFoundError(
            f"Merged MEDA file not found: {ctx.meda_merged}. Run the 'meda-merge' step first."
        )
    df = pd.read_csv(ctx.meda_merged).sort_values("SCLK")
    df["ELAPSED_SECONDS"] = df["SCLK"] - df["SCLK"].min()
    return df


# ---- Cell 29: merged MEDA plot, units still unverified -----------------------------------------------------------------

def step_meda_plot_raw_units(ctx: Context) -> None:
    """Merged MEDA sol-1 figure (axes labelled 'units unverified') + text summary."""
    df = load_merged_meda(ctx)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Pressure: plot separate points per transducer; do not connect across gaps.
    for sensor, group in df.dropna(subset=["PRESSURE"]).groupby("TRANSDUCER"):
        axes[0].scatter(
            group["ELAPSED_SECONDS"], group["PRESSURE"],
            s=9, alpha=0.65, label=f"Transducer {sensor:g}",
        )
    axes[0].set_title("MEDA pressure")
    axes[0].set_ylabel("Reported value\n(units unverified)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Plot available TIRS samples only.
    for column, label in [
        ("DOWNWARD_LW_IRRADIANCE", "Downward"),
        ("UPWARD_LW_IRRADIANCE", "Upward"),
    ]:
        valid = df.dropna(subset=[column])
        axes[1].plot(
            valid["ELAPSED_SECONDS"], valid[column],
            marker="o", markersize=3, linewidth=1, label=label,
        )
    axes[1].set_title("MEDA thermal infrared (TIRS)")
    axes[1].set_ylabel("Reported value\n(units unverified)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # Relative humidity is missing in this file; show the available sensor temperature.
    valid_temp = df.dropna(subset=["HUMIDITY_LOCAL_TEMP"])
    axes[2].plot(
        valid_temp["ELAPSED_SECONDS"], valid_temp["HUMIDITY_LOCAL_TEMP"],
        marker=".", markersize=3, linewidth=1, color="tab:orange",
    )
    axes[2].set_title("MEDA humidity-sensor temperature")
    axes[2].set_ylabel("Reported value\n(units unverified)")
    axes[2].set_xlabel("Elapsed spacecraft-clock seconds")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Perseverance MEDA sample — sol 1 (historical data)", fontsize=14)
    fig.tight_layout()

    png_path = ctx.outputs / "MEDA_sol1_presentation.png"
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    finish_figure(ctx, fig)

    summary = f"""MEDA sample summary
Sol: 1 (from the LTST strings in the supplied files)
SCLK range: {df['SCLK'].min()} to {df['SCLK'].max()}
Pressure valid rows: {df['PRESSURE'].notna().sum()}
TIRS downward valid rows: {df['DOWNWARD_LW_IRRADIANCE'].notna().sum()}
TIRS upward valid rows: {df['UPWARD_LW_IRRADIANCE'].notna().sum()}
Humidity-sensor temperature valid rows: {df['HUMIDITY_LOCAL_TEMP'].notna().sum()}
Relative-humidity valid rows: {df['LOCAL_RELATIVE_HUMIDITY'].notna().sum()}

Important: verify measurement units and quality flags from the PDS product labels
before interpreting values. This sol-1 sample is not contemporaneous with the
sol-836 example image and is not a forecast or safety assessment.
"""
    summary_path = ctx.outputs / "MEDA_sol1_summary.txt"
    summary_path.write_text(summary)

    print(f"Saved figure: {png_path}")
    print(f"Saved summary: {summary_path}")
    print(summary)


# ---- Cell 30: check column units in the MEDA SIS PDF ---------------------------------------------------------------------

def step_meda_sis_pdf(ctx: Context) -> None:
    """Download the MEDA Environmental RDR SIS and print pages that define the columns."""
    import fitz  # PyMuPDF
    import requests

    pdf_path = ctx.outputs / "meda_env_rdr_sis.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        response = requests.get(MEDA_SIS_URL, timeout=60)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

    doc = fitz.open(pdf_path)
    print(f"Opened {pdf_path.name}: {len(doc)} pages")

    terms = [
        "PRESSURE", "DOWNWARD_LW_IRRADIANCE", "UPWARD_LW_IRRADIANCE",
        "HUMIDITY_LOCAL_TEMP", "LOCAL_RELATIVE_HUMIDITY",
        "VOLUME_MIXING_RATIO", "UNCERTAINTY", "TRANSDUCER",
    ]

    matches = []
    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")
        upper_text = text.upper()
        found = [term for term in terms if term in upper_text]
        if found:
            matches.append((page_number, found, text))

    if not matches:
        print(
            "No exact column names found in the extracted PDF text. "
            "Try searching the PDF manually for 'pressure', 'irradiance', "
            "'humidity', or 'CSV'."
        )
    else:
        for page_number, found, text in matches:
            print(f"\n--- PDF page {page_number}; matched: {', '.join(found)} ---")
            print(text[:7000])


# ---- Cell 31: merged MEDA plot with verified units -------------------------------------------------------------------------

def step_meda_plot_verified(ctx: Context) -> None:
    """Merged MEDA sol-1 figure with units confirmed from the SIS (Pa, W/m², °C)."""
    df = load_merged_meda(ctx)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    # Pressure in pascals; show each transducer separately.
    for sensor, group in df.dropna(subset=["PRESSURE"]).groupby("TRANSDUCER"):
        axes[0].scatter(
            group["ELAPSED_SECONDS"], group["PRESSURE"],
            s=8, alpha=0.65, label=f"Transducer {sensor:g}",
        )
    axes[0].set_title("MEDA pressure")
    axes[0].set_ylabel("Pressure (Pa)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Calibrated longwave irradiance in W/m².
    for column, label in [
        ("DOWNWARD_LW_IRRADIANCE", "Downward longwave"),
        ("UPWARD_LW_IRRADIANCE", "Upward longwave"),
    ]:
        valid = df.dropna(subset=[column])
        axes[1].plot(
            valid["ELAPSED_SECONDS"], valid[column],
            marker="o", markersize=3, linewidth=1, label=label,
        )
    axes[1].set_title("MEDA thermal infrared (TIRS)")
    axes[1].set_ylabel("Irradiance (W/m²)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    # Plot available humidity-sensor temperature; convert K to °C for readability.
    valid_temp = df.dropna(subset=["HUMIDITY_LOCAL_TEMP"]).copy()
    valid_temp["TEMP_C"] = valid_temp["HUMIDITY_LOCAL_TEMP"] - 273.15
    axes[2].plot(
        valid_temp["ELAPSED_SECONDS"], valid_temp["TEMP_C"],
        marker=".", markersize=3, linewidth=1, color="tab:orange",
    )
    axes[2].set_title("MEDA humidity-sensor temperature (RH unavailable)")
    axes[2].set_ylabel("Temperature (°C)")
    axes[2].set_xlabel("Elapsed spacecraft-clock seconds")
    axes[2].grid(alpha=0.25)

    fig.suptitle("Perseverance MEDA sample — sol 1 (historical data)", fontsize=14)
    fig.text(
        0.5, 0.005,
        "Source: MEDA Environmental RDR SIS, Appendix A. "
        "Relative-humidity field has no valid values in this sample.",
        ha="center", fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    png_path = ctx.meda_verified_png
    fig.savefig(png_path, dpi=180, bbox_inches="tight")
    finish_figure(ctx, fig)

    print(f"Saved verified-unit figure: {png_path}")
    print("Verified from the MEDA Environmental RDR SIS:")
    print("- PRESSURE: Pa")
    print("- DOWNWARD_LW_IRRADIANCE / UPWARD_LW_IRRADIANCE: W/m²")
    print("- HUMIDITY_LOCAL_TEMP: K in the source; plotted here as °C")
    print(f"- LOCAL_RELATIVE_HUMIDITY valid rows: {df['LOCAL_RELATIVE_HUMIDITY'].notna().sum()}")


# ---- Cell 32: add the MEDA panel + single layer control, save the final EVA map ---------------------------------------------

def step_meda_panel(ctx: Context) -> None:
    """Add the MEDA chart panel and one LayerControl; save the final EVA map."""
    import folium
    from folium import Element

    meda_png = ctx.meda_verified_png
    if not meda_png.is_file():
        raise FileNotFoundError(
            f"MEDA plot not found: {meda_png}. Run the 'meda-plot-verified' step first."
        )

    m = get_map(ctx)

    # Embed the MEDA chart in a small panel. Clicks outside the panel still reach the map.
    meda_data = base64.b64encode(meda_png.read_bytes()).decode("ascii")
    m.get_root().html.add_child(Element(f"""
<div style="position:fixed; top:12px; left:12px; z-index:9998;
            pointer-events:none;">
  <div id="marswalk-meda-panel"
       style="width:min(330px,32vw); max-height:34vh; overflow:auto;
              pointer-events:auto; background:white;
              border:1px solid #536273; border-radius:8px;
              padding:8px; box-shadow:0 2px 8px #0005;
              font:12px/1.35 Arial,sans-serif;">
    <b>Perseverance MEDA · Sol 1 historical sample</b>
    <img src="data:image/png;base64,{meda_data}"
         alt="MEDA sol 1 pressure, TIRS irradiance, and humidity-sensor temperature plots"
         style="display:block; width:100%; height:auto; margin:5px 0;">
    <div>Relative humidity is unavailable in this sample. These historical
         measurements are not contemporaneous with the sol-836 image and
         are not a forecast.</div>
  </div>
</div>
"""))

    # Add exactly one layer control after the map layers are already present.
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    control_count = sum(isinstance(c, folium.LayerControl) for c in m._children.values())
    assert control_count == 1, f"Expected one layer control; found {control_count}"

    final_html = ctx.outputs / "marswalk_eva_map_with_meda.html"
    m.save(str(final_html))

    print(f"Layer controls on map: {control_count}")
    print(f"Saved map with MEDA: {final_html} ({final_html.stat().st_size:,} bytes)")


# --------------------------------------------------------------------------
# Step registry and command line
# --------------------------------------------------------------------------

# (name, function, runs in the default sequence?, one-line description)
STEPS = [
    ("setup", step_setup, True, "check/create project folders                       [cell 0]"),
    ("registry", step_registry, True, "write source registry + input inventory             [cell 1]"),
    ("dem-inspect", step_dem_inspect, True, "inspect the CTX DEM, save metadata + preview        [cell 2]"),
    ("crop", step_crop, True, "cut the 3 km test crop                            [cell 3]"),
    ("slope", step_slope, True, "provisional slope of the crop                     [cell 4]"),
    ("static-demo", step_static_demo, True, "static elevation/slope map with example route      [cell 5]"),
    ("plan-route", step_plan_route, True, "route planner PNG (was the ipywidgets UI)          [cell 6]"),
    ("leaflet-route", step_leaflet_route, True, "Leaflet map of the crop with a route               [cell 7]"),
    ("package-demo", step_package_demo, True, "standalone demo HTML + README                     [cell 8]"),
    ("gradio", step_gradio, False, "launch the Gradio planner app (blocks; opt-in)     [cell 9]"),
    ("overview-map", step_overview_map, True, "Leaflet overview of the full DEM                  [cell 10]"),
    ("download-usgs", step_download_usgs, True, "download + extract USGS geology archive            [cell 11]"),
    ("inspect-gdb", step_inspect_gdb, True, "list Nili.gdb layers                             [cell 12]"),
    ("geology", step_geology, True, "rasterize geology onto the DEM grid                [cell 13]"),
    ("corners", step_corners, True, "DEM corner lon/lat                               [cell 15]"),
    ("final-map", step_final_map, True, "final terrain explorer map                       [cell 17]"),
    ("nasa-image", step_nasa_image, True, "download NASA PIA25917 image + metadata           [cell 18]"),
    ("photo-card", step_photo_card, True, "add the photo card to the map                    [cell 19]"),
    ("hirise-inspect", step_hirise_inspect, True, "find/inspect HiRISE DTM + MEDA CSVs               [cell 20]"),
    ("hirise-align", step_hirise_align, True, "reproject HiRISE DTM to CTX CRS (5 m)             [cell 21]"),
    ("hirise-labels", step_hirise_labels, True, "HiRISE label / embedded metadata lines            [cell 22]"),
    ("hirise-metadata", step_hirise_metadata, True, "dump HiRISE metadata namespaces                  [cell 23]"),
    ("hirise-derivatives", step_hirise_derivatives, True, "HiRISE slope + hillshade GeoTIFFs                [cell 24]"),
    ("meda-plot", step_meda_plot, True, "MEDA sol-1 plot vs. local time                   [cell 27]"),
    ("meda-merge", step_meda_merge, True, "merge MEDA products by SCLK                      [cell 28]"),
    ("meda-plot-raw-units", step_meda_plot_raw_units, True, "merged MEDA plot, units unverified + summary      [cell 29]"),
    ("meda-sis-pdf", step_meda_sis_pdf, True, "search the MEDA SIS PDF for column definitions    [cell 30]"),
    ("meda-plot-verified", step_meda_plot_verified, True, "merged MEDA plot with verified units               [cell 31]"),
    ("meda-panel", step_meda_panel, True, "add MEDA panel + layer control, save EVA map     [cell 32]"),
]
STEP_FUNCS = {name: func for name, func, _, _ in STEPS}
DEFAULT_STEPS = [name for name, _, default, _ in STEPS if default]

# Import name -> pip package name, for friendlier "missing dependency" messages.
PIP_NAMES = {"fitz": "pymupdf", "PIL": "pillow", "osgeo": "gdal"}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Marswalk EVA map — Jezero Crater prototype (converted from a Colab notebook).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list-steps to see the available steps.",
    )
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(os.environ.get("MARSWALK_ROOT", "Marswalk")),
        help="Marswalk project folder (default: $MARSWALK_ROOT or ./Marswalk)",
    )
    parser.add_argument(
        "--steps", default="all",
        help="comma-separated step names, or 'all' for the default sequence (default: all)",
    )
    parser.add_argument("--list-steps", action="store_true", help="list steps and exit")
    parser.add_argument("--create-dirs", action="store_true",
                        help="create the project folders if they are missing")
    parser.add_argument("--show", action="store_true",
                        help="display matplotlib figures in a window as well as saving them")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="keep running later steps if one step fails")

    route = parser.add_argument_group("route planner (step: plan-route)")
    route.add_argument("--start-row", type=int, default=8)
    route.add_argument("--start-col", type=int, default=8)
    route.add_argument("--end-row", type=int, default=None, help="default: last row - 8")
    route.add_argument("--end-col", type=int, default=None, help="default: last column - 8")
    route.add_argument("--slope-cutoff", type=float, default=10.0,
                       help="illustrative slope cutoff in degrees (default: 10)")
    route.add_argument("--layer", choices=["Elevation", "Slope"], default="Elevation",
                       help="map layer for the planner PNG (default: Elevation)")

    gradio = parser.add_argument_group("Gradio app (step: gradio)")
    gradio.add_argument("--share", action="store_true",
                        help="create a public gradio.live link")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.list_steps:
        width = max(len(name) for name in STEP_FUNCS)
        for name, _, default, description in STEPS:
            flag = " " if default else "*"
            print(f"{flag} {name:<{width}}  {' '.join(description.split())}")
        print("\n* = not in the default sequence; run it explicitly with --steps")
        return 0

    if args.steps.strip().lower() == "all":
        selected = list(DEFAULT_STEPS)
    else:
        selected = [s.strip() for s in args.steps.split(",") if s.strip()]
        unknown = [s for s in selected if s not in STEP_FUNCS]
        if unknown:
            print(f"Unknown step(s): {', '.join(unknown)}. Use --list-steps.", file=sys.stderr)
            return 2

    if not args.show:
        matplotlib.use("Agg")  # no GUI needed when figures are only saved

    ctx = Context(root=args.project_root.expanduser().resolve(), args=args)

    # Folders must exist before any other step runs (cell 0 of the notebook).
    if "setup" not in selected:
        selected.insert(0, "setup")

    failures = []
    for name in selected:
        print(f"\n{'=' * 70}\nSTEP: {name}\n{'=' * 70}")
        try:
            STEP_FUNCS[name](ctx)
        except ModuleNotFoundError as exc:
            package = PIP_NAMES.get(exc.name, exc.name)
            print(f"Missing dependency '{exc.name}'. Install it with: pip install {package}",
                  file=sys.stderr)
            failures.append(name)
            if not args.continue_on_error:
                break
        except Exception as exc:  # noqa: BLE001 - report and (optionally) continue
            print(f"\nStep '{name}' failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            if args.continue_on_error:
                traceback.print_exc()
            failures.append(name)
            if not args.continue_on_error:
                break

    print("\n" + "-" * 70)
    if failures:
        print(f"Finished with failures in: {', '.join(failures)}")
        return 1
    print(f"Finished {len(selected)} step(s) without errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
