#!/usr/bin/env python3
"""High-resolution per-tree validation sheets.

Renders one page per tree so a measurement can be checked against the
points it was derived from, rather than trusted from a CSV:

  diameter sections  the real points in each horizontal slice, with the
                     fitted circle drawn over them -- the slice is
                     rebuilt with the same |hag - h| <= thickness/2 rule
                     compute_taper used, so what is drawn is what was
                     measured, not an approximation of it
  taper profile      measured vs monotonic-corrected diameter against
                     height, with DBH and the crown base marked
  crown mesh         the exterior surface of the occupied voxel set whose
                     count *is* crown_volume_m3 -- only faces with no
                     occupied neighbour are drawn, so the crown envelope
                     is visible instead of a solid block

The DTM is rebuilt here rather than approximated from z_base, because
every height on the page is a height above ground and an approximation
would silently shift the sections away from the ones that were measured.

Example:
  python tupisat_inference/forest_metrics/tree_report.py \
      --input-las data/05-PWOOD/plot_pwood.laz \
      --metrics-dir data/06-METRICS-WOOD --stem P01_wood \
      --tree-ids 33 28 2 15 --output-dir data/07-RELATORIO --lang en
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tupisat_inference.las_to_pandas import las_to_pandas
from tupisat_inference.forest_metrics.config import ForestMetricsConfig
from tupisat_inference.forest_metrics.dtm import build_dtm

SECTION_ROWS, SECTION_COLS = 3, 4
N_SECTIONS = SECTION_ROWS * SECTION_COLS

INK = "#1a1a1a"
MUTED = "#8a8a8a"
POINTS = "#4a6fa5"
FIT = "#d1495b"
CORRECTED = "#3f8f5f"
CROWN = "#6a8f3f"


# Report language. Every user-facing string on the page lives here, so a
# new language is a dict entry rather than a pass through the drawing code.
STRINGS = {
    "en": {
        "tree": "Tree",
        "height": "height", "dbh": "DBH", "crown_base": "crown base",
        "stem_volume": "stem volume", "crown_volume": "crown volume",
        "live_crown_ratio": "live crown ratio",
        "sections_caption": ("Diameter sections \u2014 the actual points in each slice, the fitted "
                              "circle (red) and, where they differ, the corrected diameter (dashed green)"),
        "dbh_tag": "DBH", "diameter": "diameter (cm)",
        "height_axis": "height above ground (m)",
        "taper_title": "Taper profile", "measured": "measured", "corrected": "corrected",
        "no_fit": "no fit", "crown_base_label": "crown base",
        "assortments": "Assortments", "sawlog": "sawlog volume", "pulpwood": "pulpwood volume",
        "merchantable": "merchantable volume", "logs": "logs",
        "stem_volume_hdr": "Stem volume", "by_taper": "by taper integration",
        "conic": "conic (reference)",
        "quality_hdr": "Stem quality", "sections_measured": "sections measured",
        "median_cci": "median coverage (CCI)", "refitted": "axis-refitted sections",
        "grade": "overall", "flags": "flags", "none": "none",
        "tortuosity_hdr": "Stem tortuosity", "tortuosity": "tortuosity (path/chord)",
        "max_sweep": "maximum sweep", "lean": "lean",
        "taper_hdr": "Taper", "taper_rate": "taper rate", "form_quotient": "form quotient (d50%/DBH)",
        "butt_swell": "butt swell (d0.1/DBH)",
        "grades": {"excellent": "excellent", "good": "good", "fair": "fair", "poor": "poor"},
        "tree_full": "Whole tree \u2014 crown above the stem",
        "crown_only": "Crown alone \u2014 mesh of the measured volume",
        "voxels_of": "voxels of", "x_axis": "x (m)", "y_axis": "y (m)", "z_axis": "z (m)",
        "height_short": "height (m)",
    },
    "pt": {
        "tree": "\u00c1rvore",
        "height": "altura", "dbh": "DAP", "crown_base": "base de copa",
        "stem_volume": "volume de fuste", "crown_volume": "volume de copa",
        "live_crown_ratio": "raz\u00e3o de copa viva",
        "sections_caption": ("Se\u00e7\u00f5es de di\u00e2metro \u2014 pontos reais da fatia, c\u00edrculo ajustado "
                              "(vermelho) e, quando diferem, o di\u00e2metro corrigido (verde tracejado)"),
        "dbh_tag": "DAP", "diameter": "di\u00e2metro (cm)",
        "height_axis": "altura acima do solo (m)",
        "taper_title": "Perfil de afilamento", "measured": "medido", "corrected": "corrigido",
        "no_fit": "sem ajuste", "crown_base_label": "base de copa",
        "assortments": "Sortimentos", "sawlog": "volume serraria", "pulpwood": "volume celulose",
        "merchantable": "volume comercial", "logs": "toras",
        "stem_volume_hdr": "Volume de fuste", "by_taper": "por afilamento",
        "conic": "c\u00f4nico (refer\u00eancia)",
        "quality_hdr": "Qualidade da \u00e1rvore", "sections_measured": "se\u00e7\u00f5es medidas",
        "median_cci": "cobertura mediana (CCI)", "refitted": "se\u00e7\u00f5es reajustadas",
        "grade": "classifica\u00e7\u00e3o", "flags": "sinaliza\u00e7\u00f5es", "none": "nenhuma",
        "tortuosity_hdr": "Tortuosidade do tronco", "tortuosity": "tortuosidade (caminho/corda)",
        "max_sweep": "desvio m\u00e1ximo", "lean": "inclina\u00e7\u00e3o",
        "taper_hdr": "Conicidade", "taper_rate": "afilamento", "form_quotient": "quociente de forma (d50%/DAP)",
        "butt_swell": "alargamento da base (d0,1/DAP)",
        "grades": {"excellent": "excelente", "good": "boa", "fair": "regular", "poor": "ruim"},
        "tree_full": "\u00c1rvore completa \u2014 copa sobre o fuste",
        "crown_only": "Copa isolada \u2014 malha do volume medido",
        "voxels_of": "voxels de", "x_axis": "x (m)", "y_axis": "y (m)", "z_axis": "z (m)",
        "height_short": "altura (m)",
    },
    "es": {
        "tree": "\u00c1rbol",
        "height": "altura", "dbh": "DAP", "crown_base": "base de copa",
        "stem_volume": "volumen de fuste", "crown_volume": "volumen de copa",
        "live_crown_ratio": "raz\u00f3n de copa viva",
        "sections_caption": ("Secciones de di\u00e1metro \u2014 puntos reales de la rodaja, c\u00edrculo ajustado "
                              "(rojo) y, cuando difieren, el di\u00e1metro corregido (verde discontinuo)"),
        "dbh_tag": "DAP", "diameter": "di\u00e1metro (cm)",
        "height_axis": "altura sobre el suelo (m)",
        "taper_title": "Perfil de ahusamiento", "measured": "medido", "corrected": "corregido",
        "no_fit": "sin ajuste", "crown_base_label": "base de copa",
        "assortments": "Surtidos", "sawlog": "volumen aserr\u00edo", "pulpwood": "volumen pulpa",
        "merchantable": "volumen comercial", "logs": "trozas",
        "stem_volume_hdr": "Volumen de fuste", "by_taper": "por ahusamiento",
        "conic": "c\u00f3nico (referencia)",
        "quality_hdr": "Calidad del \u00e1rbol", "sections_measured": "secciones medidas",
        "median_cci": "cobertura mediana (CCI)", "refitted": "secciones reajustadas",
        "grade": "clasificaci\u00f3n", "flags": "se\u00f1alizaciones", "none": "ninguna",
        "tortuosity_hdr": "Tortuosidad del fuste", "tortuosity": "tortuosidad (camino/cuerda)",
        "max_sweep": "desviaci\u00f3n m\u00e1xima", "lean": "inclinaci\u00f3n",
        "taper_hdr": "Ahusamiento", "taper_rate": "tasa de ahusamiento", "form_quotient": "cociente de forma (d50%/DAP)",
        "butt_swell": "ensanchamiento basal (d0,1/DAP)",
        "grades": {"excellent": "excelente", "good": "buena", "fair": "regular", "poor": "mala"},
        "tree_full": "\u00c1rbol completo \u2014 copa sobre el fuste",
        "crown_only": "Copa aislada \u2014 malla del volumen medido",
        "voxels_of": "voxels de", "x_axis": "x (m)", "y_axis": "y (m)", "z_axis": "z (m)",
        "height_short": "altura (m)",
    },
}


def stem_quality_metrics(taper_df, tree, cfg):
    """The three things a forester grades a stem by, all derived from the
    fitted sections rather than from internal flags.

    tortuosity  path length through the section centres divided by the
                straight chord between the first and last of them. 1.0 is a
                perfectly straight stem; a swept or forked one exceeds it.
                Reported with the largest perpendicular departure from that
                chord, which is what actually costs sawlog recovery.
    taper rate  robust (Theil-Sen) slope of diameter against height above
                breast height, in cm per metre. Least squares would be
                dragged by any section the fitter still got wrong.
    form quotient  diameter at half the measured stem length over DBH. A
                pure shape descriptor: no volume, no length, so it stays
                comparable between trees of different size. A classic form
                *factor* (volume over the DBH cylinder) was tried first and
                rejected -- with volume covering only the merchantable stem,
                butt swell over a short bole pushes it above 1.0, which
                reads as a bug even though it is arithmetically correct.
    """
    from scipy.stats import theilslopes

    out = {"n_measured": 0, "n_total": int(len(taper_df)), "frac_measured": np.nan,
           "median_cci": np.nan, "n_refitted": 0, "tortuosity": np.nan,
           "max_sweep_cm": np.nan, "taper_rate_cm_per_m": np.nan,
           "form_quotient": np.nan, "butt_swell": np.nan, "grade": "poor"}
    if taper_df.empty:
        return out

    measured = taper_df[taper_df["diameter_cm"].notna()].sort_values("height_m")
    out["n_measured"] = int(len(measured))
    out["frac_measured"] = out["n_measured"] / out["n_total"] if out["n_total"] else np.nan
    if "fit_source" in taper_df:
        out["n_refitted"] = int(taper_df["fit_source"].isin(("axis", "cylinder")).sum())
    if len(measured):
        out["median_cci"] = float(measured["cci"].median())

    cx = measured["center_x"].to_numpy(dtype=float) if len(measured) else np.array([])
    cy = measured["center_y"].to_numpy(dtype=float) if len(measured) else np.array([])
    hh = measured["height_m"].to_numpy(dtype=float) if len(measured) else np.array([])
    ok = np.isfinite(cx) & np.isfinite(cy)
    if int(ok.sum()) >= 3:
        pts = np.column_stack([cx[ok], cy[ok], hh[ok]])
        path = float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
        chord_vec = pts[-1] - pts[0]
        chord = float(np.linalg.norm(chord_vec))
        if chord > 0:
            out["tortuosity"] = path / chord
            unit = chord_vec / chord
            rel = pts - pts[0]
            perp = rel - np.outer(rel @ unit, unit)
            out["max_sweep_cm"] = float(np.max(np.linalg.norm(perp, axis=1)) * 100)

    above = measured[measured["height_m"] >= cfg.dbh_height_m]
    if len(above) >= 4 and np.ptp(above["height_m"].to_numpy()) > 0:
        slope = theilslopes(above["diameter_cm"].to_numpy(dtype=float),
                             above["height_m"].to_numpy(dtype=float))
        out["taper_rate_cm_per_m"] = float(-slope[0])

    dbh = float(tree.get("dbh_cm", np.nan))
    if len(measured) and np.isfinite(dbh) and dbh > 0:
        length = float(measured["height_m"].max())
        mid = measured.iloc[(measured["height_m"] - length / 2).abs().argsort().iloc[0]]
        out["form_quotient"] = float(mid["diameter_cm"]) / dbh
        low = measured[measured["height_m"] <= 0.35]
        if len(low):
            out["butt_swell"] = float(low["diameter_cm"].iloc[0]) / dbh

    frac, cci = out["frac_measured"], out["median_cci"]
    if np.isfinite(frac) and np.isfinite(cci):
        if frac >= 0.85 and cci >= 0.90:
            out["grade"] = "excellent"
        elif frac >= 0.70 and cci >= 0.75:
            out["grade"] = "good"
        elif frac >= 0.50 and cci >= 0.50:
            out["grade"] = "fair"
    return out


def voxel_surface_faces(points_xyz, voxel_m):
    """Exterior faces of the occupied voxel set -- the mesh whose volume
    is reported as crown_volume_m3.

    Only faces without an occupied neighbour are emitted. Drawing every
    face of every voxel would be ~6x more polygons and would render as an
    opaque block, hiding the shape the volume actually came from."""
    idx = np.floor(points_xyz / voxel_m).astype(np.int64)
    occupied = np.unique(idx, axis=0)
    if occupied.shape[0] == 0:
        return [], 0
    occ_set = set(map(tuple, occupied))

    # Unit-cube face corners, per axis and direction.
    corners = {
        (0, -1): [(0, 0, 0), (0, 1, 0), (0, 1, 1), (0, 0, 1)],
        (0, 1): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
        (1, -1): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        (1, 1): [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
        (2, -1): [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        (2, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    }
    faces = []
    for cell in occ_set:
        for (axis, direction), offsets in corners.items():
            neighbour = list(cell)
            neighbour[axis] += direction
            if tuple(neighbour) in occ_set:
                continue
            faces.append([
                ((cell[0] + ox) * voxel_m, (cell[1] + oy) * voxel_m, (cell[2] + oz) * voxel_m)
                for ox, oy, oz in offsets
            ])
    return faces, occupied.shape[0]


def pick_section_heights(taper_df, cfg, n_wanted=N_SECTIONS):
    """Sections to draw: always breast height (the one number everyone
    checks first), then the rest spread evenly over the measured range so
    the page shows the whole stem rather than a dense cluster near the
    base."""
    valid = taper_df[taper_df["n_points"] >= cfg.min_points_for_taper_slice]
    if valid.empty:
        return []
    heights = valid["height_m"].to_numpy()
    dbh_h = heights[np.argmin(np.abs(heights - cfg.dbh_height_m))]
    remaining = heights[heights != dbh_h]
    if remaining.size <= n_wanted - 1:
        chosen = list(remaining)
    else:
        take = np.linspace(0, remaining.size - 1, n_wanted - 1).round().astype(int)
        chosen = list(remaining[np.unique(take)])
    return sorted(set([dbh_h] + chosen))


def draw_section(ax, slice_xy, row, cfg, is_dbh, T):
    """One horizontal slice: its real points plus the circle that was fitted
    to them. Axes are metres relative to the fitted centre so every section
    is at the same scale and can be compared down the stem."""
    d_cm = row["diameter_cm"]
    d_corr = row.get("diameter_corrected_cm", np.nan)
    xc, yc = row["center_x"], row["center_y"]

    if not np.isfinite(xc):  # no valid fit: centre on the points themselves
        xc, yc = (slice_xy.mean(axis=0) if slice_xy.shape[0] else (0.0, 0.0))

    rel = slice_xy - np.array([xc, yc]) if slice_xy.shape[0] else np.empty((0, 2))
    ax.scatter(rel[:, 0], rel[:, 1], s=1.2, c=POINTS, alpha=0.55, linewidths=0, zorder=2)

    if np.isfinite(d_cm):
        ax.add_patch(Circle((0, 0), d_cm / 200.0, fill=False, ec=FIT, lw=1.6, zorder=3))
    if np.isfinite(d_corr) and np.isfinite(d_cm) and abs(d_corr - d_cm) > 0.05:
        ax.add_patch(Circle((0, 0), d_corr / 200.0, fill=False, ec=CORRECTED,
                             lw=1.4, ls=(0, (4, 2)), zorder=4))

    if np.isfinite(d_cm):
        span = max(0.35, (np.nanmax([d_cm, d_corr]) / 100.0) * 0.9)
    elif rel.shape[0]:
        # No circle was fitted, so there is no diameter to scale to -- frame
        # the points themselves, otherwise the panel renders empty and hides
        # exactly the case the reader most needs to see.
        span = max(0.35, float(np.percentile(np.abs(rel), 98)) * 1.15)
    else:
        span = 0.35
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d5d5d5")

    d_txt = f"{d_cm:.1f} cm" if np.isfinite(d_cm) else T["no_fit"]
    cci = row["cci"]
    cci_txt = f"CCI {cci:.2f}" if np.isfinite(cci) else "CCI -"
    title = f"{row['height_m']:.1f} m" + (f"  ({T['dbh_tag']})" if is_dbh else "")
    ax.set_title(title, fontsize=9, color=INK, pad=3,
                 fontweight="bold" if is_dbh else "normal")
    ax.text(0.5, -0.10, f"{d_txt}  ·  {cci_txt}  ·  n={int(row['n_points'])}",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.5,
            color=MUTED if np.isfinite(d_cm) else FIT)


def draw_taper(ax, taper_df, tree, cfg, T):
    ax.plot(taper_df["diameter_cm"], taper_df["height_m"], "o-", color=FIT,
            ms=3.5, lw=1.2, label=T["measured"], zorder=3)
    if "diameter_corrected_cm" in taper_df:
        ax.plot(taper_df["diameter_corrected_cm"], taper_df["height_m"],
                color=CORRECTED, lw=1.6, ls=(0, (4, 2)), label=T["corrected"], zorder=2)
    ax.axhline(cfg.dbh_height_m, color=MUTED, lw=0.8, ls=":")
    # Anchored in axes coordinates, not data -- a data-space label placed at
    # the right spine spills into the next panel of the page.
    ax.text(0.985, cfg.dbh_height_m, f"{T['dbh_tag']} 1.3 m", va="bottom", ha="right",
            transform=ax.get_yaxis_transform(), fontsize=7.5, color=MUTED)
    cbh = tree["crown_base_height_m"]
    if np.isfinite(cbh):
        ax.axhline(cbh, color=CROWN, lw=1.2)
        ax.text(0.985, cbh, f"{T['crown_base_label']} {cbh:.2f} m", va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=8.5, color=CROWN,
                fontweight="bold")
    ax.set_xlabel(T["diameter"], fontsize=9)
    ax.set_ylabel(T["height_axis"], fontsize=9)
    ax.set_title(T["taper_title"], fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.grid(alpha=0.15, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw_crown(ax, tree_xyz, hag, tree, cfg, T, crown_only=False):
    """Crown voxel mesh over the point cloud.

    A full 18m tree in a 4m footprint renders as an unreadable sliver at
    true aspect, so the page carries two views: the whole tree for context
    and the crown alone -- which is roughly as wide as it is tall, fills
    its panel, and is the part the volume actually measures. Both keep
    true proportions; neither stretches an axis."""
    cbh = tree["crown_base_height_m"]
    stem = tree_xyz[hag < cbh] if np.isfinite(cbh) else np.empty((0, 3))
    crown = tree_xyz[hag >= cbh] if np.isfinite(cbh) else tree_xyz

    shown = crown if crown_only else tree_xyz
    if shown.shape[0] == 0:
        ax.axis("off")
        return
    origin = shown.min(axis=0)

    if not crown_only and stem.shape[0]:
        sp = stem[::max(1, stem.shape[0] // 4000)] - origin
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=0.6, c=MUTED, alpha=0.40, linewidths=0)

    n_vox = 0
    if crown.shape[0]:
        cp = crown[::max(1, crown.shape[0] // (9000 if crown_only else 5000))] - origin
        ax.scatter(cp[:, 0], cp[:, 1], cp[:, 2], s=0.7, c=POINTS,
                   alpha=0.28 if crown_only else 0.18, linewidths=0)
        faces, n_vox = voxel_surface_faces(crown - origin, cfg.crown_voxel_size_m)
        if faces:
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=CROWN, edgecolors="#2f4f1f", linewidths=0.12,
                alpha=0.26 if crown_only else 0.32))

    pts = shown - origin
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(pts[:, 2].min(), pts[:, 2].max())
    try:
        ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), np.ptp(pts[:, 2])))
    except Exception:
        pass
    ax.view_init(elev=16, azim=-58)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel(T["x_axis"], fontsize=8, labelpad=-2)
    ax.set_ylabel(T["y_axis"], fontsize=8, labelpad=-2)
    ax.set_zlabel(T["height_short"] if crown_only else T["z_axis"], fontsize=8, labelpad=-2)

    if crown_only:
        vol = tree["crown_volume_m3"]
        title = (f"{T['crown_only']}\n{n_vox:,} {T['voxels_of']} "
                 f"{cfg.crown_voxel_size_m:g} m  ·  {vol:,.1f} m³")
        ax.set_title(title, fontsize=10, color=INK)
    else:
        ax.set_title(T["tree_full"], fontsize=10, color=INK)


def _flags_text(value):
    """quality_flags is NaN (a float, and truthy) when no flag was raised, so
    an `or` fallback silently prints "nan"."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() in ("", "nan"):
        return "nenhuma"
    return str(value)


def header_text(tree, T):
    def g(k, fmt="{:.2f}"):
        v = tree.get(k, np.nan)
        return fmt.format(v) if np.isfinite(v) else "-"

    left = (f"{T['height']} {g('height_m')} m      {T['dbh']} {g('dbh_cm', '{:.1f}')} cm "
            f"(CCI {g('dbh_cci')})      {T['crown_base']} {g('crown_base_height_m')} m")
    right = (f"{T['stem_volume']} {g('stem_volume_taper_m3')} m\u00b3      "
             f"{T['crown_volume']} {g('crown_volume_m3', '{:.1f}')} m\u00b3      "
             f"{T['live_crown_ratio']} {g('live_crown_ratio')}")
    return left, right


def render_tree(tree_id, tree, taper_df, tree_xyz, hag, cfg, out_path, dpi, T):
    fig = plt.figure(figsize=(16, 21))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.85, 0.95, 1.30], hspace=0.24, wspace=0.16,
                           left=0.055, right=0.965, top=0.893, bottom=0.035)

    fig.suptitle(f"{T['tree']} {int(tree_id)}", x=0.055, y=0.977, ha="left",
                 fontsize=22, fontweight="bold", color=INK)
    left, right = header_text(tree, T)
    fig.text(0.055, 0.953, left, ha="left", fontsize=10.5, color=INK)
    fig.text(0.055, 0.938, right, ha="left", fontsize=10.5, color=MUTED)

    fig.text(0.055, 0.910, T["sections_caption"],
             ha="left", va="bottom", fontsize=10.5, color=INK, fontweight="bold")

    # --- sections -----------------------------------------------------
    sec_gs = gs[0, :].subgridspec(SECTION_ROWS, SECTION_COLS, hspace=0.42, wspace=0.12)
    heights = pick_section_heights(taper_df, cfg)
    half = cfg.taper_slice_thickness_m / 2
    for i in range(N_SECTIONS):
        ax = fig.add_subplot(sec_gs[i // SECTION_COLS, i % SECTION_COLS])
        if i >= len(heights):
            ax.axis("off")
            continue
        h = heights[i]
        row = taper_df.iloc[(taper_df["height_m"] - h).abs().argsort().iloc[0]]
        slice_xy = tree_xyz[np.abs(hag - h) <= half][:, :2]
        draw_section(ax, slice_xy, row, cfg, abs(h - cfg.dbh_height_m) < 1e-6, T)


    ax_taper = fig.add_subplot(gs[1, 0])
    draw_taper(ax_taper, taper_df, tree, cfg, T)

    ax_txt = fig.add_subplot(gs[1, 1])
    ax_txt.axis("off")
    q = stem_quality_metrics(taper_df, tree, cfg)

    def num(v, fmt="{:.2f}", suffix=""):
        return (fmt.format(v) + suffix) if np.isfinite(v) else "-"

    lines = [
        (T["assortments"], ""),
        (T["sawlog"], f"{tree.get('volume_sawlog_m3', np.nan):.3f} m\u00b3  "
                       f"({int(tree.get('log_count_sawlog', 0) or 0)} {T['logs']})"),
        (T["pulpwood"], f"{tree.get('volume_pulpwood_m3', np.nan):.3f} m\u00b3  "
                         f"({int(tree.get('log_count_pulpwood', 0) or 0)} {T['logs']})"),
        (T["merchantable"], f"{tree.get('merchantable_volume_m3', np.nan):.3f} m\u00b3"),
        ("", ""),
        (T["stem_volume_hdr"], ""),
        (T["by_taper"], f"{tree.get('stem_volume_taper_m3', np.nan):.3f} m\u00b3"),
        (T["conic"], f"{tree.get('stem_volume_conic_m3', np.nan):.3f} m\u00b3"),
        ("", ""),
        (T["quality_hdr"], ""),
        (T["sections_measured"], f"{q['n_measured']} / {q['n_total']}"
                                  f"  ({num(100 * q['frac_measured'], '{:.0f}', '%')})"),
        (T["median_cci"], num(q["median_cci"])),
        (T["refitted"], str(q["n_refitted"])),
        (T["grade"], T["grades"][q["grade"]]),
        ("", ""),
        (T["tortuosity_hdr"], ""),
        (T["tortuosity"], num(q["tortuosity"], "{:.3f}")),
        (T["max_sweep"], num(q["max_sweep_cm"], "{:.0f}", " cm")),
        (T["lean"], num(float(tree.get("stem_lean_deg", np.nan)), "{:.1f}", "\u00b0")),
        ("", ""),
        (T["taper_hdr"], ""),
        (T["taper_rate"], num(q["taper_rate_cm_per_m"], "{:.2f}", " cm/m")),
        (T["form_quotient"], num(q["form_quotient"], "{:.2f}")),
        (T["butt_swell"], num(q["butt_swell"], "{:.2f}")),
    ]
    y = 0.985
    for k, v in lines:
        if k and not v:
            ax_txt.text(0, y, k, fontsize=10.5, color=INK, fontweight="bold", va="top")
        elif k:
            ax_txt.text(0.02, y, k, fontsize=9.5, color=MUTED, va="top")
            ax_txt.text(0.66, y, v, fontsize=9.5, color=INK, va="top")
        y -= 0.0415

    draw_crown(fig.add_subplot(gs[2, 0], projection="3d"), tree_xyz, hag, tree, cfg, T,
               crown_only=False)
    draw_crown(fig.add_subplot(gs[2, 1], projection="3d"), tree_xyz, hag, tree, cfg, T,
               crown_only=True)

    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-las", required=True, help="Point cloud the metrics were computed from.")
    p.add_argument("--metrics-dir", required=True, help="Directory holding <stem>_taper.csv and <stem>_tree_metrics.csv.")
    p.add_argument("--stem", required=True)
    p.add_argument("--tree-ids", nargs="+", type=int,
                    help="Trees to render. Omit and pass --all-trees for every tree in the plot.")
    p.add_argument("--all-trees", action="store_true",
                    help="Render every tree in <stem>_tree_metrics.csv. The point cloud is read "
                         "and the DTM built once for the whole run, so this costs far less per "
                         "tree than repeated single-tree invocations.")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dpi", type=int, default=200, help="Output resolution (default: %(default)s).")
    p.add_argument("--lang", default="en", choices=sorted(STRINGS),
                    help="Report language (default: %(default)s).")
    args = p.parse_args()
    if not args.tree_ids and not args.all_trees:
        p.error("pass --tree-ids, or --all-trees for every tree in the plot")

    cfg = ForestMetricsConfig()
    os.makedirs(args.output_dir, exist_ok=True)

    taper_all = pd.read_csv(os.path.join(args.metrics_dir, f"{args.stem}_taper.csv"))
    metrics_all = pd.read_csv(os.path.join(args.metrics_dir, f"{args.stem}_tree_metrics.csv"))

    T = STRINGS[args.lang]

    print(f"Reading {args.input_las} ...", flush=True)
    df = las_to_pandas(args.input_las)
    xyz_all = df[["X", "Y", "Z"]].values
    print("Building DTM (same one the metrics used) ...", flush=True)
    dtm, warnings = build_dtm(xyz_all[df["PredSemantic"].values == 0], xyz_all, cfg)
    for w in warnings:
        print(f"WARNING: {w}", flush=True)

    inst = df["PredInstance"].values
    tree_ids = args.tree_ids
    if args.all_trees:
        tree_ids = sorted(int(t) for t in metrics_all["tree_id"].unique())
        print(f"Rendering all {len(tree_ids)} tree(s) in {args.stem}", flush=True)
    for tree_id in tree_ids:
        rows = metrics_all[metrics_all["tree_id"] == tree_id]
        if rows.empty:
            print(f"tree {tree_id}: not in {args.stem}_tree_metrics.csv, skipping", flush=True)
            continue
        tree = rows.iloc[0]
        taper_df = taper_all[taper_all["tree_id"] == tree_id].sort_values("height_m").reset_index(drop=True)
        mask = (inst == tree_id) & (df["PredSemantic"].values == 1)
        if not mask.any():
            print(f"tree {tree_id}: no points, skipping", flush=True)
            continue
        tree_xyz = xyz_all[mask]
        hag = dtm.height_above_ground(tree_xyz)
        out_path = os.path.join(args.output_dir, f"{args.stem}_tree{int(tree_id)}.png")
        print(f"tree {tree_id}: {mask.sum():,} points, {len(taper_df)} taper rows -> {out_path}", flush=True)
        render_tree(tree_id, tree, taper_df, tree_xyz, hag, cfg, out_path, args.dpi, T)
    print("done", flush=True)


if __name__ == "__main__":
    main()
