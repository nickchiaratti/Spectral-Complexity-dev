"""
Generate a KMZ file containing bounding boxes and processing grid footprints
for all locations defined in locations_config.yaml.

Compatible with Google Earth, QGIS, ArcGIS Pro, and GDAL/OGR.
"""

import argparse
import math
import os
import sys
import zipfile
from typing import Dict, Any, List, Tuple

import pyproj
import yaml


def kml_color(hex_rgb: str, alpha_hex: str = "ff") -> str:
    """
    Convert a hex RGB string ('#RRGGBB' or 'RRGGBB') to KML 'aabbggrr' format.
    """
    hex_rgb = hex_rgb.lstrip("#")
    if len(hex_rgb) != 6:
        raise ValueError(f"Invalid RGB hex code: '{hex_rgb}'. Expected 6 characters.")
    r = hex_rgb[0:2]
    g = hex_rgb[2:4]
    b = hex_rgb[4:6]
    return f"{alpha_hex.lower()}{b.lower()}{g.lower()}{r.lower()}"


def densify_and_project_utm_box(
    epsg: int,
    ul_x: float,
    ul_y: float,
    width: int,
    height: int,
    gsd: float,
    n_steps: int = 10,
) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    """
    Densify the rectangular UTM boundary along all 4 edges and reproject each
    vertex to WGS84 (EPSG:4326) longitude/latitude.

    Returns:
        wgs_coords: list of (lon, lat) tuples forming a closed polygon ring.
        center_wgs: (center_lon, center_lat) tuple.
    """
    lr_x = ul_x + width * gsd
    lr_y = ul_y - height * gsd

    edge_pts = []
    # Top edge: (ul_x, ul_y) -> (lr_x, ul_y)
    for i in range(n_steps):
        frac = i / n_steps
        edge_pts.append((ul_x + frac * (lr_x - ul_x), ul_y))
    # Right edge: (lr_x, ul_y) -> (lr_x, lr_y)
    for i in range(n_steps):
        frac = i / n_steps
        edge_pts.append((lr_x, ul_y + frac * (lr_y - ul_y)))
    # Bottom edge: (lr_x, lr_y) -> (ul_x, lr_y)
    for i in range(n_steps):
        frac = i / n_steps
        edge_pts.append((lr_x + frac * (ul_x - lr_x), lr_y))
    # Left edge: (ul_x, lr_y) -> (ul_x, ul_y)
    for i in range(n_steps):
        frac = i / n_steps
        edge_pts.append((ul_x, lr_y + frac * (ul_y - lr_y)))
    # Close polygon
    edge_pts.append((ul_x, ul_y))

    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    wgs_coords = [transformer.transform(x, y) for x, y in edge_pts]

    center_x = (ul_x + lr_x) / 2.0
    center_y = (ul_y + lr_y) / 2.0
    center_wgs = transformer.transform(center_x, center_y)

    return wgs_coords, center_wgs


def build_kml_document(locations: Dict[str, Any]) -> str:
    """
    Build complete KML 2.2 XML document string from the parsed locations dictionary.
    """
    palette = {
        "Rochesterv2": "#0099FF",      # Blue
        "Tait": "#FF9900",             # Amber / Orange
        "MtEtna-Catania": "#9933FF",   # Violet
        "MtEtna": "#FF3366",           # Coral Red
        "BuenosAires": "#00CC66",      # Emerald Green
        "Hurlingham": "#88EE00",       # Lime
        "Palisades": "#00CED1",        # Dark Turquoise
        "Malibu": "#FFD700",           # Gold
        "CentralGreece": "#3399FF",    # Sky Blue
        "SantaBarbara": "#7B1FA2",     # Purple
        "SanRafael": "#FF1493",        # Deep Pink
    }

    kml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2">',
        '<Document>',
        '  <name>Spectral Complexity Research Locations</name>',
        '  <open>1</open>',
        '  <description><![CDATA[',
        '    <h3>Spectral Complexity Research Locations</h3>',
        '    <p>Bounding boxes and sensor footprints defined in <code>locations_config.yaml</code>.</p>',
        '  ]]></description>',
    ]

    # Location styles
    for loc_name in locations.keys():
        rgb = palette.get(loc_name, "#00B0FF")
        line_col = kml_color(rgb, alpha_hex="FF")
        poly_col = kml_color(rgb, alpha_hex="2E")  # ~18% semi-transparent fill

        style_xml = f"""  <Style id="style_{loc_name}">
    <LineStyle>
      <color>{line_col}</color>
      <width>2.5</width>
    </LineStyle>
    <PolyStyle>
      <color>{poly_col}</color>
      <fill>1</fill>
      <outline>1</outline>
    </PolyStyle>
    <BalloonStyle>
      <text><![CDATA[
        <div style="font-family: Arial, sans-serif; min-width: 300px; padding: 4px;">
          <h2 style="margin: 0 0 8px 0; color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 4px;">$[name]</h2>
          $[description]
        </div>
      ]]></text>
    </BalloonStyle>
  </Style>"""
        kml_lines.append(style_xml)

    # Style for MGRS Grids
    mgrs_line_col = kml_color("#FFFFFF", alpha_hex="A0")
    mgrs_poly_col = kml_color("#FFFFFF", alpha_hex="10")
    mgrs_style = f"""  <Style id="style_MGRS">
    <LineStyle>
      <color>{mgrs_line_col}</color>
      <width>1.8</width>
    </LineStyle>
    <PolyStyle>
      <color>{mgrs_poly_col}</color>
      <fill>1</fill>
      <outline>1</outline>
    </PolyStyle>
    <BalloonStyle>
      <text><![CDATA[
        <div style="font-family: Arial, sans-serif; min-width: 300px; padding: 4px;">
          <h2 style="margin: 0 0 8px 0; color: #5f6368; border-bottom: 2px solid #5f6368; padding-bottom: 4px;">$[name]</h2>
          $[description]
        </div>
      ]]></text>
    </BalloonStyle>
  </Style>"""
    kml_lines.append(mgrs_style)

    # Folder 1: ROI Bounding Boxes
    kml_lines.append('  <Folder>')
    kml_lines.append('    <name>ROI Bounding Boxes</name>')
    kml_lines.append('    <open>1</open>')
    kml_lines.append('    <description>Target Region of Interest (ROI) geographic bounding boxes</description>')

    for loc_name, params in locations.items():
        # Validate presence of required ROI coordinates
        for req_key in ("ROI_LON_MIN", "ROI_LON_MAX", "ROI_LAT_MIN", "ROI_LAT_MAX"):
            if req_key not in params or params[req_key] is None:
                raise ValueError(
                    f"CRITICAL: Location '{loc_name}' is missing required coordinate '{req_key}'."
                )

        lon_min = float(params["ROI_LON_MIN"])
        lon_max = float(params["ROI_LON_MAX"])
        lat_min = float(params["ROI_LAT_MIN"])
        lat_max = float(params["ROI_LAT_MAX"])

        if lon_min >= lon_max:
            raise ValueError(f"Invalid longitude bounds for '{loc_name}': lon_min ({lon_min}) >= lon_max ({lon_max})")
        if lat_min >= lat_max:
            raise ValueError(f"Invalid latitude bounds for '{loc_name}': lat_min ({lat_min}) >= lat_max ({lat_max})")

        center_lon = (lon_min + lon_max) / 2.0
        center_lat = (lat_min + lat_max) / 2.0

        d_lat_m = abs(lat_max - lat_min) * 111320.0
        d_lon_m = abs(lon_max - lon_min) * 111320.0 * math.cos(math.radians(center_lat))
        diag_m = math.sqrt(d_lat_m**2 + d_lon_m**2)
        lookat_range = max(diag_m * 1.6, 5000.0)

        source_cache = params.get("SOURCE_CACHE") or ""
        loc_type = "Sub-ROI" if source_cache and source_cache != loc_name else "Regional ROI"

        tanager_avail = bool(params.get("TANAGER_AVAILABLE", False))
        dragonette_avail = bool(params.get("DRAGONETTE_AVAILABLE", False))
        enmap_avail = bool(params.get("ENMAP_AVAILABLE", False))

        desc_html = f"""<table style="width: 100%; font-size: 12px; border-collapse: collapse;">
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Location Type</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{loc_type}</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Source Cache</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{source_cache if source_cache else '<em>Self (Regional Root)</em>'}</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Longitude Range</td>
    <td style="padding: 4px; border: 1px solid #ddd;">[{lon_min:.6f}, {lon_max:.6f}] ({abs(lon_max - lon_min):.4f}&deg;)</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Latitude Range</td>
    <td style="padding: 4px; border: 1px solid #ddd;">[{lat_min:.6f}, {lat_max:.6f}] ({abs(lat_max - lat_min):.4f}&deg;)</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Center (Lon, Lat)</td>
    <td style="padding: 4px; border: 1px solid #ddd;">({center_lon:.6f}, {center_lat:.6f})</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Approx. Footprint</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{d_lon_m/1000.0:.2f} km &times; {d_lat_m/1000.0:.2f} km ({(d_lon_m*d_lat_m)/1e6:.1f} km&sup2;)</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Date Range</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{params.get('START_DATE', 'N/A')} to {params.get('END_DATE', 'N/A')}</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Sensors Available</td>
    <td style="padding: 4px; border: 1px solid #ddd;">Tanager: <b>{'Yes' if tanager_avail else 'No'}</b> | Dragonette: <b>{'Yes' if dragonette_avail else 'No'}</b> | EnMAP: <b>{'Yes' if enmap_avail else 'No'}</b></td>
  </tr>
</table>"""

        ext_data = f"""      <ExtendedData>
        <Data name="Location"><value>{loc_name}</value></Data>
        <Data name="Type"><value>{loc_type}</value></Data>
        <Data name="SourceCache"><value>{source_cache}</value></Data>
        <Data name="LonMin"><value>{lon_min}</value></Data>
        <Data name="LonMax"><value>{lon_max}</value></Data>
        <Data name="LatMin"><value>{lat_min}</value></Data>
        <Data name="LatMax"><value>{lat_max}</value></Data>
        <Data name="CenterLon"><value>{center_lon:.6f}</value></Data>
        <Data name="CenterLat"><value>{center_lat:.6f}</value></Data>
        <Data name="AreaKm2"><value>{(d_lon_m*d_lat_m)/1e6:.2f}</value></Data>
        <Data name="StartDate"><value>{params.get('START_DATE', '')}</value></Data>
        <Data name="EndDate"><value>{params.get('END_DATE', '')}</value></Data>
        <Data name="Tanager"><value>{str(tanager_avail).lower()}</value></Data>
        <Data name="Dragonette"><value>{str(dragonette_avail).lower()}</value></Data>
        <Data name="EnMAP"><value>{str(enmap_avail).lower()}</value></Data>
      </ExtendedData>"""

        # Counter-clockwise closed ring (lon, lat, alt)
        coords = [
            f"{lon_min:.6f},{lat_min:.6f},0",
            f"{lon_max:.6f},{lat_min:.6f},0",
            f"{lon_max:.6f},{lat_max:.6f},0",
            f"{lon_min:.6f},{lat_max:.6f},0",
            f"{lon_min:.6f},{lat_min:.6f},0",
        ]
        coords_str = " ".join(coords)

        placemark = f"""    <Placemark>
      <name>{loc_name}</name>
      <styleUrl>#style_{loc_name}</styleUrl>
      <description><![CDATA[{desc_html}]]></description>
{ext_data}
      <LookAt>
        <longitude>{center_lon:.6f}</longitude>
        <latitude>{center_lat:.6f}</latitude>
        <altitude>0</altitude>
        <heading>0</heading>
        <tilt>0</tilt>
        <range>{lookat_range:.1f}</range>
        <altitudeMode>clampToGround</altitudeMode>
      </LookAt>
      <Polygon>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coords_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""
        kml_lines.append(placemark)

    kml_lines.append('  </Folder>')

    # Folder 2: MGRS Processing Grids
    kml_lines.append('  <Folder>')
    kml_lines.append('    <name>MGRS Processing Grids</name>')
    kml_lines.append('    <open>0</open>')
    kml_lines.append('    <description>MGRS / HLS projected processing grid extents (30m target GSD)</description>')

    processed_mgrs_keys = set()
    for loc_name, params in locations.items():
        epsg = params.get("MGRS_EPSG")
        ul_x = params.get("MGRS_UL_X")
        ul_y = params.get("MGRS_UL_Y")
        w = params.get("MGRS_WIDTH")
        h = params.get("MGRS_HEIGHT")
        gsd = float(params.get("TARGET_GSD", 30.0))

        if epsg is None and ul_x is None and ul_y is None and w is None and h is None:
            continue

        if any(v is None for v in (epsg, ul_x, ul_y, w, h)):
            raise ValueError(
                f"Incomplete MGRS grid definition for '{loc_name}': "
                f"EPSG={epsg}, UL_X={ul_x}, UL_Y={ul_y}, WIDTH={w}, HEIGHT={h}"
            )

        mgrs_key = (epsg, ul_x, ul_y, w, h, gsd)
        if mgrs_key in processed_mgrs_keys:
            continue
        processed_mgrs_keys.add(mgrs_key)

        cache_name = params.get("SOURCE_CACHE") or loc_name
        grid_name = f"MGRS Grid - {cache_name} (EPSG:{epsg})"

        wgs_pts, center_wgs = densify_and_project_utm_box(
            epsg=int(epsg),
            ul_x=float(ul_x),
            ul_y=float(ul_y),
            width=int(w),
            height=int(h),
            gsd=gsd,
            n_steps=10,
        )

        mgrs_coords_str = " ".join(f"{lon:.6f},{lat:.6f},0" for lon, lat in wgs_pts)
        mgrs_range = max(math.sqrt((w * gsd)**2 + (h * gsd)**2) * 1.6, 10000.0)

        mgrs_desc = f"""<table style="width: 100%; font-size: 12px; border-collapse: collapse;">
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Associated Cache</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{cache_name}</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Projected CRS</td>
    <td style="padding: 4px; border: 1px solid #ddd;">EPSG:{epsg}</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Upper-Left (X, Y)</td>
    <td style="padding: 4px; border: 1px solid #ddd;">({ul_x:.1f}, {ul_y:.1f}) m</td>
  </tr>
  <tr>
    <td style="padding: 4px; border: 1px solid #ddd;">Grid Dimensions</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{w} cols &times; {h} rows ({w*h:,} pixels)</td>
  </tr>
  <tr style="background-color: #f8f9fa;">
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Target GSD</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{gsd:.1f} m</td>
  </tr>
  <tr>
    <td style="padding: 4px; font-weight: bold; border: 1px solid #ddd;">Physical Dimensions</td>
    <td style="padding: 4px; border: 1px solid #ddd;">{(w*gsd)/1000.0:.2f} km &times; {(h*gsd)/1000.0:.2f} km ({((w*gsd)*(h*gsd))/1e6:.1f} km&sup2;)</td>
  </tr>
</table>"""

        mgrs_ext_data = f"""      <ExtendedData>
        <Data name="CacheName"><value>{cache_name}</value></Data>
        <Data name="EPSG"><value>{epsg}</value></Data>
        <Data name="UL_X"><value>{ul_x}</value></Data>
        <Data name="UL_Y"><value>{ul_y}</value></Data>
        <Data name="Width"><value>{w}</value></Data>
        <Data name="Height"><value>{h}</value></Data>
        <Data name="TargetGSD"><value>{gsd}</value></Data>
        <Data name="TotalPixels"><value>{w*h}</value></Data>
      </ExtendedData>"""

        mgrs_placemark = f"""    <Placemark>
      <name>{grid_name}</name>
      <styleUrl>#style_MGRS</styleUrl>
      <description><![CDATA[{mgrs_desc}]]></description>
{mgrs_ext_data}
      <LookAt>
        <longitude>{center_wgs[0]:.6f}</longitude>
        <latitude>{center_wgs[1]:.6f}</latitude>
        <altitude>0</altitude>
        <heading>0</heading>
        <tilt>0</tilt>
        <range>{mgrs_range:.1f}</range>
        <altitudeMode>clampToGround</altitudeMode>
      </LookAt>
      <Polygon>
        <tessellate>1</tessellate>
        <altitudeMode>clampToGround</altitudeMode>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{mgrs_coords_str}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>"""
        kml_lines.append(mgrs_placemark)

    kml_lines.append('  </Folder>')
    kml_lines.append('</Document>')
    kml_lines.append('</kml>')

    return "\n".join(kml_lines)


def generate_kmz(config_path: str, output_kmz_path: str) -> None:
    """
    Read config_path, generate KML document, and pack into output_kmz_path.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    locations = cfg.get("locations", {})
    if not locations:
        raise ValueError(f"No 'locations' dictionary found in {config_path}")

    kml_content = build_kml_document(locations)

    out_dir = os.path.dirname(os.path.abspath(output_kmz_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with zipfile.ZipFile(output_kmz_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_content.encode("utf-8"))

    print(f"Successfully generated KMZ archive: {output_kmz_path}")
    print(f"  Total Locations Processed: {len(locations)}")
    print(f"  KMZ File Size: {os.path.getsize(output_kmz_path):,} bytes")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config = os.path.join(script_dir, "locations_config.yaml")
    default_output = os.path.join(script_dir, "locations.kmz")

    parser = argparse.ArgumentParser(
        description="Generate a KMZ file of bounding boxes from locations_config.yaml"
    )
    parser.add_argument(
        "-c",
        "--config",
        default=default_config,
        help=f"Path to locations_config.yaml (default: {default_config})",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_output,
        help=f"Output path for KMZ file (default: {default_output})",
    )

    args = parser.parse_args()
    generate_kmz(args.config, args.output)


if __name__ == "__main__":
    main()
