# Marswalk

Interactive educational explorer for Jezero Crater combining Mars terrain, geology, and Perseverance mission data.

**Marswalk was developed as a NASA Space Apps Challenge 2026 project.** Project data and references were assembled through the project workflow and the provider resources listed below. NASA Space Apps participation does not imply NASA endorsement of the map or its interpretations.

## What it shows

The interactive Folium map includes elevation and derived hillshade, USGS geology units and clickable geology boundaries, and a provisional slope-screening visualization. It also includes a historical MEDA Sol 1 chart and a Perseverance Mastcam-Z image context panel.

The map is intended for exploration and education—not rover navigation or operational route planning. The slope-screening layer is provisional; terrain resolution and vertical units may be uncertain, and radiation, weather, obstacles, and actual rover mobility are not modeled. Do not treat colors, routes, or measurements as safety assessments.

## Data and acknowledgments

The map and analysis use or derive from the following products. Keep the provider credits and caveats with redistributed versions:

- **CTX elevation data for Jezero:** the project input is named `M20_JezeroCrater_CTXDEM_20m.tif`. The map's elevation and hillshade are based on this DEM; hillshade and slope screening are derived visualizations, not separate source datasets. Confirm and include the exact original product citation and download record when redistributing the raster—the repository source currently records the filename but not its full product-level provenance.
- **HiRISE stereo DTM:** `DTEEC_023524_1985_023379_1985_U01.IMG`, a digital terrain model from the High Resolution Imaging Science Experiment (HiRISE) on NASA's Mars Reconnaissance Orbiter. The project workflow inspects and aligns this DTM to the CTX DEM projection and derives slope and hillshade products from it. The current final interactive map's elevation and hillshade layers are based on the CTX DEM, not this HiRISE DTM. Product page: [HiRISE DTM: Proposed Landing Site in Jezero Crater](https://www.uahirise.org/dtm/dtm.php?ID=DTEEC_023524_1985_023379_1985_U01). Suggested image credit: NASA/JPL-Caltech/UArizona. Check the product label and PDS records for full product citation and data terms.
- **Geology:** USGS–NASA Planetary Geologic Mapping Program, *SIM 3464* Jezero/Nili geology database, `GeoUnits_v1` layer. The workflow downloads the archive from [USGS SIM 3464](https://pubs.usgs.gov/sim/3464/sim3464_database.zip), reprojects it to the DEM coordinate reference system, and derives the geology overlay and clickable boundaries. Credit the USGS and NASA source publication/database; retain its publication details and any applicable data-use terms.
- **Perseverance image context panel:** NASA image **PIA25917**, “Perseverance Finds a Hollowed-Out Rock,” captured by Mastcam-Z at Jezero Crater on sol 836. [NASA image record](https://images.nasa.gov/details/PIA25917). The image is context only; its location is not plotted on the map.
- **MEDA historical sample:** Perseverance MEDA Environmental RDR Sol 1 pressure (PS), thermal infrared (TIRS), and humidity-sensor (RHS) data. The workflow uses the corresponding Sol 1 CSV products and the [MEDA Environmental RDR Software Interface Specification](https://pds-atmospheres.nmsu.edu/PDS/data/PDS4/Mars2020/mars2020_meda/document/meda_env_rdr_sis.pdf) to interpret plotted units. The displayed sample is historical, not contemporaneous with the sol-836 image or a forecast; relative humidity has no valid values in this sample.
- **Mars exploration reference:** JPL's [Mars Trek](https://trek.nasa.gov/mars/) is acknowledged as a web-based Mars exploration and data-discovery reference. It is not represented as the direct source of the rendered DEM or geology layer; those are identified above.

The repository's MIT License applies to the Marswalk project code and documentation, **not automatically to third-party datasets, publications, or imagery**. Verify source-specific attribution and redistribution terms before including those assets in this repository or distributing them with the map.

## Project status

The repository contains the Colab notebook, Python script, generated HTML map, README, and license. The source workflow expects local input files (including the CTX DEM, HiRISE DTM, and MEDA CSV products) and can download some products, such as the USGS geology archive and NASA context image. Large inputs and raw datasets are not included in this repository. See the input filenames and paths in `Marswalk_EVA_Map.py` before running it; a fresh clone is not currently a complete, self-contained rebuild environment.

## Viewing the generated map

Open `marswalk_eva_map_with_meda.html` in a modern browser. If hosting it, serve the HTML over HTTP(S); a local file path or runtime-only Colab path may not work when embedded in an iframe. The HTML is a generated output; modify the notebook or Python source to change the map.

## Development

The workflow uses Python, Folium, NumPy, SciPy, pandas, Matplotlib, Rasterio, GeoPandas, Fiona, Shapely, PyProj, Pillow, Requests, PyMuPDF, and Gradio for selected steps. See the install command and step options in `Marswalk_EVA_Map.py`; exact dependency versions and input acquisition details should be pinned/documented for reproducible runs.

When contributing:

1. Preserve provider attribution and limitations in the map and README.
2. Add only one Folium layer control, after all toggleable map layers are created.
3. Test the saved HTML itself, not only the notebook's displayed output.
4. Do not present provisional screening or illustrative routes as validated rover-safety guidance.

## License

Marswalk project code and documentation are licensed under the [MIT License](LICENSE). Third-party data and imagery may have separate terms and attribution requirements; see **Data and acknowledgments**.
