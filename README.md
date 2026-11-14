# Marswalk

Interactive educational explorer for Jezero Crater combining Mars terrain, geology, and Perseverance mission data.

## What it shows

The Marswalk map is an interactive Folium product with selectable terrain and geology layers, clickable geology boundaries, and a provisional slope-screening layer, built as a NASA Space Apps 2026 project. The current map also includes a MEDA Sol 1 historical-data chart and a Perseverance image context panel.

The map is intended for exploration and education—not rover navigation or operational route planning. The slope-screening layer is explicitly provisional; terrain resolution, vertical units, radiation, weather, obstacles, and actual rover mobility are not fully modeled. Do not treat colors, routes, or measurements as safety assessments.

## Project status

The repository currently contains the project documentation and license. The map was developed in Google Colab, with output saved separately in Google Drive. The map-building notebook, Python source, input datasets, and generated HTML have not yet been added here. Add the source notebook/script and document its required inputs before expecting a fresh clone to reproduce the map.

## Map contents and data notes

- Elevation and derived hillshade for terrain context.
- USGS geology units and clickable geology boundaries.
- A provisional slope-screening visualization, not a safety rating.
- MEDA Sol 1 historical sample chart. It is historical context, not a live feed or a measurement contemporaneous with the map imagery; relative humidity is unavailable in the displayed sample.
- Perseverance image context panel. The image is for context and is not plotted as a map location unless explicitly stated in the map.

Use the data-source citations included in the map and its source notebook. Keep those citations with any redistributed map or derivative product, and verify dataset versions, coordinate reference systems, units, and license/attribution requirements before reuse.

## Viewing the generated map

The current generated product is an HTML map (the recent Colab output was named `marswalk_eva_map_with_meda.html`). Open the HTML file in a modern browser. If hosting it, serve the HTML and any external assets over HTTP(S); opening a local file or embedding a runtime-only path in an iframe may not work in every environment.

The generated HTML is an output, not the source of the map. To rebuild or modify the map, use the map-building notebook/script once it has been added to this repository, along with its documented inputs.

## Development

The source workflow uses Python and Folium for the interactive map, with pandas and Matplotlib for MEDA data processing and plotting. Exact dependency versions and input-file locations should be recorded in the source notebook or a requirements file when that workflow is committed.

When contributing changes:

1. Preserve data attribution and caveats in the map.
2. Keep the final map control setup idempotent—add one layer control after the map layers are created, not once per rerun or helper call.
3. Test the saved HTML itself, not only the notebook's displayed output.
4. Do not describe provisional screening or illustrative routes as validated rover-safety guidance.

## License

The Marswalk project code and documentation are licensed under the MIT License; see [LICENSE](LICENSE). Third-party datasets, imagery, and other assets may have separate terms and attribution requirements. Check their source-specific conditions before redistribution.
