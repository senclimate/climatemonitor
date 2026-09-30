import warnings
warnings.filterwarnings("ignore")

import os
import json
from datetime import date

import numpy as np
import xarray as xr
import plotly.utils as putils

import senpy as sp
from senpy.plot_interactive import *
from senpy.xtrend import *


MYHOME = os.getenv("MYHOME")
os.environ["BROWSER_PATH"] = f"{MYHOME}/apps/chrome-linux64/chrome"

this_year = date.today().year
overwrite = False

# ============================================================
# Step 1: Download daily OISST
# ============================================================

sst_dir = f"{MYHOME}/data/sst/OISSTv2"
day_down_script = f"{sst_dir}/down_day_oisst.sh"

# os.system(f"bash {day_down_script}")

# ============================================================
# Step 2: Calculate daily SST indices
# ============================================================

save_dir = "./data"
os.makedirs(save_dir, exist_ok=True)

for yr in range(1982, this_year + 1):
    save_file = f"{save_dir}/ssti.day.mean.{yr}.nc"

    if os.path.exists(save_file):
        if yr != this_year or not overwrite:
            continue

    print(f"Processing {yr}...")
    sst_file = f"{sst_dir}/sst.day.mean.{yr}.nc"

    with xr.open_dataset(sst_file) as ds:
        ssti_ds = (
            sp.global_sst_indices(ds["sst"])
            .rename({"SASD2": "SASD"})
        )
        ssti_ds.to_netcdf(save_file)

# ============================================================
# Step 3: Prepare data
# ============================================================

ssti_ds = xr.open_mfdataset(
    f"{save_dir}/ssti.day.mean.*.nc"
).load()

ssti_ds2 = ssti_ds.dayclim.set_calendar(calendar="noleap")

clim_slice = slice("1991-01-01", "2020-12-31")

ssti_c2 = ssti_ds2.dayclim.climatology(
    clim_slice=clim_slice,
    calendar="noleap",
)

ssti_a2 = ssti_ds2.dayclim.anomalies(
    clim_slice=clim_slice,
    smooth=False,
    calendar="noleap",
)

ssti_da2 = ssti_a2.trendPolyfit.detrend(
    trend_period=slice("1982-01", "2025-12"),
    order=1,
)

# ============================================================
# Step 4: Plot settings
# ============================================================

plot_vars = [
    "Nino34", "Nino3", "Nino4", "Nino12", "NPMM", "SPMM", "IOD", "IODe", "IODw", "IOB", "SIOD",
    "TNA", "ATL3", "SASD", "tropic", "AMO_G", "SouthernOcean",
]

index_names = {
    v: sp.translate_index_name(v)
    for v in plot_vars
}

year_categories = {
    "All": list(range(1982, this_year + 1))
}

year_colors = {
    1972: "#3B6FB6",
    1982: "#E69F00",
    1997: "#7A5AA6",
    2015: "#56B4E9",
    2023: "#D55E00",
    this_year: "#B2182B",
}

highlight_years = [1982, 1997, 2015, this_year]

data_modes = {
    "anomaly": {
        "data": ssti_a2,
        "data_clim": None,
        "title": "Daily {index} anomaly relative to 1991–2020",
        "y_title": "SST anomaly (°C)",
    },
    "detrended": {
        "data": ssti_da2,
        "data_clim": None,
        "title": (
            "Daily {index} detrended anomaly relative to 1991–2020 "
            "(detrended over 1982–2025)"
        ),
        "y_title": "SST anomaly (°C)",
    },
    "raw": {
        "data": ssti_ds2,
        "data_clim": ssti_c2,
        "title": "Daily {index}",
        "y_title": "SST (°C)",
    },
}

# ============================================================
# Step 5: Generate / incrementally update Plotly data
# ============================================================

json_file = "daily_ssti.json"
overwrite_json = True

def clean_values(values):
    if values is None:
        return []
    values = np.asarray(values, dtype=float)
    values = np.round(values, 4)
    return [None if not np.isfinite(v) else float(v) for v in values]

def make_fig(variable, mode):
    config = data_modes[mode]
    fig = plot_daily_years(
        data=config["data"],
        year_categories=year_categories,
        variable=variable,
        data_clim=config["data_clim"],
        year_colors=year_colors,
        highlight_years=highlight_years,
        day_extent=(30, 150),
        title=config["title"].format(index=index_names[variable]),
        y_title=config["y_title"],
        legend_title="",
        width=1000,
        height=500,
    )
    # fig.update_layout(
    #     font=dict(size=16),
    #     title=None,
    #     legend=dict(font=dict(size=14), title=dict(font=dict(size=15))),
    #     xaxis=dict(title=dict(font=dict(size=14)), tickfont=dict(size=12)),
    #     yaxis=dict(
    #         title=dict(text=config["y_title"], font=dict(size=14)),
    #         tickfont=dict(size=12),
    #     ),
    # )
    return fig

def make_template(fig):
    template = {
        "layout": fig.layout.to_plotly_json(),
        "traces": [],
    }
    for trace in fig.data:
        trace_dict = trace.to_plotly_json()
        trace_dict.pop("y", None)
        if trace_dict.get("meta") == "latest_point":
            trace_dict.pop("x", None)
            trace_dict.pop("text", None)
        template["traces"].append(trace_dict)
    return template

def make_trace_values(fig):
    trace_values = []
    for trace in fig.data:
        item = {"y": clean_values(trace.y)}
        if trace.meta == "latest_point":
            item["x"] = np.asarray(trace.x, dtype=float).tolist()
            item["text"] = list(trace.text)
        trace_values.append(item)
    return trace_values

def generate_full_json():
    print("Generating complete JSON...")
    templates = {}
    values = {}

    for mode in data_modes:
        print(f"Creating {mode} template...")
        fig = make_fig("Nino34", mode)
        templates[mode] = make_template(fig)
        values[mode] = {}

        for variable in plot_vars:
            print(f"  {variable}")
            fig = make_fig(variable, mode)
            values[mode][variable] = make_trace_values(fig)

    web_data = {
        "metadata": {
            "calendar": "noleap",
            "climatology_period": "1991-2020",
            "detrend_period": "1982-2025",
            "start_year": 1982,
            "end_year": this_year,
            "indices": plot_vars,
            "index_names": index_names,
            "default_index": "Nino34",
            "default_mode": "anomaly",
        },
        "templates": templates,
        "values": values,
    }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            web_data,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            cls=putils.PlotlyJSONEncoder,
            allow_nan=False,
        )

    print(f"Saved {json_file} ({os.path.getsize(json_file) / 1024**2:.2f} MB)")
    return web_data

def update_current_year_json(web_data):
    print(f"Updating {this_year} only...")

    for mode in data_modes:
        for variable in plot_vars:
            fig = make_fig(variable, mode)
            old_values = web_data["values"][mode][variable]

            if len(fig.data) != len(old_values):
                raise ValueError(
                    f"Trace number changed for {mode}/{variable}. "
                    "Run with overwrite_json=True."
                )

            for i, trace in enumerate(fig.data):
                new_value = make_trace_values(fig)[i]

                if trace.meta == "latest_point":
                    old_values[i] = new_value
                elif str(trace.name) == str(this_year):
                    old_values[i]["y"] = new_value["y"]

    web_data["metadata"]["end_year"] = this_year

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(
            web_data,
            f,
            ensure_ascii=False,
            separators=(",", ":"),
            cls=putils.PlotlyJSONEncoder,
            allow_nan=False,
        )

    print(f"Updated {json_file} ({os.path.getsize(json_file) / 1024**2:.2f} MB)")
    return web_data

if overwrite_json or not os.path.exists(json_file):
    web_data = generate_full_json()
else:
    with open(json_file, "r", encoding="utf-8") as f:
        web_data = json.load(f)

    if web_data["metadata"]["end_year"] != this_year:
        print("New year detected; rebuilding JSON...")
        web_data = generate_full_json()
    else:
        web_data = update_current_year_json(web_data)


# ============================================================
# Step 7: Create index.html
# ============================================================

index_options = "\n".join(
    f'<option value="{v}">{index_names[v]}</option>'
    for v in plot_vars
)

html = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Climate Monitor</title>

<script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script>

<style>
body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    color: #222;
    background: #fff;
}

.container {
    width: 94%;
    max-width: 1400px;
    margin: 0 auto;
}

header {
    padding: 25px 0 15px;
    border-bottom: 1px solid #ddd;
}

header h1 {
    margin: 0;
    font-size: 28px;
}

header p {
    margin: 5px 0 0;
    color: #666;
}

.controls {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 25px;
    margin: 20px 0 10px;
}

.control-group,
.mode-controls {
    display: flex;
    align-items: center;
    gap: 8px;
}

.mode-controls {
    gap: 14px;
}

.control-label {
    font-weight: 600;
}

select {
    font-size: 15px;
    padding: 6px 10px;
    border: 1px solid #aaa;
    border-radius: 5px;
    background: #fff;
}

.mode-checkbox {
    display: flex;
    align-items: center;
    gap: 5px;
    cursor: pointer;
}

.mode-checkbox input {
    width: 15px;
    height: 15px;
}

#plot {
    width: 100%;
}

#loading {
    margin: 20px 0;
    color: #666;
}

footer {
    margin-top: 20px;
    padding: 15px 0 25px;
    border-top: 1px solid #ddd;
    color: #777;
    font-size: 13px;
}
</style>
</head>

<body>

<header>
<div class="container">
<h1>Climate Monitor</h1>
<p>Daily pantropical climate mode indices</p>
</div>
</header>

<main class="container">

<div class="controls">

<div class="control-group">
<span class="control-label">Index:</span>
<select id="index-select">
__INDEX_OPTIONS__
</select>
</div>

<div class="mode-controls">
<span class="control-label">Data:</span>

<label class="mode-checkbox">
<input type="checkbox" name="data-mode" value="anomaly" checked>
Anomaly
</label>

<label class="mode-checkbox">
<input type="checkbox" name="data-mode" value="detrended">
Detrended
</label>

<label class="mode-checkbox">
<input type="checkbox" name="data-mode" value="raw">
Raw
</label>
</div>

</div>

<div id="loading">Loading climate data...</div>
<div id="plot"></div>

</main>

<footer>
<div class="container">

<div class="data-info">
<strong>Data & Methods</strong><br>
Daily Sea Surface Temperature (SST) from
<a href="https://www.ncei.noaa.gov/products/optimum-interpolation-sst"
target="_blank">NOAA OISST v2.1</a> (0.25° global grid).
Climate mode indices are calculated following
<a href="https://doi.org/10.1038/s41586-024-07534-6"
target="_blank">Zhao et al. (2024, Nature)</a>.
Anomalies use the 1991–2020 climatology; detrended indices are linearly
detrended over 1982–2025.
</div>

<br>

<a href="https://senzhao.netlify.app/" target="_blank">Sen Zhao</a>
· University of Hawaiʻi at Mānoa

</div>
</footer>


<script>

let climateData = null;

const plotDiv = document.getElementById("plot");
const loadingDiv = document.getElementById("loading");
const indexSelect = document.getElementById("index-select");
const modeCheckboxes = document.querySelectorAll(
    'input[name="data-mode"]'
);

function getSelectedMode() {
    for (const checkbox of modeCheckboxes) {
        if (checkbox.checked) {
            return checkbox.value;
        }
    }
    return "anomaly";
}

function clone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

function getIndexName(variable) {
    return climateData.metadata.index_names[variable];
}

function buildFigure(variable, mode) {
    const template = climateData.templates[mode];
    const traceValues = climateData.values[mode][variable];

    const data = template.traces.map((trace, i) => {
        const t = clone(trace);
        const v = traceValues[i];
        t.y = v.y;

        if (trace.meta === "latest_point") {
            t.x = v.x;
            t.text = v.text;
        }
        return t;
    });

    const layout = clone(template.layout);
    const name = getIndexName(variable);

    if (mode === "anomaly") {
        layout.title.text = `Daily ${name} anomaly relative to 1991–2020`;
    } else if (mode === "detrended") {
        layout.title.text =
            `Daily ${name} detrended anomaly relative to 1991–2020 ` +
            `(detrended over 1982–2025)`;
    } else {
        layout.title.text = `Daily ${name}`;
    }

    return {data, layout};
}

function getFilename() {
    const variable = indexSelect.value;
    const mode = getSelectedMode();

    const indexName = getIndexName(variable)
        .replace(/[^a-zA-Z0-9]+/g, "_")
        .replace(/^_|_$/g, "");

    return `ClimateMonitor_${indexName}_${mode}`;
}

function showPlot() {
    if (!climateData) {
        return;
    }

    const variable = indexSelect.value;
    const mode = getSelectedMode();
    const fig = buildFigure(variable, mode);

    Plotly.react(
        plotDiv,
        fig.data,
        fig.layout,
        {
            responsive: true,
            displaylogo: false,
            modeBarButtonsToAdd: [
                {
                    name: "Download PNG",
                    icon: Plotly.Icons.camera,
                    click: function(gd) {
                        Plotly.downloadImage(gd, {
                            format: "png",
                            filename: getFilename(),
                            scale: 4
                        });
                    }
                },
                {
                    name: "Download JPEG",
                    icon: Plotly.Icons.camera,
                    click: function(gd) {
                        Plotly.downloadImage(gd, {
                            format: "jpeg",
                            filename: getFilename(),
                            scale: 4
                        });
                    }
                },
                {
                    name: "Download SVG",
                    icon: Plotly.Icons.disk,
                    click: function(gd) {
                        Plotly.downloadImage(gd, {
                            format: "svg",
                            filename: getFilename()
                        });
                    }
                }
            ]
        }
    );
}

indexSelect.addEventListener("change", showPlot);

modeCheckboxes.forEach(checkbox => {
    checkbox.addEventListener("change", function() {

        if (this.checked) {
            modeCheckboxes.forEach(other => {
                if (other !== this) {
                    other.checked = false;
                }
            });
        } else {
            const anyChecked = Array.from(modeCheckboxes)
                .some(box => box.checked);

            if (!anyChecked) {
                this.checked = true;
            }
        }

        showPlot();
    });
});

fetch("__JSON_FILE__")
    .then(response => {
        if (!response.ok) {
            throw new Error(
                `HTTP ${response.status}: ${response.statusText}`
            );
        }
        return response.json();
    })
    .then(data => {
        climateData = data;
        loadingDiv.style.display = "none";
        showPlot();
    })
    .catch(error => {
        console.error("Climate data loading error:", error);
        loadingDiv.textContent =
            "Unable to load climate data. Check the browser console.";
    });

</script>

</body>
</html>
"""

html = html.replace("__INDEX_OPTIONS__", index_options)
html = html.replace("__JSON_FILE__", json_file)

with open("index.html", "w", encoding="utf-8") as f:
    f.write(html)

print("Saved index.html")
print("Done.")