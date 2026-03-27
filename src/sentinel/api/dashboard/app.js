/**
 * Sentinel Dashboard -- frontend JavaScript.
 *
 * Provides API helpers, file upload handling, dataset exploration
 * (Chart.js), PI System connector integration, and prompt chat.
 */

const API_BASE = "/api";

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

/**
 * Perform a JSON API request.
 * @param {string} path - API path (relative to /api).
 * @param {object} options - fetch options.
 * @returns {Promise<object>} Parsed JSON response.
 */
async function apiFetch(path, options) {
    options = options || {};
    const url = API_BASE + path;
    const resp = await fetch(url, options);
    if (!resp.ok) {
        let detail = resp.statusText;
        try {
            const body = await resp.json();
            detail = body.detail || body.error || detail;
        } catch (_) {}
        throw new Error(detail);
    }
    return resp.json();
}

/**
 * Show an alert message in the given container element.
 * @param {string} containerId - ID of the alert container div.
 * @param {string} message - Alert text.
 * @param {string} type - One of: success, error, warning, info.
 */
function showAlert(containerId, message, type) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML =
        '<div class="alert alert-' + type + '">' + message + "</div>";
}

/** Clear an alert container. */
function clearAlert(containerId) {
    const el = document.getElementById(containerId);
    if (el) el.innerHTML = "";
}

/** Format bytes to human-readable string. */
function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

/** Assign a set of colors for chart lines. */
const CHART_COLORS = [
    "#38bdf8", "#4ade80", "#fbbf24", "#f87171", "#a78bfa",
    "#fb923c", "#2dd4bf", "#e879f9", "#60a5fa", "#34d399",
];

// ---------------------------------------------------------------------------
// Dashboard page (index.html)
// ---------------------------------------------------------------------------

async function loadDashboard() {
    // Health check.
    try {
        const health = await apiFetch("/../health");
        document.getElementById("api-status").textContent = health.status;
        document.getElementById("api-status-detail").textContent =
            "API: " + health.api + " | Ollama: " + health.ollama;
    } catch (err) {
        document.getElementById("api-status").textContent = "error";
        document.getElementById("api-status-detail").textContent = err.message;
    }

    // Datasets.
    try {
        const data = await apiFetch("/data?page=1&limit=100");
        const items = data.items || [];
        document.getElementById("dataset-count").textContent = data.total || items.length;
        document.getElementById("datasets-loading").classList.add("hidden");

        if (items.length === 0) {
            document.getElementById("datasets-empty").classList.remove("hidden");
        } else {
            const table = document.getElementById("datasets-table");
            table.classList.remove("hidden");
            const body = document.getElementById("datasets-body");
            body.innerHTML = "";
            for (const ds of items) {
                const tr = document.createElement("tr");
                const shape = ds.shape
                    ? ds.shape[0] + " x " + ds.shape[1]
                    : ds.rows + " x " + ds.columns;
                tr.innerHTML =
                    "<td>" + (ds.dataset_id || "").substring(0, 12) + "...</td>" +
                    "<td>" + (ds.name || "") + "</td>" +
                    "<td>" + shape + "</td>" +
                    "<td>" + (ds.source || "") + "</td>" +
                    "<td>" + (ds.uploaded_at || "").substring(0, 19) + "</td>" +
                    '<td><a href="/ui/explore.html?id=' + ds.dataset_id +
                    '" class="btn btn-secondary" style="padding:0.25rem 0.5rem;font-size:0.8rem;">Explore</a></td>';
                body.appendChild(tr);
            }
        }
    } catch (err) {
        document.getElementById("datasets-loading").classList.add("hidden");
        showAlert("status-alert", "Failed to load datasets: " + err.message, "error");
    }

    // Models.
    try {
        const data = await apiFetch("/models");
        const models = data.models || [];
        document.getElementById("model-count").textContent = models.length;
        document.getElementById("models-loading").classList.add("hidden");

        if (models.length > 0) {
            const table = document.getElementById("models-table");
            table.classList.remove("hidden");
            const body = document.getElementById("models-body");
            body.innerHTML = "";
            for (const m of models) {
                const tr = document.createElement("tr");
                tr.innerHTML =
                    "<td>" + m.name + "</td>" +
                    "<td>" + (m.category || "") + "</td>" +
                    "<td>" + (m.description || "") + "</td>";
                body.appendChild(tr);
            }
        }
    } catch (err) {
        document.getElementById("models-loading").classList.add("hidden");
        showAlert("status-alert", "Failed to load models: " + err.message, "warning");
    }
}

// ---------------------------------------------------------------------------
// Upload page (upload.html)
// ---------------------------------------------------------------------------

function initUploadPage() {
    const area = document.getElementById("upload-area");
    const fileInput = document.getElementById("file-input");
    const uploadBtn = document.getElementById("upload-btn");
    let selectedFile = null;

    if (!area) return;

    // Click to browse.
    area.addEventListener("click", function () {
        fileInput.click();
    });

    // Drag and drop.
    area.addEventListener("dragover", function (e) {
        e.preventDefault();
        area.classList.add("dragover");
    });
    area.addEventListener("dragleave", function () {
        area.classList.remove("dragover");
    });
    area.addEventListener("drop", function (e) {
        e.preventDefault();
        area.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            selectFile(fileInput.files[0]);
        }
    });

    function selectFile(file) {
        selectedFile = file;
        document.getElementById("file-name").textContent = file.name;
        document.getElementById("file-size").textContent = formatBytes(file.size);
        document.getElementById("file-info").classList.remove("hidden");
        uploadBtn.disabled = false;
        clearAlert("upload-alert");

        // Check size limit (100 MB).
        if (file.size > 100 * 1024 * 1024) {
            showAlert("upload-alert", "File exceeds 100 MB limit.", "error");
            uploadBtn.disabled = true;
        }
    }

    uploadBtn.addEventListener("click", async function () {
        if (!selectedFile) return;
        uploadBtn.disabled = true;
        document.getElementById("upload-spinner").classList.remove("hidden");
        clearAlert("upload-alert");

        const formData = new FormData();
        formData.append("file", selectedFile);

        try {
            const result = await fetch(API_BASE + "/data/upload", {
                method: "POST",
                body: formData,
            });
            if (!result.ok) {
                let detail = result.statusText;
                try {
                    const body = await result.json();
                    detail = body.detail || body.error || detail;
                } catch (_) {}
                throw new Error(detail);
            }
            const data = await result.json();

            // Show result.
            document.getElementById("upload-result").classList.remove("hidden");
            const body = document.getElementById("result-body");
            const shape = data.shape ? data.shape[0] + " x " + data.shape[1] : "N/A";
            const features = (data.features || []).join(", ");
            const timeRange = data.time_range
                ? (data.time_range.start || "") + " to " + (data.time_range.end || "")
                : "N/A";
            body.innerHTML =
                "<tr><td>Dataset ID</td><td>" + data.dataset_id + "</td></tr>" +
                "<tr><td>Shape</td><td>" + shape + "</td></tr>" +
                "<tr><td>Features</td><td>" + features + "</td></tr>" +
                "<tr><td>Time Range</td><td>" + timeRange + "</td></tr>";

            document.getElementById("explore-link").href =
                "/ui/explore.html?id=" + data.dataset_id;

            showAlert("upload-alert", "Upload successful!", "success");
        } catch (err) {
            showAlert("upload-alert", "Upload failed: " + err.message, "error");
        } finally {
            uploadBtn.disabled = false;
            document.getElementById("upload-spinner").classList.add("hidden");
        }
    });
}

// ---------------------------------------------------------------------------
// Explore page (explore.html)
// ---------------------------------------------------------------------------

async function initExplorePage() {
    const params = new URLSearchParams(window.location.search);
    const datasetId = params.get("id");

    if (!datasetId) {
        const el = document.getElementById("no-dataset");
        if (el) el.classList.remove("hidden");
        return;
    }

    // Load dataset info.
    try {
        const info = await apiFetch("/data/" + datasetId);
        document.getElementById("dataset-info").classList.remove("hidden");
        document.getElementById("info-id").textContent =
            info.dataset_id || datasetId;
        const shape = info.shape
            ? info.shape[0] + " rows x " + info.shape[1] + " cols"
            : (info.rows || 0) + " rows x " + (info.columns || 0) + " cols";
        document.getElementById("info-shape").textContent = shape;
        const features = info.features || [];
        document.getElementById("info-features").textContent =
            features.join(", ") || "N/A";
        const tr = info.time_range || {};
        document.getElementById("info-range").textContent =
            (tr.start || "N/A") + " to " + (tr.end || "N/A");
    } catch (err) {
        showAlert("explore-alert", "Failed to load dataset info: " + err.message, "error");
        return;
    }

    // Load preview data.
    try {
        const preview = await apiFetch("/data/" + datasetId + "/preview?rows=20");
        const rows = preview.rows || preview || [];
        if (rows.length > 0) {
            document.getElementById("preview-card").classList.remove("hidden");
            const cols = Object.keys(rows[0]);
            const head = document.getElementById("preview-head");
            head.innerHTML = "<tr>" + cols.map(function(c) {
                return "<th>" + c + "</th>";
            }).join("") + "</tr>";
            const body = document.getElementById("preview-body");
            body.innerHTML = "";
            for (const row of rows) {
                const tr = document.createElement("tr");
                tr.innerHTML = cols.map(function(c) {
                    const v = row[c];
                    return "<td>" + (v !== null && v !== undefined ? v : "") + "</td>";
                }).join("");
                body.appendChild(tr);
            }
        }
    } catch (err) {
        // Preview may not be available; non-critical.
    }

    // Load chart data.
    try {
        const plotData = await apiFetch(
            "/data/" + datasetId + "/plot?max_points=2000"
        );
        const points = plotData.rows || plotData || [];
        if (points.length > 0 && typeof Chart !== "undefined") {
            document.getElementById("chart-card").classList.remove("hidden");
            const labels = points.map(function(p) { return p.timestamp || ""; });
            const featureKeys = Object.keys(points[0]).filter(function(k) {
                return k !== "timestamp" && k !== "is_anomaly";
            });

            const datasets = featureKeys.map(function(key, i) {
                return {
                    label: key,
                    data: points.map(function(p) { return p[key]; }),
                    borderColor: CHART_COLORS[i % CHART_COLORS.length],
                    backgroundColor: "transparent",
                    borderWidth: 1.5,
                    pointRadius: 0,
                    tension: 0.1,
                };
            });

            const ctx = document.getElementById("timeseries-chart").getContext("2d");
            new Chart(ctx, {
                type: "line",
                data: { labels: labels, datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    interaction: { mode: "index", intersect: false },
                    plugins: {
                        legend: {
                            labels: { color: "#f1f5f9", font: { size: 11 } },
                        },
                    },
                    scales: {
                        x: {
                            ticks: { color: "#94a3b8", maxTicksLimit: 12 },
                            grid: { color: "rgba(71,85,105,0.3)" },
                        },
                        y: {
                            ticks: { color: "#94a3b8" },
                            grid: { color: "rgba(71,85,105,0.3)" },
                        },
                    },
                },
            });
        }
    } catch (err) {
        // Chart data may not be available; non-critical.
    }
}

// ---------------------------------------------------------------------------
// PI System page (pi.html)
// ---------------------------------------------------------------------------

function initPIPage() {
    const searchBtn = document.getElementById("pi-search-btn");
    const fetchBtn = document.getElementById("pi-fetch-btn");

    if (!searchBtn) return;

    // Search tags.
    searchBtn.addEventListener("click", async function () {
        const server = document.getElementById("pi-server").value.trim();
        const pattern = document.getElementById("pi-search-pattern").value.trim();
        const port = parseInt(document.getElementById("pi-port").value) || 5450;
        const timeout = parseInt(document.getElementById("pi-timeout").value) || 30;

        if (!server || !pattern) {
            showAlert("pi-alert", "Server and pattern are required.", "warning");
            return;
        }

        clearAlert("pi-alert");
        searchBtn.disabled = true;
        document.getElementById("pi-search-spinner").classList.remove("hidden");

        try {
            const result = await apiFetch("/data/pi-search", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    server: server,
                    pattern: pattern,
                    port: port,
                    timeout: timeout,
                }),
            });

            const tags = result.tags || [];
            const container = document.getElementById("pi-tags-list");
            container.innerHTML = "";

            if (tags.length === 0) {
                container.innerHTML =
                    '<p style="color:var(--text-secondary);">No tags found.</p>';
            } else {
                for (const tag of tags) {
                    const label = document.createElement("label");
                    label.style.display = "block";
                    label.style.padding = "0.25rem 0";
                    label.style.cursor = "pointer";
                    label.innerHTML =
                        '<input type="checkbox" value="' + tag.name +
                        '" style="margin-right:0.5rem;">' +
                        '<span style="color:var(--accent);">' + tag.name + "</span>" +
                        (tag.description
                            ? ' <span style="color:var(--text-secondary); font-size:0.8rem;">-- ' +
                              tag.description + "</span>"
                            : "") +
                        (tag.uom
                            ? ' <span style="color:var(--text-secondary); font-size:0.75rem;">(' +
                              tag.uom + ")</span>"
                            : "");
                    container.appendChild(label);
                }
            }

            document.getElementById("pi-search-results").classList.remove("hidden");

            // Wire up checkbox selection to populate fetch tags input.
            container.addEventListener("change", function () {
                const checked = container.querySelectorAll("input:checked");
                const names = Array.from(checked).map(function (cb) {
                    return cb.value;
                });
                document.getElementById("pi-tags-input").value = names.join(", ");
            });
        } catch (err) {
            showAlert("pi-alert", "Tag search failed: " + err.message, "error");
        } finally {
            searchBtn.disabled = false;
            document.getElementById("pi-search-spinner").classList.add("hidden");
        }
    });

    // Fetch data.
    fetchBtn.addEventListener("click", async function () {
        const server = document.getElementById("pi-server").value.trim();
        const tagsInput = document.getElementById("pi-tags-input").value.trim();
        const start = document.getElementById("pi-start").value.trim();
        const end = document.getElementById("pi-end").value.trim();
        const interval = document.getElementById("pi-interval").value.trim();
        const port = parseInt(document.getElementById("pi-port").value) || 5450;
        const timeout = parseInt(document.getElementById("pi-timeout").value) || 30;

        if (!server || !tagsInput) {
            showAlert("pi-alert", "Server and at least one tag are required.", "warning");
            return;
        }

        const tags = tagsInput.split(",").map(function (t) { return t.trim(); })
            .filter(function (t) { return t; });

        clearAlert("pi-alert");
        fetchBtn.disabled = true;
        document.getElementById("pi-fetch-spinner").classList.remove("hidden");

        try {
            const result = await apiFetch("/data/pi-fetch", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    server: server,
                    tags: tags,
                    start: start || "*-1d",
                    end: end || "*",
                    interval: interval || "5m",
                    port: port,
                    timeout: timeout,
                }),
            });

            document.getElementById("pi-fetch-result").classList.remove("hidden");
            const body = document.getElementById("pi-result-body");
            const shape = result.shape ? result.shape[0] + " x " + result.shape[1] : "N/A";
            const features = (result.features || []).join(", ");
            const tr = result.time_range || {};
            body.innerHTML =
                "<tr><td>Dataset ID</td><td>" + result.dataset_id + "</td></tr>" +
                "<tr><td>Shape</td><td>" + shape + "</td></tr>" +
                "<tr><td>Features</td><td>" + features + "</td></tr>" +
                "<tr><td>Time Range</td><td>" +
                (tr.start || "") + " to " + (tr.end || "") + "</td></tr>";

            document.getElementById("pi-explore-link").href =
                "/ui/explore.html?id=" + result.dataset_id;

            showAlert("pi-alert", "PI data fetched and ingested successfully!", "success");
        } catch (err) {
            showAlert("pi-alert", "PI fetch failed: " + err.message, "error");
        } finally {
            fetchBtn.disabled = false;
            document.getElementById("pi-fetch-spinner").classList.add("hidden");
        }
    });
}

// ---------------------------------------------------------------------------
// Prompt page (prompt.html)
// ---------------------------------------------------------------------------

function initPromptPage() {
    const input = document.getElementById("prompt-input");
    const sendBtn = document.getElementById("prompt-send");
    const messages = document.getElementById("chat-messages");

    if (!input || !sendBtn) return;

    async function sendPrompt() {
        const text = input.value.trim();
        if (!text) return;

        // Add user message.
        const userDiv = document.createElement("div");
        userDiv.className = "chat-message user";
        userDiv.textContent = text;
        messages.appendChild(userDiv);
        input.value = "";
        messages.scrollTop = messages.scrollHeight;

        sendBtn.disabled = true;
        clearAlert("prompt-alert");

        try {
            const result = await apiFetch("/prompt", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: text }),
            });

            const assistantDiv = document.createElement("div");
            assistantDiv.className = "chat-message assistant";
            assistantDiv.textContent = result.response || "No response.";

            if (result.tools_called && result.tools_called.length > 0) {
                const toolInfo = document.createElement("p");
                toolInfo.style.fontSize = "0.75rem";
                toolInfo.style.color = "var(--text-secondary)";
                toolInfo.style.marginTop = "0.5rem";
                toolInfo.textContent = "Tools used: " + result.tools_called.join(", ");
                assistantDiv.appendChild(toolInfo);
            }

            messages.appendChild(assistantDiv);
        } catch (err) {
            const errDiv = document.createElement("div");
            errDiv.className = "chat-message assistant";
            errDiv.style.borderLeft = "3px solid var(--danger)";
            errDiv.textContent = "Error: " + err.message;
            messages.appendChild(errDiv);
        } finally {
            sendBtn.disabled = false;
            messages.scrollTop = messages.scrollHeight;
            input.focus();
        }
    }

    sendBtn.addEventListener("click", sendPrompt);
    input.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            sendPrompt();
        }
    });
}
