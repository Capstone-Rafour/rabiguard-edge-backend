/* app.js - Live Telemetry, Interactive Canvas Drawing, and Alerts Feed */

document.addEventListener("DOMContentLoaded", () => {
    // --- UI Elements ---
    const fpsVal = document.getElementById("fps-val");
    const zonesVal = document.getElementById("zones-val");
    const tracksVal = document.getElementById("tracks-val");
    const sysStatus = document.querySelector(".status-indicator");
    const telemetryBody = document.getElementById("telemetry-body");
    const zonesBody = document.getElementById("zones-body");
    const alertsContainer = document.getElementById("alerts-container");
    const btnClearAlerts = document.getElementById("btn-clear-alerts");

    // Scanning Elements
    const btnScan = document.getElementById("btn-scan");
    const scanStatusBadge = document.getElementById("scan-status-badge");
    const scanOverlay = document.getElementById("scan-overlay");

    // Canvas Drawing Elements
    const canvas = document.getElementById("drawing-canvas");
    const ctx = canvas.getContext("2d");
    const videoStream = document.getElementById("video-stream");
    
    const btnDraw = document.getElementById("btn-draw");
    const btnClearDraw = document.getElementById("btn-clear-draw");
    const btnSave = document.getElementById("btn-save");
    
    // Modal Elements
    const zoneModal = document.getElementById("zone-modal");
    const inputZoneName = document.getElementById("zone-name");
    const inputZoneThreshold = document.getElementById("zone-threshold");
    const btnModalCancel = document.getElementById("btn-modal-cancel");
    const btnModalSave = document.getElementById("btn-modal-save");

    // Drawing State variables
    let isDrawingMode = false;
    let drawnPoints = []; // List of [x, y] in canvas coordinates
    let isScanningActive = false;

    // --- Adjust Canvas Bounds to match Video Element ---
    function resizeCanvas() {
        const rect = videoStream.getBoundingClientRect();
        canvas.width = rect.width;
        canvas.height = rect.height;
        canvas.style.top = `${videoStream.offsetTop}px`;
        canvas.style.left = `${videoStream.offsetLeft}px`;
        drawCurrentPolygon();
    }

    // Monitor resize and stream load events
    videoStream.addEventListener("load", resizeCanvas);
    window.addEventListener("resize", resizeCanvas);
    // Initial delay resize to ensure bounding box has rendered
    setTimeout(resizeCanvas, 500);

    // --- Interactive Polygon Drawing Logic ---
    btnDraw.addEventListener("click", () => {
        isDrawingMode = !isDrawingMode;
        if (isDrawingMode) {
            btnDraw.textContent = "🛑 Stop Drawing";
            btnDraw.classList.remove("btn-primary");
            btnDraw.classList.add("btn-secondary");
            canvas.style.pointerEvents = "auto";
            resizeCanvas();
        } else {
            resetDrawing();
        }
    });

    canvas.addEventListener("click", (e) => {
        if (!isDrawingMode) return;
        
        const rect = canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;
        
        drawnPoints.push([x, y]);
        
        btnClearDraw.disabled = false;
        if (drawnPoints.length >= 3) {
            btnSave.disabled = false;
        }
        
        drawCurrentPolygon();
    });

    function drawCurrentPolygon() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (drawnPoints.length === 0) return;
        
        // Draw lines
        ctx.beginPath();
        ctx.moveTo(drawnPoints[0][0], drawnPoints[0][1]);
        for (let i = 1; i < drawnPoints.length; i++) {
            ctx.lineTo(drawnPoints[i][0], drawnPoints[i][1]);
        }
        
        if (isDrawingMode) {
            ctx.strokeStyle = "#ffd600"; // Glowing yellow line while drawing
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        
        // Draw dots at vertices
        drawnPoints.forEach(([px, py]) => {
            ctx.beginPath();
            ctx.arc(px, py, 5, 0, 2 * Math.PI);
            ctx.fillStyle = "#ffd600";
            ctx.fill();
            ctx.lineWidth = 1;
            ctx.strokeStyle = "#ffffff";
            ctx.stroke();
        });
    }

    btnClearDraw.addEventListener("click", () => {
        drawnPoints = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        btnClearDraw.disabled = true;
        btnSave.disabled = true;
    });

    function resetDrawing() {
        isDrawingMode = false;
        drawnPoints = [];
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        btnDraw.textContent = "➕ Draw Zone";
        btnDraw.classList.remove("btn-secondary");
        btnDraw.classList.add("btn-primary");
        btnClearDraw.disabled = true;
        btnSave.disabled = true;
        canvas.style.pointerEvents = "none";
    }

    // --- Zone Management Logic ---
    async function fetchAndRenderZones() {
        try {
            const response = await fetch("/api/zones");
            const zones = await response.json();
            
            if (Object.keys(zones).length === 0) {
                zonesBody.innerHTML = `
                    <tr class="empty-row">
                        <td colspan="4">No zones configured. Use the 'Draw Zone' tool to create one.</td>
                    </tr>
                `;
                return;
            }
            
            let html = "";
            for (const [zoneId, data] of Object.entries(zones)) {
                const isAuto = data.class_name === "auto";
                const statusClass = data.is_active ? "status-pill safe" : "status-pill";
                const statusText = isAuto ? "AUTO" : (data.is_active ? "ACTIVE" : "INACTIVE");
                
                html += `
                    <tr>
                        <td><strong>${zoneId}</strong></td>
                        <td>${data.enter_threshold_sec}s</td>
                        <td><span class="${statusClass}">${statusText}</span></td>
                        <td>
                            <button class="btn btn-text btn-delete-zone" data-id="${zoneId}" style="padding: 2px 8px; color: #ff5252;">
                                🗑️ Delete
                            </button>
                        </td>
                    </tr>
                `;
            }
            zonesBody.innerHTML = html;
            
            // Add delete event listeners
            document.querySelectorAll(".btn-delete-zone").forEach(btn => {
                btn.addEventListener("click", () => deleteZone(btn.dataset.id));
            });
        } catch (err) {
            console.error("Error fetching zones:", err);
        }
    }

    async function deleteZone(zoneId) {
        if (!confirm(`Are you sure you want to delete zone '${zoneId}'?`)) return;
        
        try {
            const response = await fetch(`/api/zones/${zoneId}`, { method: "DELETE" });
            const data = await response.json();
            if (data.success) {
                fetchAndRenderZones();
            } else {
                alert(`Failed to delete zone: ${data.error}`);
            }
        } catch (err) {
            console.error("Error deleting zone:", err);
        }
    }

    // --- Scanning Logic ---
    if (btnScan) {
        btnScan.addEventListener("click", async () => {
            if (isScanningActive) return;

            try {
                const response = await fetch("/api/scan", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ duration: 3.0 })
                });
                const data = await response.json();
                
                if (data.success) {
                    console.log("Scan started...");
                    isScanningActive = true;
                    btnScan.disabled = true;
                    btnScan.textContent = "⌛ Scanning...";
                    scanOverlay.classList.remove("hidden");
                    scanStatusBadge.textContent = "SCANNING";
                    scanStatusBadge.className = "badge badge-purple pulse";
                    
                    // Wait for scan duration + buffer
                    setTimeout(async () => {
                        isScanningActive = false;
                        btnScan.disabled = false;
                        btnScan.textContent = "🔍 Scan Room (Auto-ROI)";
                        scanOverlay.classList.add("hidden");
                        scanStatusBadge.textContent = "IDLE";
                        scanStatusBadge.className = "badge badge-purple";
                        
                        // Refresh zone list to show new auto-zones
                        await fetchAndRenderZones();
                    }, (data.duration + 0.5) * 1000);
                }
            } catch (err) {
                console.error("Error triggering scan:", err);
                alert("Failed to start scan.");
            }
        });
    }

    // Initial load
    fetchAndRenderZones();

    // --- Modal Configuration Handlers ---
    btnSave.addEventListener("click", () => {
        zoneModal.classList.remove("hidden");
        inputZoneName.focus();
    });

    btnModalCancel.addEventListener("click", () => {
        zoneModal.classList.add("hidden");
    });

    btnModalSave.addEventListener("click", async () => {
        const zoneId = inputZoneName.value.trim();
        const threshold = parseFloat(inputZoneThreshold.value);
        
        if (!zoneId) {
            alert("Please specify a valid Zone ID.");
            return;
        }

        // Map drawn canvas coordinates back to original 640x480 resolution coordinates
        const rect = canvas.getBoundingClientRect();
        const scaleX = 640 / rect.width;
        const scaleY = 480 / rect.height;
        
        const originalPolygon = drawnPoints.map(([cx, cy]) => {
            return [
                Math.round(cx * scaleX),
                Math.round(cy * scaleY)
            ];
        });

        // Submit zone via POST request
        try {
            const response = await fetch("/api/zones", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    zone_id: zoneId,
                    polygon: originalPolygon,
                    enter_threshold_sec: threshold,
                    is_active: true
                })
            });

            const data = await response.json();
            if (data.success) {
                console.log(`Zone '${zoneId}' successfully added.`);
                zoneModal.classList.add("hidden");
                resetDrawing();
                fetchAndRenderZones(); // Refresh list
            } else {
                alert(`Error saving zone: ${data.error}`);
            }
        } catch (err) {
            console.error("Network error saving zone:", err);
            alert("Network error: Failed to save zone.");
        }
    });

    // --- Connect to Server-Sent Events (SSE) ---
    console.log("Establishing SSE connection...");
    const eventSource = new EventSource("/api/events");

    eventSource.addEventListener("telemetry", (event) => {
        const data = JSON.parse(event.data);
        
        // 1. Update Header Metrics
        fpsVal.textContent = data.fps;
        zonesVal.textContent = data.active_zones_count;
        tracksVal.textContent = data.current_tracks_count;
        
        // 2. Determine and Update Global Security State
        let hasActiveIntrusion = false;
        
        // 3. Render Live Tracks Console Table
        if (data.tracks.length === 0) {
            telemetryBody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="7">No active tracking targets in the scene.</td>
                </tr>
            `;
        } else {
            let rowsHtml = "";
            data.tracks.forEach((track) => {
                const isIntrusion = track.status === "INTRUSION";
                if (isIntrusion) hasActiveIntrusion = true;
                
                const statusBadgeClass = `status-pill ${track.status.toLowerCase()}`;
                
                rowsHtml += `
                    <tr>
                        <td><strong>#${track.id}</strong></td>
                        <td><span class="badge">${track.zone_id}</span></td>
                        <td>${track.p_depth.toFixed(2)}m</td>
                        <td>${track.z_depth > 0 ? track.z_depth.toFixed(2) + 'm' : '-'}</td>
                        <td>${track.z_depth > 0 ? track.diff.toFixed(2) + 'm' : '-'}</td>
                        <td><span class="${statusBadgeClass}">${track.status}</span></td>
                        <td>${track.elapsed.toFixed(1)}s</td>
                    </tr>
                `;
            });
            telemetryBody.innerHTML = rowsHtml;
        }

        // Adjust global warning indicator
        if (hasActiveIntrusion) {
            sysStatus.textContent = "INTRUSION ALARM";
            sysStatus.className = "status-indicator warning";
        } else {
            sysStatus.textContent = "SECURED";
            sysStatus.className = "status-indicator secured";
        }

        // 4. Update scanning state if UI missed it
        if (data.is_scanning && !isScanningActive) {
             isScanningActive = true;
             btnScan.disabled = true;
             btnScan.textContent = "⌛ Scanning...";
             scanOverlay.classList.remove("hidden");
             scanStatusBadge.textContent = "SCANNING";
             scanStatusBadge.className = "badge badge-purple pulse";
        }
    });

    eventSource.addEventListener("alert", (event) => {
        const alert = JSON.parse(event.data);
        console.log("🚨 Intrusion Event Triggered!", alert);
        
        // Remove no-alerts placeholder if it exists
        const placeholder = document.querySelector(".no-alerts");
        if (placeholder) {
            placeholder.remove();
        }
        
        // Create Alert Card
        const alertCard = document.createElement("div");
        alertCard.className = "alert-item";
        alertCard.innerHTML = `
            <div class="alert-visual">
                <img src="${alert.image_url}" alt="Intrusion Snapshot">
                <span class="alert-badge">BREACH</span>
            </div>
            <div class="alert-meta">
                <div class="alert-title">
                    <h3>${alert.zone_id} BREACHED</h3>
                    <span>${alert.timestamp.split(" ")[1]}</span>
                </div>
                <div class="alert-metrics">
                    <div class="alert-stat">
                        <span class="label">PERSON DEPTH</span>
                        <span class="value">${alert.p_depth.toFixed(2)}m</span>
                    </div>
                    <div class="alert-stat">
                        <span class="label">ZONE DEPTH</span>
                        <span class="value">${alert.z_depth.toFixed(2)}m</span>
                    </div>
                    <div class="alert-stat diff-stat">
                        <span class="label">SPATIAL DIFF</span>
                        <span class="value">±${alert.diff.toFixed(2)}m</span>
                    </div>
                    <div class="alert-stat">
                        <span class="label">TRACK TARGET</span>
                        <span class="value">ID #${alert.track_id}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Prepend card to top of alerts panel
        alertsContainer.insertBefore(alertCard, alertsContainer.firstChild);
        
        // Dynamic sound or notification (optional visual flash)
        document.body.style.animation = "alert-flash 0.3s ease-out";
        setTimeout(() => {
            document.body.style.animation = "";
        }, 300);
    });

    btnClearAlerts.addEventListener("click", () => {
        alertsContainer.innerHTML = `
            <div class="no-alerts">
                <div class="no-alerts-icon">🛡️</div>
                <h3>No Intrusion Alerts</h3>
                <p>Spatial depth verification is currently monitoring for active breaches.</p>
            </div>
        `;
    });

    // Custom animation definition inside CSS was triggered via js
    const styleSheet = document.createElement("style");
    styleSheet.type = "text/css";
    styleSheet.innerText = `
        @keyframes alert-flash {
            0% { background-color: transparent; }
            50% { background-color: rgba(255, 23, 68, 0.15); }
            100% { background-color: transparent; }
        }
        .pulse { animation: blink-pill 1s infinite alternate; }
    `;
    document.head.appendChild(styleSheet);
});
