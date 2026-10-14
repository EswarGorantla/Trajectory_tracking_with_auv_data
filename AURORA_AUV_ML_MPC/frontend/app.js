// AURORA AUV ML-MPC Dashboard Application

let cachedSequenceHistory = null;
let cachedActualNext = null;

// Chart Instances
let predictChartInstance = null;
let seqChartXYInstance = null;
let seqChartDepthInstance = null;
let mpcChartXYInstance = null;
let mpcChartDepthInstance = null;
let uploadChartXYInstance = null;
let uploadChartDepthInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    checkHealth();
    initCharts();
});

// Tab Switcher
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    
    event.target.classList.add("active");
    document.getElementById(tabId).classList.add("active");
}

// 1. Health Check
async function checkHealth() {
    const statusPill = document.getElementById("system-status");
    const statusText = document.getElementById("status-text");
    try {
        const res = await fetch("/health");
        const data = await res.json();
        if (data.status === "healthy" && data.model_loaded) {
            statusPill.className = "status-pill status-online";
            statusText.innerText = "Model Loaded";
            document.getElementById("model-tag").innerText = `Model: ${data.model_type}`;
            document.getElementById("device-tag").innerText = `Device: ${data.device}`;
        } else {
            statusPill.className = "status-pill status-error";
            statusText.innerText = "Model Error";
        }
    } catch (err) {
        statusPill.className = "status-pill status-error";
        statusText.innerText = "Offline";
        console.error("Health check failed:", err);
    }
}

// Clear Custom State Form
function clearCustomState() {
    document.getElementById("in-north").value = "0.00";
    document.getElementById("in-east").value = "0.00";
    document.getElementById("in-depth").value = "20.00";
    document.getElementById("in-speed").value = "1.20";
    document.getElementById("in-pitch").value = "0.00";
    document.getElementById("in-heading").value = "0.00";
    document.getElementById("in-roll").value = "0.00";

    document.getElementById("cmd-speed").value = "1.20";
    document.getElementById("cmd-pitch").value = "0.00";
    document.getElementById("cmd-heading").value = "0.00";

    cachedSequenceHistory = null;
    cachedActualNext = null;

    document.getElementById("res-act-north").innerText = "-";
    document.getElementById("res-act-east").innerText = "-";
    document.getElementById("res-act-depth").innerText = "-";
    document.getElementById("res-act-speed").innerText = "-";
    document.getElementById("res-act-heading").innerText = "-";
    document.getElementById("improvement-box").style.display = "none";
}

// 2. Load Real AURORA Test Sample
async function loadAURORASample() {
    const btn = document.getElementById("btn-load-sample");
    btn.innerText = "⏳ Loading...";
    btn.disabled = true;

    try {
        const res = await fetch("/sample");
        const data = await res.json();

        // Populate Current State
        document.getElementById("in-north").value = data.current_state.north.toFixed(2);
        document.getElementById("in-east").value = data.current_state.east.toFixed(2);
        document.getElementById("in-depth").value = data.current_state.depth.toFixed(2);
        document.getElementById("in-speed").value = data.current_state.speed.toFixed(2);
        document.getElementById("in-pitch").value = (data.current_state.pitch * 180 / Math.PI).toFixed(2);
        document.getElementById("in-heading").value = (data.current_state.heading * 180 / Math.PI).toFixed(2);
        document.getElementById("in-roll").value = (data.current_state.roll * 180 / Math.PI).toFixed(2);

        // Populate Commands
        document.getElementById("cmd-speed").value = data.commands.u_cmd.toFixed(2);
        document.getElementById("cmd-pitch").value = (data.commands.theta_cmd * 180 / Math.PI).toFixed(2);
        document.getElementById("cmd-heading").value = (data.commands.psi_cmd * 180 / Math.PI).toFixed(2);

        cachedSequenceHistory = data.sequence_history;
        cachedActualNext = data.actual_next_state;

        // Populate Actual Next column in table
        document.getElementById("res-act-north").innerText = data.actual_next_state.north.toFixed(3);
        document.getElementById("res-act-east").innerText = data.actual_next_state.east.toFixed(3);
        document.getElementById("res-act-depth").innerText = data.actual_next_state.depth.toFixed(3);
        document.getElementById("res-act-speed").innerText = data.actual_next_state.speed.toFixed(3);
        document.getElementById("res-act-heading").innerText = (data.actual_next_state.heading * 180 / Math.PI).toFixed(2);

        btn.innerText = `🎯 Sample #${data.sample_index}`;
        setTimeout(() => { btn.innerText = "🎯 Load Sample"; btn.disabled = false; }, 2000);
    } catch (err) {
        alert("Failed to load sample: " + err);
        btn.innerText = "🎯 Load Sample";
        btn.disabled = false;
    }
}

// 3. Single-Step Prediction
async function runPrediction() {
    const btn = document.getElementById("btn-predict");
    btn.innerText = "⚡ Calculating...";
    btn.disabled = true;

    const north = parseFloat(document.getElementById("in-north").value);
    const east = parseFloat(document.getElementById("in-east").value);
    const depth = parseFloat(document.getElementById("in-depth").value);
    const speed = parseFloat(document.getElementById("in-speed").value);
    const pitch = parseFloat(document.getElementById("in-pitch").value) * Math.PI / 180;
    const heading = parseFloat(document.getElementById("in-heading").value) * Math.PI / 180;
    const roll = parseFloat(document.getElementById("in-roll").value) * Math.PI / 180;

    const u_cmd = parseFloat(document.getElementById("cmd-speed").value);
    const theta_cmd = parseFloat(document.getElementById("cmd-pitch").value) * Math.PI / 180;
    const psi_cmd = parseFloat(document.getElementById("cmd-heading").value) * Math.PI / 180;

    const modelName = document.getElementById("select-model").value;

    const payload = {
        current_state: { north, east, depth, roll, pitch, heading, speed },
        commands: { u_cmd, theta_cmd, psi_cmd },
        sequence_history: cachedSequenceHistory,
        model_name: modelName
    };

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!data.success) throw new Error("Inference failed");

        // Display Execution Time
        document.getElementById("pred-time-badge").innerText = `${data.execution_time_ms} ms (${data.model_used})`;

        // Populate Table Columns
        document.getElementById("res-cur-north").innerText = north.toFixed(3);
        document.getElementById("res-cur-east").innerText = east.toFixed(3);
        document.getElementById("res-cur-depth").innerText = depth.toFixed(3);
        document.getElementById("res-cur-speed").innerText = speed.toFixed(3);
        document.getElementById("res-cur-heading").innerText = (heading * 180 / Math.PI).toFixed(2);

        document.getElementById("res-phy-north").innerText = data.physics_prediction.north.toFixed(3);
        document.getElementById("res-phy-east").innerText = data.physics_prediction.east.toFixed(3);
        document.getElementById("res-phy-depth").innerText = data.physics_prediction.depth.toFixed(3);
        document.getElementById("res-phy-speed").innerText = data.physics_prediction.speed.toFixed(3);
        document.getElementById("res-phy-heading").innerText = (data.physics_prediction.heading * 180 / Math.PI).toFixed(2);

        document.getElementById("res-ai-north").innerText = (data.ai_residual_correction.d_north_m >= 0 ? "+" : "") + data.ai_residual_correction.d_north_m.toFixed(3);
        document.getElementById("res-ai-east").innerText = (data.ai_residual_correction.d_east_m >= 0 ? "+" : "") + data.ai_residual_correction.d_east_m.toFixed(3);
        document.getElementById("res-ai-depth").innerText = (data.ai_residual_correction.d_depth_m >= 0 ? "+" : "") + data.ai_residual_correction.d_depth_m.toFixed(3);
        document.getElementById("res-ai-speed").innerText = (data.ai_residual_correction.d_speed_m_s >= 0 ? "+" : "") + data.ai_residual_correction.d_speed_m_s.toFixed(3);
        document.getElementById("res-ai-heading").innerText = (data.ai_residual_correction.d_heading_rad * 180 / Math.PI >= 0 ? "+" : "") + (data.ai_residual_correction.d_heading_rad * 180 / Math.PI).toFixed(2);

        document.getElementById("res-fin-north").innerText = data.final_prediction.north.toFixed(3);
        document.getElementById("res-fin-east").innerText = data.final_prediction.east.toFixed(3);
        document.getElementById("res-fin-depth").innerText = data.final_prediction.depth.toFixed(3);
        document.getElementById("res-fin-speed").innerText = data.final_prediction.speed.toFixed(3);
        document.getElementById("res-fin-heading").innerText = (data.final_prediction.heading * 180 / Math.PI).toFixed(2);

        // If ground truth is loaded, compute and show error improvement
        if (cachedActualNext) {
            const dPhy = Math.sqrt(
                Math.pow(data.physics_prediction.north - cachedActualNext.north, 2) +
                Math.pow(data.physics_prediction.east - cachedActualNext.east, 2) +
                Math.pow(data.physics_prediction.depth - cachedActualNext.depth, 2)
            );
            const dHyb = Math.sqrt(
                Math.pow(data.final_prediction.north - cachedActualNext.north, 2) +
                Math.pow(data.final_prediction.east - cachedActualNext.east, 2) +
                Math.pow(data.final_prediction.depth - cachedActualNext.depth, 2)
            );
            const imp = ((dPhy - dHyb) / dPhy) * 100;

            document.getElementById("phy-err-val").innerText = `${dPhy.toFixed(3)} m`;
            document.getElementById("hyb-err-val").innerText = `${dHyb.toFixed(3)} m`;
            document.getElementById("imp-pct-val").innerText = `${imp >= 0 ? "+" : ""}${imp.toFixed(1)}%`;
            document.getElementById("improvement-box").style.display = "flex";
        } else {
            document.getElementById("improvement-box").style.display = "none";
        }

        // Update 2D Trajectory Shift Chart
        updatePredictChart(
            { x: east, y: north },
            { x: data.physics_prediction.east, y: data.physics_prediction.north },
            { x: data.final_prediction.east, y: data.final_prediction.north },
            cachedActualNext ? { x: cachedActualNext.east, y: cachedActualNext.north } : null
        );

    } catch (err) {
        alert("Prediction Error: " + err);
    } finally {
        btn.innerText = "⚡ Predict Future State";
        btn.disabled = false;
    }
}

// 4. Batch CSV Upload and Prediction
async function uploadCSVAndPredict() {
    const fileInput = document.getElementById("csv-file-input");
    if (!fileInput.files || fileInput.files.length === 0) {
        alert("Please select a .csv file to upload.");
        return;
    }

    const file = fileInput.files[0];
    const modelName = document.getElementById("upload-model-select").value;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch(`/upload-csv?model_name=${modelName}`, {
            method: "POST",
            body: formData
        });
        const data = await res.json();
        if (!data.success) throw new Error(data.detail || "Failed to process CSV.");

        document.getElementById("upload-results-container").style.display = "block";

        // Populate Table
        const tbody = document.getElementById("upload-table-body");
        tbody.innerHTML = "";
        
        const inputXY = [];
        const predictXY = [];
        const inputDepth = [];
        const predictDepth = [];
        const times = [];

        data.predictions.forEach((row, idx) => {
            times.push(row.time_s);
            inputXY.push({ x: row.input_state.east, y: row.input_state.north });
            predictXY.push({ x: row.final_predicted_state.east, y: row.final_predicted_state.north });
            inputDepth.push(row.input_state.depth);
            predictDepth.push(row.final_predicted_state.depth);

            if (idx < 50) { // Render first 50 rows in DOM table for speed
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>#${row.step}</strong></td>
                    <td>${row.time_s}s</td>
                    <td>(${row.input_state.north}, ${row.input_state.east}, ${row.input_state.depth}m)</td>
                    <td>${row.input_state.speed}m/s | ${row.input_state.heading_deg}°</td>
                    <td>(${row.physics_prediction.north}, ${row.physics_prediction.east}, ${row.physics_prediction.depth}m)</td>
                    <td class="text-accent">ΔX:${row.ai_correction.d_north_m}m, ΔY:${row.ai_correction.d_east_m}m, ΔZ:${row.ai_correction.d_depth_m}m</td>
                    <td class="text-highlight">(${row.final_predicted_state.north}, ${row.final_predicted_state.east}, ${row.final_predicted_state.depth}m)</td>
                `;
                tbody.appendChild(tr);
            }
        });

        // Update Charts
        uploadChartXYInstance.data.datasets[0].data = inputXY;
        uploadChartXYInstance.data.datasets[1].data = predictXY;
        uploadChartXYInstance.update();

        uploadChartDepthInstance.data.labels = times;
        uploadChartDepthInstance.data.datasets[0].data = inputDepth;
        uploadChartDepthInstance.data.datasets[1].data = predictDepth;
        uploadChartDepthInstance.update();

    } catch (err) {
        alert("Upload Error: " + err);
    }
}

// 5. Multi-Step Trajectory Replay
async function runSequenceForecast() {
    const startIdx = parseInt(document.getElementById("seq-start-idx").value);
    const steps = parseInt(document.getElementById("seq-steps").value);

    try {
        const res = await fetch("/predict-sequence", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ start_index: startIdx, steps: steps, model_name: "LSTM" })
        });
        const data = await res.json();
        if (!data.success) throw new Error("Forecast failed");

        // Update XY Chart
        const actXY = data.actual_trajectory.map(p => ({ x: p.east, y: p.north }));
        const phyXY = data.physics_trajectory.map(p => ({ x: p.east, y: p.north }));
        const aiXY = data.ai_trajectory.map(p => ({ x: p.east, y: p.north }));

        seqChartXYInstance.data.datasets[0].data = actXY;
        seqChartXYInstance.data.datasets[1].data = phyXY;
        seqChartXYInstance.data.datasets[2].data = aiXY;
        seqChartXYInstance.update();

        // Update Depth Chart
        const times = data.timesteps;
        seqChartDepthInstance.data.labels = times;
        seqChartDepthInstance.data.datasets[0].data = data.actual_trajectory.map(p => p.depth);
        seqChartDepthInstance.data.datasets[1].data = data.physics_trajectory.map(p => p.depth);
        seqChartDepthInstance.data.datasets[2].data = data.ai_trajectory.map(p => p.depth);
        seqChartDepthInstance.update();

        // Update metrics bar
        document.getElementById("seq-phy-pos").innerText = `${data.rmse_physics.position_rmse_m} m`;
        document.getElementById("seq-phy-depth").innerText = `${data.rmse_physics.depth_rmse_m} m`;
        document.getElementById("seq-ai-pos").innerText = `${data.rmse_ai.position_rmse_m} m`;
        document.getElementById("seq-ai-depth").innerText = `${data.rmse_ai.depth_rmse_m} m`;

    } catch (err) {
        alert("Forecast error: " + err);
    }
}

// 6. Run Closed-Loop MPC Simulation
async function runMPCSimulation() {
    const trajType = document.getElementById("mpc-traj-select").value;
    const distEnabled = document.getElementById("mpc-dist-check").checked;

    try {
        const res = await fetch("/mpc", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                trajectory_type: trajType,
                simulation_steps: 50,
                enable_disturbance: distEnabled,
                disturbance_current_east: 0.15,
                ai_model_name: "LSTM"
            })
        });
        const data = await res.json();
        if (!data.success) throw new Error("MPC run failed");

        // XY Plot
        mpcChartXYInstance.data.datasets[0].data = data.desired_trajectory.map(p => ({ x: p.east, y: p.north }));
        mpcChartXYInstance.data.datasets[1].data = data.pid_trajectory.map(p => ({ x: p.east, y: p.north }));
        mpcChartXYInstance.data.datasets[2].data = data.physics_mpc_trajectory.map(p => ({ x: p.east, y: p.north }));
        mpcChartXYInstance.data.datasets[3].data = data.ai_mpc_trajectory.map(p => ({ x: p.east, y: p.north }));
        mpcChartXYInstance.update();

        // Depth Plot
        const times = data.desired_trajectory.map(p => p.time_s);
        mpcChartDepthInstance.data.labels = times;
        mpcChartDepthInstance.data.datasets[0].data = data.desired_trajectory.map(p => p.depth);
        mpcChartDepthInstance.data.datasets[1].data = data.pid_trajectory.map(p => p.depth);
        mpcChartDepthInstance.data.datasets[2].data = data.physics_mpc_trajectory.map(p => p.depth);
        mpcChartDepthInstance.data.datasets[3].data = data.ai_mpc_trajectory.map(p => p.depth);
        mpcChartDepthInstance.update();

        // Metrics Table
        document.getElementById("mpc-pid-pos").innerText = data.metrics.PID.position_rmse_m.toFixed(2);
        document.getElementById("mpc-pid-depth").innerText = data.metrics.PID.depth_rmse_m.toFixed(2);
        document.getElementById("mpc-pid-head").innerText = data.metrics.PID.heading_rmse_rad.toFixed(2);
        document.getElementById("mpc-pid-eff").innerText = data.metrics.PID.control_effort.toFixed(1);

        document.getElementById("mpc-phy-pos").innerText = data.metrics.Physics_MPC.position_rmse_m.toFixed(2);
        document.getElementById("mpc-phy-depth").innerText = data.metrics.Physics_MPC.depth_rmse_m.toFixed(2);
        document.getElementById("mpc-phy-head").innerText = data.metrics.Physics_MPC.heading_rmse_rad.toFixed(2);
        document.getElementById("mpc-phy-eff").innerText = data.metrics.Physics_MPC.control_effort.toFixed(1);

        document.getElementById("mpc-ai-pos").innerText = data.metrics.AI_MPC.position_rmse_m.toFixed(2);
        document.getElementById("mpc-ai-depth").innerText = data.metrics.AI_MPC.depth_rmse_m.toFixed(2);
        document.getElementById("mpc-ai-head").innerText = data.metrics.AI_MPC.heading_rmse_rad.toFixed(2);
        document.getElementById("mpc-ai-eff").innerText = data.metrics.AI_MPC.control_effort.toFixed(1);

    } catch (err) {
        alert("MPC Simulation Error: " + err);
    }
}

// Chart Initializations
function initCharts() {
    const commonDarkOptions = {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { grid: { color: "#1e334d" }, ticks: { color: "#8ba2be" } },
            y: { grid: { color: "#1e334d" }, ticks: { color: "#8ba2be" } }
        },
        plugins: {
            legend: { labels: { color: "#e6f1ff", font: { size: 11 } } }
        }
    };

    // 1. Single-Step Position Shift Chart
    const ctxPredict = document.getElementById("predictChart").getContext("2d");
    predictChartInstance = new Chart(ctxPredict, {
        type: "scatter",
        data: {
            datasets: [
                { label: "Current Input (t)", data: [], backgroundColor: "#8ba2be", pointRadius: 8, pointStyle: "rectRot" },
                { label: "Physics Predict (t+1)", data: [], backgroundColor: "#ffc107", pointRadius: 7, pointStyle: "triangle" },
                { label: "Final AI Predict (t+1)", data: [], backgroundColor: "#3d84ff", pointRadius: 8, pointStyle: "circle" },
                { label: "Actual Next (t+1)", data: [], backgroundColor: "#00e676", pointRadius: 9, pointStyle: "crossRot" }
            ]
        },
        options: {
            ...commonDarkOptions,
            scales: {
                x: { ...commonDarkOptions.scales.x, title: { display: true, text: "East Y (m)", color: "#8ba2be" } },
                y: { ...commonDarkOptions.scales.y, title: { display: true, text: "North X (m)", color: "#8ba2be" } }
            }
        }
    });

    // 2. Uploaded CSV Trajectory XY Chart
    const ctxUpXY = document.getElementById("uploadChartXY").getContext("2d");
    uploadChartXYInstance = new Chart(ctxUpXY, {
        type: "scatter",
        data: {
            datasets: [
                { label: "Uploaded Input Path", data: [], showLine: true, borderColor: "#8ba2be", pointRadius: 3 },
                { label: "Model Predicted Future Path (t+1)", data: [], showLine: true, borderColor: "#00d2ff", pointRadius: 3 }
            ]
        },
        options: { ...commonDarkOptions, plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Uploaded vs Predicted Horizontal Path (East vs North)", color: "#e6f1ff" } } }
    });

    // 3. Uploaded CSV Depth Chart
    const ctxUpDepth = document.getElementById("uploadChartDepth").getContext("2d");
    uploadChartDepthInstance = new Chart(ctxUpDepth, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Input Depth", data: [], borderColor: "#8ba2be" },
                { label: "Predicted Future Depth", data: [], borderColor: "#00d2ff" }
            ]
        },
        options: {
            ...commonDarkOptions,
            scales: {
                ...commonDarkOptions.scales,
                y: { ...commonDarkOptions.scales.y, reverse: true, title: { display: true, text: "Depth (m)", color: "#8ba2be" } }
            },
            plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Uploaded vs Predicted Depth Profile", color: "#e6f1ff" } }
        }
    });

    // 4. Sequence XY Chart
    const ctxSeqXY = document.getElementById("sequenceChartXY").getContext("2d");
    seqChartXYInstance = new Chart(ctxSeqXY, {
        type: "scatter",
        data: {
            datasets: [
                { label: "Actual Trajectory", data: [], showLine: true, borderColor: "#e6f1ff", pointRadius: 3 },
                { label: "Physics Rollout", data: [], showLine: true, borderColor: "#ffc107", borderDash: [5, 5], pointRadius: 3 },
                { label: "AI Rollout", data: [], showLine: true, borderColor: "#00d2ff", pointRadius: 3 }
            ]
        },
        options: { ...commonDarkOptions, plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Horizontal (East vs North)", color: "#e6f1ff" } } }
    });

    // 5. Sequence Depth Chart
    const ctxSeqDepth = document.getElementById("sequenceChartDepth").getContext("2d");
    seqChartDepthInstance = new Chart(ctxSeqDepth, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Actual Depth", data: [], borderColor: "#e6f1ff" },
                { label: "Physics Depth", data: [], borderColor: "#ffc107", borderDash: [5, 5] },
                { label: "AI Depth", data: [], borderColor: "#00d2ff" }
            ]
        },
        options: {
            ...commonDarkOptions,
            scales: {
                ...commonDarkOptions.scales,
                y: { ...commonDarkOptions.scales.y, reverse: true, title: { display: true, text: "Depth (m, inverted)", color: "#8ba2be" } }
            },
            plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Depth Profile (Time vs Depth)", color: "#e6f1ff" } }
        }
    });

    // 6. MPC XY Chart
    const ctxMpcXY = document.getElementById("mpcChartXY").getContext("2d");
    mpcChartXYInstance = new Chart(ctxMpcXY, {
        type: "scatter",
        data: {
            datasets: [
                { label: "Desired Reference", data: [], showLine: true, borderColor: "#ffffff", borderDash: [4, 4], pointRadius: 0 },
                { label: "PID Tracking", data: [], showLine: true, borderColor: "#ff5252", pointRadius: 2 },
                { label: "Physics MPC", data: [], showLine: true, borderColor: "#ffc107", pointRadius: 2 },
                { label: "AI-Enhanced MPC (LSTM)", data: [], showLine: true, borderColor: "#00e676", pointRadius: 3, borderWidth: 2 }
            ]
        },
        options: { ...commonDarkOptions, plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Closed-Loop Horizontal Tracking (East vs North)", color: "#e6f1ff" } } }
    });

    // 7. MPC Depth Chart
    const ctxMpcDepth = document.getElementById("mpcChartDepth").getContext("2d");
    mpcChartDepthInstance = new Chart(ctxMpcDepth, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Desired Depth", data: [], borderColor: "#ffffff", borderDash: [4, 4], pointRadius: 0 },
                { label: "PID Depth", data: [], borderColor: "#ff5252", pointRadius: 2 },
                { label: "Physics MPC Depth", data: [], borderColor: "#ffc107", pointRadius: 2 },
                { label: "AI-Enhanced MPC Depth", data: [], borderColor: "#00e676", pointRadius: 2, borderWidth: 2 }
            ]
        },
        options: {
            ...commonDarkOptions,
            scales: {
                ...commonDarkOptions.scales,
                y: { ...commonDarkOptions.scales.y, reverse: true, title: { display: true, text: "Depth (m)", color: "#8ba2be" } }
            },
            plugins: { ...commonDarkOptions.plugins, title: { display: true, text: "Closed-Loop Depth Tracking", color: "#e6f1ff" } }
        }
    });
}

function updatePredictChart(current, physics, ai, actual) {
    predictChartInstance.data.datasets[0].data = [current];
    predictChartInstance.data.datasets[1].data = [physics];
    predictChartInstance.data.datasets[2].data = [ai];
    predictChartInstance.data.datasets[3].data = actual ? [actual] : [];
    predictChartInstance.update();
}
