// Autosub6000 Neural MPC Mission Terminal - Client Logic
let simData = null;
let currentStep = 0;
let isPlaying = false;
let playbackSpeed = 1;
let animTimer = null;
let mapPickerMode = null; // 'start', 'goal', 'obstacle', or null
let mouseWorldPos = null;
let obstacles = [];

// Canvas elements
const radarCanvas = document.getElementById('radarCanvas');
const radarCtx = radarCanvas.getContext('2d');
const depthCanvas = document.getElementById('depthCanvas');
const depthCtx = depthCanvas.getContext('2d');

// Canonical Scenario Presets
let scenarioCache = {};

window.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  resizeCanvases();
  window.addEventListener('resize', resizeCanvases);
  await fetchScenarios();
  loadInitialDefault();
  setupRadarClickHandler();
  initAutoDownloadPref();
  setupInputListeners();
});

function resizeCanvases() {
  const rw = document.getElementById('radarWrapper');
  radarCanvas.width = rw.clientWidth;
  radarCanvas.height = rw.clientHeight;

  const dw = depthCanvas.parentElement;
  depthCanvas.width = dw.clientWidth;
  depthCanvas.height = dw.clientHeight;

  renderAll();
}

async function fetchScenarios() {
  try {
    const res = await fetch('/api/scenarios');
    scenarioCache = await res.json();
  } catch (err) {
    console.error("Error fetching scenarios:", err);
  }
}

function loadInitialDefault() {
  if (scenarioCache['basic_A_to_B']) {
    applyScenario(scenarioCache['basic_A_to_B']);
  } else {
    obstacles = [];
    renderObstacleList();
    renderAll();
  }
}

function loadSelectedScenario() {
  const sel = document.getElementById('scenarioSelect').value;
  if (sel !== 'custom' && scenarioCache[sel]) {
    applyScenario(scenarioCache[sel]);
  }
}

function applyScenario(scen) {
  document.getElementById('start_n').value = scen.start[0];
  document.getElementById('start_e').value = scen.start[1];
  document.getElementById('start_d').value = scen.start[2];

  document.getElementById('goal_n').value = scen.goal[0];
  document.getElementById('goal_e').value = scen.goal[1];
  document.getElementById('goal_d').value = scen.goal[2];

  const b = scen.boundaries || {};
  document.getElementById('b_n_min').value = b.north_min ?? 0.0;
  document.getElementById('b_n_max').value = b.north_max ?? 120.0;
  document.getElementById('b_e_min').value = b.east_min ?? 0.0;
  document.getElementById('b_e_max').value = b.east_max ?? 100.0;
  document.getElementById('b_d_min').value = b.depth_min ?? 10.0;
  document.getElementById('b_d_max').value = b.depth_max ?? 60.0;

  const c = scen.current || [0, 0, 0];
  document.getElementById('curr_vx').value = c[0] || 0.0;
  document.getElementById('curr_vy').value = c[1] || 0.0;
  document.getElementById('curr_vz').value = c[2] || 0.0;

  obstacles = (scen.obstacles || []).map(o => ({
    north: o.north,
    east: o.east,
    radius: o.radius || 8.0,
    margin: o.margin || 4.0,
    depth_min: o.depth_min ?? 0.0,
    depth_max: o.depth_max ?? 80.0
  }));

  const retChk = document.getElementById('returnToStartChk');
  if (retChk) {
    if (scen.return_to_start !== undefined) {
      retChk.checked = Boolean(scen.return_to_start);
    }
    const runBtn = document.getElementById('runBtn');
    if (runBtn) {
      runBtn.innerHTML = retChk.checked ? `<span>🚀 RUN ROUND-TRIP MPC (A → B → A)</span>` : `<span>🚀 RUN OUTBOUND MPC (A → B ONLY)</span>`;
    }
  }
  const sSlider = document.getElementById('stepsSlider');
  const sVal = document.getElementById('stepsVal');
  if (retChk && retChk.checked) {
    if (sSlider) {
      sSlider.max = 250;
      sSlider.value = 180;
    }
    if (sVal) sVal.innerText = 180;
  }

  renderObstacleList();
  invalidateOldSimulation();
  renderAll();
}

function invalidateOldSimulation() {
  pause();
  simData = null;
  currentStep = 0;
  const slider = document.getElementById('timelineSlider');
  if (slider) {
    slider.max = 0;
    slider.value = 0;
  }
  const jBtn = document.getElementById('jumpReturnBtn');
  if (jBtn) jBtn.style.display = 'none';
  const info = document.getElementById('timelineInfo') || document.getElementById('timelineReadout');
  if (info) info.innerText = 'Time: 0.0s | Ready';
}

function renderObstacleList() {
  const list = document.getElementById('obstacleList');
  if (!list) return;
  list.innerHTML = '';
  if (obstacles.length === 0) {
    list.innerHTML = '<div style="font-size: 11px; color: var(--text-dim); padding: 4px;">No obstacles in area. Click map or + Midpoint to add.</div>';
    return;
  }
  obstacles.forEach((obs, idx) => {
    const item = document.createElement('div');
    item.style.cssText = "display: flex; align-items: center; justify-content: space-between; background: var(--bg-input); padding: 5px 8px; border-radius: 4px; font-size: 11px; border: 1px solid rgba(245, 158, 11, 0.2);";
    item.innerHTML = `
      <span><strong style="color: #fbbf24;">#${idx+1}</strong> (${obs.north.toFixed(1)}N, ${obs.east.toFixed(1)}E) | r=${obs.radius}m</span>
      <button onclick="removeObstacle(${idx})" title="Delete Hazard" style="background: none; border: none; color: #ef4444; cursor: pointer; font-size: 13px; font-weight: bold; padding: 0 4px;">✕</button>
    `;
    list.appendChild(item);
  });
}

function addObstacle() {
  const n = (parseFloat(document.getElementById('start_n').value) + parseFloat(document.getElementById('goal_n').value)) / 2;
  const e = (parseFloat(document.getElementById('start_e').value) + parseFloat(document.getElementById('goal_e').value)) / 2;
  const r = parseFloat(document.getElementById('obsRadiusSelect')?.value || 8.0);
  obstacles.push({ north: parseFloat(n.toFixed(1)), east: parseFloat(e.toFixed(1)), radius: r, margin: 4.0, depth_min: 0.0, depth_max: 80.0 });
  renderObstacleList();
  invalidateOldSimulation();
  renderAll();
}

function removeObstacle(idx) {
  obstacles.splice(idx, 1);
  renderObstacleList();
  invalidateOldSimulation();
  renderAll();
}

function clearAllObstacles() {
  obstacles = [];
  renderObstacleList();
  invalidateOldSimulation();
  renderAll();
}

// Map Click Picker for Start / Goal / Obstacles
function toggleMapPicker(mode) {
  const startBtn = document.getElementById('pickStartBtn');
  const goalBtn = document.getElementById('pickGoalBtn');
  const obsBtn = document.getElementById('pickObsBtn');
  const obsHelp = document.getElementById('obsPickerHelp');

  if (mapPickerMode === mode) {
    mapPickerMode = null;
    if (startBtn) startBtn.classList.remove('active');
    if (goalBtn) goalBtn.classList.remove('active');
    if (obsBtn) {
      obsBtn.classList.remove('active');
      obsBtn.innerHTML = "📍 Add Obstacles on Radar Map (Click to Place)";
    }
    if (obsHelp) obsHelp.style.display = 'none';
    radarCanvas.classList.remove('canvas-crosshair');
    mouseWorldPos = null;
  } else {
    mapPickerMode = mode;
    if (startBtn) startBtn.classList.toggle('active', mode === 'start');
    if (goalBtn) goalBtn.classList.toggle('active', mode === 'goal');
    if (obsBtn) {
      obsBtn.classList.toggle('active', mode === 'obstacle');
      obsBtn.innerHTML = (mode === 'obstacle') ? "🛑 Finish Adding Obstacles" : "📍 Add Obstacles on Radar Map (Click to Place)";
    }
    if (obsHelp) obsHelp.style.display = (mode === 'obstacle') ? 'block' : 'none';
    radarCanvas.classList.add('canvas-crosshair');
  }
  renderAll();
}

function setupRadarClickHandler() {
  radarCanvas.addEventListener('click', (e) => {
    if (!mapPickerMode) return;
    const rect = radarCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const coords = screenToWorld(x, y);
    if (mapPickerMode === 'start') {
      document.getElementById('start_n').value = coords.north.toFixed(1);
      document.getElementById('start_e').value = coords.east.toFixed(1);
      toggleMapPicker('start'); // exit mode
    } else if (mapPickerMode === 'goal') {
      document.getElementById('goal_n').value = coords.north.toFixed(1);
      document.getElementById('goal_e').value = coords.east.toFixed(1);
      toggleMapPicker('goal'); // exit mode
    } else if (mapPickerMode === 'obstacle') {
      // Add obstacle at clicked location!
      // Leave mode active so user can add obstacles one after the other!
      const rad = parseFloat(document.getElementById('obsRadiusSelect')?.value || 8.0);
      obstacles.push({
        north: parseFloat(coords.north.toFixed(1)),
        east: parseFloat(coords.east.toFixed(1)),
        radius: rad,
        margin: 4.0,
        depth_min: 0.0,
        depth_max: 80.0
      });
      renderObstacleList();
    }

    invalidateOldSimulation();
    renderAll();
  });

  radarCanvas.addEventListener('mousemove', (e) => {
    const rect = radarCanvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const w = screenToWorld(x, y);
    const hud = document.getElementById('radarHud');

    if (hud) {
      if (mapPickerMode === 'obstacle') {
        const rad = parseFloat(document.getElementById('obsRadiusSelect')?.value || 8.0);
        hud.innerHTML = `RADAR: 2D BATHYMETRIC PLANE<br><span style="color: #fbbf24; font-weight: bold;">[ADD OBSTACLE MODE] Click to place r=${rad}m obstacle at (${w.north.toFixed(1)}m N, ${w.east.toFixed(1)}m E)</span>`;
      } else if (mapPickerMode === 'start') {
        hud.innerHTML = `RADAR: 2D BATHYMETRIC PLANE<br><span style="color: #10b981; font-weight: bold;">[SET START] Click to set Point A at (${w.north.toFixed(1)}m N, ${w.east.toFixed(1)}m E)</span>`;
      } else if (mapPickerMode === 'goal') {
        hud.innerHTML = `RADAR: 2D BATHYMETRIC PLANE<br><span style="color: #ef4444; font-weight: bold;">[SET GOAL] Click to set Point B at (${w.north.toFixed(1)}m N, ${w.east.toFixed(1)}m E)</span>`;
      } else {
        hud.innerHTML = `RADAR: 2D BATHYMETRIC PLANE (NORTH vs EAST)<br><span style="color: #38bdf8;">Cursor: ${w.north.toFixed(1)}m N, ${w.east.toFixed(1)}m E</span>`;
      }
    }

    if (mapPickerMode === 'obstacle') {
      mouseWorldPos = w;
      renderAll();
    }
  });

  radarCanvas.addEventListener('mouseleave', () => {
    if (mapPickerMode === 'obstacle') {
      mouseWorldPos = null;
      renderAll();
    }
  });
}

function setupInputListeners() {
  const ids = ['start_n', 'start_e', 'start_d', 'goal_n', 'goal_e', 'goal_d', 'b_n_min', 'b_n_max', 'b_e_min', 'b_e_max', 'b_d_min', 'b_d_max', 'curr_vx', 'curr_vy', 'curr_vz'];
  ids.forEach(id => {
    const el = document.getElementById(id);
    if (el) {
      el.addEventListener('input', () => {
        invalidateOldSimulation();
        renderAll();
      });
    }
  });

  const retChk = document.getElementById('returnToStartChk');
  const runBtn = document.getElementById('runBtn');
  const updateRunBtnText = () => {
    if (!runBtn) return;
    const isRet = retChk ? retChk.checked : true;
    runBtn.innerHTML = isRet ? `<span>🚀 RUN ROUND-TRIP MPC (A → B → A)</span>` : `<span>🚀 RUN OUTBOUND MPC (A → B ONLY)</span>`;
  };

  if (retChk) {
    retChk.addEventListener('change', (e) => {
      const isChecked = e.target.checked;
      const sSlider = document.getElementById('stepsSlider');
      const sVal = document.getElementById('stepsVal');
      if (isChecked) {
        if (sSlider) {
          sSlider.max = 250;
          if (parseInt(sSlider.value) < 180) {
            sSlider.value = 180;
            if (sVal) sVal.innerText = 180;
          }
        }
      }
      updateRunBtnText();
      invalidateOldSimulation();
      renderAll();
    });
    updateRunBtnText();
  }
}

// World <-> Screen Coordinate Mapping
function getBounds() {
  return {
    north_min: parseFloat(document.getElementById('b_n_min').value) || 0.0,
    north_max: parseFloat(document.getElementById('b_n_max').value) || 120.0,
    east_min: parseFloat(document.getElementById('b_e_min').value) || 0.0,
    east_max: parseFloat(document.getElementById('b_e_max').value) || 100.0,
    depth_min: parseFloat(document.getElementById('b_d_min').value) || 10.0,
    depth_max: parseFloat(document.getElementById('b_d_max').value) || 60.0,
  };
}

function worldToScreen(north, east) {
  const margin = 45;
  const b = getBounds();
  const w = radarCanvas.width;
  const h = radarCanvas.height;

  const spanE = Math.max(1, b.east_max - b.east_min);
  const spanN = Math.max(1, b.north_max - b.north_min);

  // Strict 1:1 aspect ratio - prevents any horizontal or vertical coordinate stretching
  const scale = Math.min((w - 2 * margin) / spanE, (h - 2 * margin) / spanN);
  const offsetX = margin + ((w - 2 * margin) - spanE * scale) / 2;
  const offsetY = margin + ((h - 2 * margin) - spanN * scale) / 2;

  return {
    x: offsetX + (east - b.east_min) * scale,
    y: h - offsetY - (north - b.north_min) * scale,
    scaleX: scale,
    scaleY: scale,
    scale: scale
  };
}

function screenToWorld(x, y) {
  const margin = 45;
  const b = getBounds();
  const w = radarCanvas.width;
  const h = radarCanvas.height;

  const spanE = Math.max(1, b.east_max - b.east_min);
  const spanN = Math.max(1, b.north_max - b.north_min);
  const scale = Math.min((w - 2 * margin) / spanE, (h - 2 * margin) / spanN);
  const offsetX = margin + ((w - 2 * margin) - spanE * scale) / 2;
  const offsetY = margin + ((h - 2 * margin) - spanN * scale) / 2;

  const east = (x - offsetX) / scale + b.east_min;
  const north = (h - offsetY - y) / scale + b.north_min;

  return { north, east };
}

// Run Simulation Action
async function runSimulation() {
  const btn = document.getElementById('runBtn');
  const alertBox = document.getElementById('simStatusAlert');

  btn.disabled = true;
  btn.innerHTML = `<span>⏳ COMPUTING CLOSED-LOOP MPC...</span>`;
  alertBox.style.display = 'block';
  alertBox.style.background = 'rgba(56, 189, 248, 0.15)';
  alertBox.style.color = '#38bdf8';
  alertBox.style.border = '1px solid rgba(56, 189, 248, 0.3)';
  alertBox.innerText = "Running decoupled RK4 plant with Phase 1 GRU residual model...";

  const payload = {
    start: {
      north: parseFloat(document.getElementById('start_n').value),
      east: parseFloat(document.getElementById('start_e').value),
      depth: parseFloat(document.getElementById('start_d').value),
    },
    goal: {
      north: parseFloat(document.getElementById('goal_n').value),
      east: parseFloat(document.getElementById('goal_e').value),
      depth: parseFloat(document.getElementById('goal_d').value),
      tolerance: 5.0
    },
    boundaries: getBounds(),
    obstacles: obstacles,
    method: document.getElementById('methodSelect').value,
    horizon: parseInt(document.getElementById('horizonSlider').value),
    num_candidates: parseInt(document.getElementById('candSelect').value) || 20,
    iterations: (parseInt(document.getElementById('candSelect').value) || 20) <= 25 ? 2 : 3,
    max_steps: parseInt(document.getElementById('stepsSlider').value),
    current_velocity: [
      parseFloat(document.getElementById('curr_vx').value) || 0.0,
      parseFloat(document.getElementById('curr_vy').value) || 0.0,
      parseFloat(document.getElementById('curr_vz').value) || 0.0
    ],
    use_kalman_filter: true,
    randomize_plant_mismatch: false,
    usbl_sigma: 0.50,
    seed: 1000,
    return_to_start: document.getElementById('returnToStartChk')?.checked || false
  };

  try {
    const res = await fetch('/api/simulate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Simulation execution failed");
    }

    simData = await res.json();
    currentStep = 0;
    pause();

    // Setup timeline scrubber
    const tSlider = document.getElementById('timelineSlider');
    tSlider.max = Math.max(0, simData.trajectory.length - 1);
    tSlider.value = 0;

    // Update KPIs
    updateMissionKPIs(simData.metrics);

    // Setup jump button for return leg
    const jBtn = document.getElementById('jumpReturnBtn');
    if (jBtn) {
      if (simData.metrics && simData.metrics.return_to_start && simData.metrics.step_reached_b) {
        jBtn.style.display = 'inline-flex';
        jBtn.innerText = `🔙 Jump to Return (Step ${simData.metrics.step_reached_b})`;
      } else {
        jBtn.style.display = 'none';
      }
    }

    renderAll();
    syncStepUI();
    play(); // automatically start playback

    // Render the new mathematical proof section with calculations and clearance table
    renderMathProofSection(simData);

    // Trigger download of calculation report file only if auto-download preference is enabled
    if (getAutoDownloadPref()) {
      downloadCalculationDoc();
    }

    // Post-simulation modal
    showSimCompleteModal();
  } catch (err) {
    alertBox.style.background = 'rgba(239, 68, 68, 0.15)';
    alertBox.style.color = '#f87171';
    alertBox.style.border = '1px solid rgba(239, 68, 68, 0.3)';
    alertBox.innerText = "Error: " + err.message;
  } finally {
    btn.disabled = false;
    const isRet = document.getElementById('returnToStartChk')?.checked;
    btn.innerHTML = isRet ? `<span>🚀 RUN ROUND-TRIP MPC (A → B → A)</span>` : `<span>🚀 RUN OUTBOUND MPC (A → B ONLY)</span>`;
  }
}

function updateMissionKPIs(m) {
  const badge = document.getElementById('verdictBadge');
  if (m.boundary_violations > 0) {
    badge.className = "status-badge status-failed";
    badge.innerText = `GEOFENCE BREACH (${(m.max_boundary_violation_m || 0).toFixed(1)}m)`;
  } else if (m.physical_collisions > 0) {
    badge.className = "status-badge status-failed";
    badge.innerText = `COLLISION DETECTED (${m.physical_collisions})`;
  } else if (m.return_to_start) {
    if (m.success_5m) {
      badge.className = "status-badge status-reached";
      badge.innerText = "A → B → A MISSION COMPLETED";
    } else if (m.reached_b) {
      badge.className = "status-badge status-failed";
      badge.innerText = "B REACHED (RETURN INCOMPLETE)";
    } else {
      badge.className = "status-badge status-failed";
      badge.innerText = "MISSION TIMEOUT";
    }
  } else if (m.success_5m) {
    badge.className = "status-badge status-reached";
    badge.innerText = "GOAL REACHED (≤5m)";
  } else {
    badge.className = "status-badge status-failed";
    badge.innerText = "CLOSEST APPROACH";
  }

  document.getElementById('kpiDist').innerText = `${m.final_distance_m.toFixed(2)} m`;
  document.getElementById('kpiTime').innerText = `${m.time_to_goal_sec.toFixed(1)} s`;
  document.getElementById('kpiEff').innerText = `${m.path_efficiency_pct.toFixed(1)} %`;

  const colEl = document.getElementById('kpiCols');
  colEl.innerText = m.physical_collisions;
  colEl.style.color = m.physical_collisions === 0 ? "#34d399" : "#ef4444";

  const bufEl = document.getElementById('kpiBuffer');
  bufEl.innerText = m.buffer_violations;
  bufEl.style.color = m.buffer_violations === 0 ? "#34d399" : "#f59e0b";

  document.getElementById('kpiSolve').innerText = `${(m.solve_time_mean_sec * 1000.0).toFixed(1)} ms`;
}

// High-Performance 60 FPS Continuous Playback Engine
let animFrameId = null;
let currentFractionalStep = 0.0;
let lastAnimTimestamp = null;

function togglePlay() {
  if (isPlaying) {
    pause();
  } else {
    play();
  }
}

function play() {
  if (!simData || !simData.trajectory || !simData.trajectory.length) return;
  isPlaying = true;
  const playBtn = document.getElementById('playBtn');
  if (playBtn) playBtn.innerText = '⏸';

  if (currentFractionalStep >= simData.trajectory.length - 1) {
    currentFractionalStep = 0.0;
    currentStep = 0;
  }
  cancelAnimationFrame(animFrameId);
  lastAnimTimestamp = null;
  animFrameId = requestAnimationFrame(animStep);
}

function animStep(timestamp) {
  if (!isPlaying || !simData || !simData.trajectory || !simData.trajectory.length) return;
  if (!lastAnimTimestamp) lastAnimTimestamp = timestamp;
  const deltaSec = Math.min(0.08, (timestamp - lastAnimTimestamp) / 1000.0);
  lastAnimTimestamp = timestamp;

  const speedEl = document.getElementById('speedSelect') || document.getElementById('playbackSpeed');
  if (speedEl) playbackSpeed = parseFloat(speedEl.value) || 1.0;

  // Real step is 2.0s of physical simulation. Paced for ultra-smooth fluid visual glide
  const stepsPerSec = 2.8 * playbackSpeed;
  currentFractionalStep += deltaSec * stepsPerSec;
  const maxStep = simData.trajectory.length - 1;

  if (currentFractionalStep >= maxStep) {
    currentFractionalStep = maxStep;
    currentStep = maxStep;
    syncStepUI();
    renderAll(currentFractionalStep);
    pause();
    return;
  }

  currentStep = Math.floor(currentFractionalStep);
  syncStepUI();
  renderAll(currentFractionalStep);
  animFrameId = requestAnimationFrame(animStep);
}

function pause() {
  isPlaying = false;
  const playBtn = document.getElementById('playBtn');
  if (playBtn) playBtn.innerText = '▶';
  cancelAnimationFrame(animFrameId);
  lastAnimTimestamp = null;
}

function stepForward() {
  pause();
  if (simData && simData.trajectory && currentFractionalStep < simData.trajectory.length - 1) {
    currentFractionalStep = Math.min(simData.trajectory.length - 1, Math.floor(currentFractionalStep) + 1);
    currentStep = Math.floor(currentFractionalStep);
    syncStepUI();
    renderAll(currentFractionalStep);
  }
}

function stepBackward() {
  pause();
  if (simData && simData.trajectory && currentFractionalStep > 0) {
    currentFractionalStep = Math.max(0, Math.ceil(currentFractionalStep) - 1);
    currentStep = Math.floor(currentFractionalStep);
    syncStepUI();
    renderAll(currentFractionalStep);
  }
}

function resetPlayback() {
  pause();
  currentFractionalStep = 0.0;
  currentStep = 0;
  syncStepUI();
  renderAll(currentFractionalStep);
}

function jumpToReturnLeg() {
  if (!simData || !simData.trajectory || !simData.metrics) return;
  const stepB = simData.metrics.step_reached_b;
  if (stepB === undefined || stepB === null) return;
  pause();
  const targetStep = Math.min(simData.trajectory.length - 1, Math.max(0, stepB));
  currentFractionalStep = targetStep;
  currentStep = targetStep;
  syncStepUI();
  renderAll(currentFractionalStep);
  play();
}

function onScrubTimeline(val) {
  pause();
  currentFractionalStep = parseFloat(val) || 0.0;
  currentStep = Math.floor(currentFractionalStep);
  syncStepUI();
  renderAll(currentFractionalStep);
}

// Alias for index.html scrubber event
function onScrubberInput(val) {
  onScrubTimeline(val);
}

function updatePlaybackSpeed() {
  const speedEl = document.getElementById('speedSelect') || document.getElementById('playbackSpeed');
  if (speedEl) {
    playbackSpeed = parseFloat(speedEl.value) || 1.0;
  }
}

function syncStepUI() {
  const slider = document.getElementById('timelineSlider');
  if (slider) slider.value = currentStep;
  if (!simData || !simData.trajectory || !simData.trajectory.length) return;

  const pt = simData.trajectory[currentStep];
  if (!pt) return;
  const maxStep = simData.trajectory.length - 1;

  const readout = document.getElementById('timelineInfo') || document.getElementById('timelineReadout');
  if (readout) {
    readout.innerText = `Time: ${(pt.time_sec || 0).toFixed(1)}s | Step: ${currentStep} / ${maxStep}`;
  }

  // Update Telemetry HUD safely
  const setTxt = (id, txt) => {
    const el = document.getElementById(id);
    if (el) el.innerText = txt;
  };

  const isLeg2 = (pt.leg === 2 || pt.phase === 'RETURNING' || pt.phase === 'ROTATING_BACK' || pt.phase === 'ROTATING_AT_A');
  let phaseTxt = '➡️ Leg 1: Outbound Forward (A → B)';
  let phaseColor = '#38bdf8';
  if (pt.phase === 'COMPLETED') {
    phaseTxt = '✅ Mission Completed at Origin';
    phaseColor = '#34d399';
  } else if (pt.phase === 'ROTATING_BACK') {
    phaseTxt = '🔄 Rotating Back at B (Turning 180° to Point A)';
    phaseColor = '#fbbf24';
  } else if (pt.phase === 'ROTATING_AT_A') {
    phaseTxt = '🔄 Arrived at A: Rotating to Initial Pose';
    phaseColor = '#10b981';
  } else if (isLeg2) {
    phaseTxt = '🔙 Leg 2: Return Journey (B → A)';
    phaseColor = '#f59e0b';
  }
  
  const phaseEl = document.getElementById('hudPhase');
  if (phaseEl) {
    phaseEl.innerText = phaseTxt;
    phaseEl.style.color = phaseColor;
  }

  setTxt('hudCoords', `${pt.true_north.toFixed(2)}, ${pt.true_east.toFixed(2)}, ${pt.true_depth.toFixed(2)} m`);
  setTxt('hudSpeed', `${pt.true_speed.toFixed(2)} m/s (Target: ${(pt.cmd_speed !== null && pt.cmd_speed !== undefined ? pt.cmd_speed.toFixed(2) : 1.20)} m/s)`);

  const rollDeg = (pt.true_roll * 180 / Math.PI).toFixed(1);
  const pitchDeg = (pt.true_pitch * 180 / Math.PI).toFixed(1);
  const yawDeg = (((pt.true_heading * 180 / Math.PI) % 360) + 360) % 360;
  setTxt('hudAttitude', `Roll: ${rollDeg}°, Pitch: ${pitchDeg}°, Yaw: ${yawDeg.toFixed(1)}°`);

  const cmdU = (pt.cmd_speed !== null && pt.cmd_speed !== undefined) ? `${pt.cmd_speed.toFixed(2)} m/s` : 'None';
  const cmdP = (pt.cmd_pitch !== null && pt.cmd_pitch !== undefined) ? `${(pt.cmd_pitch * 180 / Math.PI).toFixed(1)}°` : 'None';
  const cmdH = (pt.cmd_heading !== null && pt.cmd_heading !== undefined) ? `${((((pt.cmd_heading * 180 / Math.PI) % 360) + 360) % 360).toFixed(1)}°` : 'None';
  setTxt('hudAction', `${cmdU}, ${cmdP}, ${cmdH}`);

  const activeDist = (pt.dist_to_target !== undefined ? pt.dist_to_target : pt.dist_to_goal);
  setTxt('hudDistGoal', `${activeDist.toFixed(2)} m (${isLeg2 ? 'to Point A' : 'to Point B'})`);

  const hazDist = pt.min_obstacle_dist;
  const hazEl = document.getElementById('hudHazardDist');
  if (hazEl) {
    if (hazDist >= 50) {
      hazEl.innerText = `Clear (${hazDist.toFixed(1)}m)`;
      hazEl.style.color = "#34d399";
    } else if (hazDist > 12) {
      hazEl.innerText = `Approaching (${hazDist.toFixed(1)}m)`;
      hazEl.style.color = "#38bdf8";
    } else if (hazDist > 8) {
      hazEl.innerText = `Safety Buffer Breach (${hazDist.toFixed(1)}m)`;
      hazEl.style.color = "#f59e0b";
    } else {
      hazEl.innerText = `CRITICAL COLLISION (${hazDist.toFixed(1)}m)`;
      hazEl.style.color = "#ef4444";
    }
  }

  const oodBadge = document.getElementById('oodBadge');
  if (oodBadge) {
    if (pt.ood_flag) {
      oodBadge.innerText = "OOD DETECTED";
      oodBadge.style.background = "rgba(239, 68, 68, 0.2)";
      oodBadge.style.color = "#f87171";
    } else {
      oodBadge.innerText = "IN-DISTRIBUTION";
      oodBadge.style.background = "rgba(16, 185, 129, 0.15)";
      oodBadge.style.color = "#34d399";
    }
  }

  // Update Cost Function Bars safely
  const jGoal = pt.cost_j_goal || 0.0;
  const jTrack = pt.cost_j_tracking || 0.0;
  const jCtrl = pt.cost_j_control || 0.0;
  const jSmooth = pt.cost_j_smooth || 0.0;
  const jConstraint = pt.cost_j_constraint || 0.0;
  const jTotal = pt.cost_total_cost || 0.0;

  setTxt('costTotalVal', jTotal > 1e4 ? jTotal.toExponential(2) : jTotal.toFixed(1));
  setTxt('costGoalVal', jGoal.toFixed(1));
  setTxt('costTrackVal', jTrack.toFixed(2));
  setTxt('costCtrlVal', jCtrl.toFixed(2));
  setTxt('costSmoothVal', jSmooth.toFixed(2));
  setTxt('costConstraintVal', jConstraint.toFixed(1));

  const maxCost = Math.max(10.0, jGoal + jTrack + jCtrl + jSmooth + jConstraint);
  const setBar = (id, val) => {
    const el = document.getElementById(id);
    if (el) el.style.width = `${Math.min(100, (val / maxCost) * 100)}%`;
  };
  setBar('costGoalBar', jGoal);
  setBar('costTrackBar', jTrack);
  setBar('costCtrlBar', jCtrl);
  setBar('costSmoothBar', jSmooth);
  setBar('costConstraintBar', jConstraint);
}

// Helper for shortest angular distance in radians
function wrapAngleRad(a) {
  return Math.atan2(Math.sin(a), Math.cos(a));
}

// Rendering Engine (Canvas)
function renderAll(fractionalStep) {
  const fStep = (fractionalStep !== undefined) ? fractionalStep : currentFractionalStep;
  renderRadar(fStep);
  renderDepth(fStep);
}

function renderRadar(fracStep) {
  const ctx = radarCtx;
  const w = radarCanvas.width;
  const h = radarCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const b = getBounds();
  const { scale } = worldToScreen(0, 0);

  // 1. Geofence Boundary Box & Soft Oceanic Bathymetric Backdrop
  const p0 = worldToScreen(b.north_min, b.east_min);
  const p1 = worldToScreen(b.north_max, b.east_max);
  const boxW = p1.x - p0.x;
  const boxH = p0.y - p1.y;

  ctx.fillStyle = "rgba(3, 22, 54, 0.45)";
  ctx.fillRect(p0.x, p1.y, boxW, boxH);

  ctx.strokeStyle = "rgba(56, 189, 248, 0.55)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.strokeRect(p0.x, p1.y, boxW, boxH);
  ctx.setLineDash([]);

  // 2. Draw Bathymetric Grid & Distance Ticks (Exact 1:1 Squares)
  ctx.strokeStyle = "rgba(100, 116, 139, 0.28)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(148, 163, 184, 0.85)";
  ctx.font = "10px ui-monospace, Consolas, monospace";

  for (let n = Math.ceil(b.north_min / 20) * 20; n <= b.north_max; n += 20) {
    const s0 = worldToScreen(n, b.east_min);
    const s1 = worldToScreen(n, b.east_max);
    ctx.beginPath();
    ctx.moveTo(s0.x, s0.y);
    ctx.lineTo(s1.x, s1.y);
    ctx.stroke();
    ctx.fillText(`${n}m N`, Math.max(4, s0.x - 38), s0.y + 3);
  }

  for (let e = Math.ceil(b.east_min / 20) * 20; e <= b.east_max; e += 20) {
    const s0 = worldToScreen(b.north_min, e);
    const s1 = worldToScreen(b.north_max, e);
    ctx.beginPath();
    ctx.moveTo(s0.x, s0.y);
    ctx.lineTo(s1.x, s1.y);
    ctx.stroke();
    ctx.fillText(`${e}m E`, s0.x - 12, Math.min(h - 4, s0.y + 16));
  }

  // Live preview for cursor when in obstacle adding mode
  if (mapPickerMode === 'obstacle' && mouseWorldPos) {
    const sCur = worldToScreen(mouseWorldPos.north, mouseWorldPos.east);
    const curRad = parseFloat(document.getElementById('obsRadiusSelect')?.value || 8.0);
    const rSafeCur = (curRad + 4.0) * scale;
    const rPhysicalCur = curRad * scale;

    ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
    ctx.strokeStyle = "rgba(245, 158, 11, 0.9)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(sCur.x, sCur.y, rSafeCur, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "rgba(239, 68, 68, 0.4)";
    ctx.strokeStyle = "#ef4444";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(sCur.x, sCur.y, rPhysicalCur, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#fbbf24";
    ctx.font = "bold 10px monospace";
    ctx.fillText(`+NEW [r=${curRad}m]`, sCur.x + 8, sCur.y - 8);
  }

  // 3. Obstacle Cylinders (Rendered with 1:1 True Circular Scale)
  obstacles.forEach((obs) => {
    const s = worldToScreen(obs.north, obs.east);
    const rSafe = (obs.radius + obs.margin) * scale;
    const rPhysical = obs.radius * scale;

    // Safety Standoff Buffer (dashed amber)
    ctx.fillStyle = "rgba(245, 158, 11, 0.08)";
    ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(s.x, s.y, rSafe, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);

    // Physical Hull Zone (solid crimson danger zone)
    ctx.fillStyle = "rgba(244, 63, 94, 0.28)";
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(s.x, s.y, rPhysical, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Center Crosshair
    ctx.strokeStyle = "#f43f5e";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(s.x - 4, s.y); ctx.lineTo(s.x + 4, s.y);
    ctx.moveTo(s.x, s.y - 4); ctx.lineTo(s.x, s.y + 4);
    ctx.stroke();
  });

  // 4. Start Point A & Goal Point B
  const sStart = worldToScreen(parseFloat(document.getElementById('start_n').value) || 0, parseFloat(document.getElementById('start_e').value) || 0);
  const sGoal = worldToScreen(parseFloat(document.getElementById('goal_n').value) || 100, parseFloat(document.getElementById('goal_e').value) || 80);

  // Direct Spatial Line-Of-Sight Vector (dashed gray)
  ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([3, 4]);
  ctx.beginPath();
  ctx.moveTo(sStart.x, sStart.y);
  ctx.lineTo(sGoal.x, sGoal.y);
  ctx.stroke();
  ctx.setLineDash([]);

  // Goal Acceptance Zone Ring (5m tolerance circle in true scale)
  const tolRadius = 5.0 * scale;
  ctx.fillStyle = "rgba(239, 68, 68, 0.12)";
  ctx.strokeStyle = "#ef4444";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(sGoal.x, sGoal.y, tolRadius, 0, 2 * Math.PI);
  ctx.fill();
  ctx.stroke();

  // If return_to_start is enabled, draw return acceptance ring around Point A
  const isReturnMode = document.getElementById('returnToStartChk')?.checked || (simData && simData.metrics && simData.metrics.return_to_start);
  if (isReturnMode) {
    ctx.fillStyle = "rgba(16, 185, 129, 0.12)";
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.arc(sStart.x, sStart.y, tolRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Goal Target Center (crimson pulsating dot)
  ctx.fillStyle = "#ef4444";
  ctx.beginPath();
  ctx.arc(sGoal.x, sGoal.y, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.font = "bold 11px system-ui, -apple-system, sans-serif";
  ctx.fillText("B (Goal)", sGoal.x + 10, sGoal.y - 4);

  // Start Pin (emerald)
  ctx.fillStyle = "#10b981";
  ctx.beginPath();
  ctx.arc(sStart.x, sStart.y, 6, 0, 2 * Math.PI);
  ctx.fill();
  ctx.fillStyle = "#10b981";
  ctx.fillText(isReturnMode ? "A (Start & Return Origin)" : "A (Start)", sStart.x + 8, sStart.y + 4);

  // 5. Continuous Sub-Step Trajectory & AUV Sprite Rendering
  if (simData && simData.trajectory && simData.trajectory.length > 0) {
    const traj = simData.trajectory;
    const fStep = (fracStep !== undefined) ? Math.max(0, Math.min(traj.length - 1, fracStep)) : currentFractionalStep;
    const i0 = Math.floor(fStep);
    const alpha = fStep - i0;
    const p0 = traj[i0];
    const p1 = traj[Math.min(traj.length - 1, i0 + 1)];

    // Interpolate continuous vehicle position and heading
    const dHdg = wrapAngleRad(p1.true_heading - p0.true_heading);
    const curN = p0.true_north + (p1.true_north - p0.true_north) * alpha;
    const curE = p0.true_east + (p1.true_east - p0.true_east) * alpha;
    const curHdg = p0.true_heading + dHdg * alpha;
    const curSpd = p0.true_speed + (p1.true_speed - p0.true_speed) * alpha;

    // Helper to identify Leg 2 (Return journey steps)
    const checkIsLeg2 = (p) => (p.leg === 2 || p.phase === 'RETURNING' || p.phase === 'ROTATING_BACK' || p.phase === 'ROTATING_AT_A');

    // 5A. Full Planned/Simulated Trajectory Trails
    // Outbound Leg 1: deep ocean cyan trail
    ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
    ctx.lineWidth = 2.0;
    ctx.setLineDash([]);
    ctx.beginPath();
    let hasLeg1Trail = false;
    for (let i = 1; i < traj.length; i++) {
      if (!checkIsLeg2(traj[i])) {
        const pPrev = worldToScreen(traj[i - 1].true_north, traj[i - 1].true_east);
        const pCurr = worldToScreen(traj[i].true_north, traj[i].true_east);
        if (!hasLeg1Trail) { ctx.moveTo(pPrev.x, pPrev.y); hasLeg1Trail = true; }
        ctx.lineTo(pCurr.x, pCurr.y);
      }
    }
    ctx.stroke();

    // Return Leg 2: vibrant dashed golden amber trail (instantly visible on full map)
    ctx.strokeStyle = "rgba(251, 191, 36, 0.85)";
    ctx.lineWidth = 2.8;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    let hasLeg2Trail = false;
    for (let i = 1; i < traj.length; i++) {
      if (checkIsLeg2(traj[i])) {
        const pPrev = worldToScreen(traj[i - 1].true_north, traj[i - 1].true_east);
        const pCurr = worldToScreen(traj[i].true_north, traj[i].true_east);
        if (!hasLeg2Trail) { ctx.moveTo(pPrev.x, pPrev.y); hasLeg2Trail = true; }
        ctx.lineTo(pCurr.x, pCurr.y);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // 5B. Traveled Path up to current step
    for (let i = 1; i <= i0; i++) {
      const pPrev = worldToScreen(traj[i - 1].true_north, traj[i - 1].true_east);
      const pCurr = worldToScreen(traj[i].true_north, traj[i].true_east);
      const isL2 = checkIsLeg2(traj[i]);
      
      if (isL2) {
        // High-contrast radiant amber glow underneath + golden dashed core
        ctx.strokeStyle = "rgba(245, 158, 11, 0.40)";
        ctx.lineWidth = 7.0;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(pPrev.x, pPrev.y);
        ctx.lineTo(pCurr.x, pCurr.y);
        ctx.stroke();

        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 3.5;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(pPrev.x, pPrev.y);
        ctx.lineTo(pCurr.x, pCurr.y);
        ctx.stroke();
        ctx.setLineDash([]);
      } else {
        // Solid cyan outbound line
        ctx.strokeStyle = "#0284c7";
        ctx.lineWidth = 3.5;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(pPrev.x, pPrev.y);
        ctx.lineTo(pCurr.x, pCurr.y);
        ctx.stroke();
      }
    }

    // Segment to interpolated current position
    if (i0 < traj.length - 1) {
      const pLast = worldToScreen(traj[i0].true_north, traj[i0].true_east);
      const pInterp = worldToScreen(curN, curE);
      const isL2 = checkIsLeg2(traj[i0]);
      
      if (isL2) {
        ctx.strokeStyle = "rgba(245, 158, 11, 0.40)";
        ctx.lineWidth = 7.0;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(pLast.x, pLast.y);
        ctx.lineTo(pInterp.x, pInterp.y);
        ctx.stroke();

        ctx.strokeStyle = "#fbbf24";
        ctx.lineWidth = 3.5;
        ctx.setLineDash([6, 3]);
      } else {
        ctx.strokeStyle = "#0284c7";
        ctx.lineWidth = 3.5;
        ctx.setLineDash([]);
      }
      ctx.beginPath();
      ctx.moveTo(pLast.x, pLast.y);
      ctx.lineTo(pInterp.x, pInterp.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 5C. Draw 180° In-Place Pivot Marker at Point B once reached
    if (simData.metrics && simData.metrics.step_reached_b && i0 >= simData.metrics.step_reached_b) {
      const stepB = simData.metrics.step_reached_b;
      const pB = worldToScreen(traj[stepB].true_north, traj[stepB].true_east);
      ctx.strokeStyle = "#fbbf24";
      ctx.lineWidth = 2.0;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.arc(pB.x, pB.y, 14, 0, 2 * Math.PI);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "#fbbf24";
      ctx.font = "bold 9px monospace";
      ctx.fillText("🔄 180° PIVOT", pB.x + 16, pB.y + 12);
    }

    const pInterp = worldToScreen(curN, curE);

    // 6. Draw 10-step MPC Lookahead Prediction Fan
    if (p0.predicted_traj && p0.predicted_traj.length > 0) {
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(pInterp.x, pInterp.y);
      p0.predicted_traj.forEach(pred => {
        const pPred = worldToScreen(pred[0], pred[1]);
        ctx.lineTo(pPred.x, pPred.y);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 7. Render AUV Submarine Sprite at exact continuous position & heading
    drawAUVSprite(ctx, pInterp.x, pInterp.y, curHdg, curSpd);
  } else {
    // Render idle AUV sprite at Start position pointing toward Goal
    const dx = sGoal.x - sStart.x;
    const dy = sGoal.y - sStart.y;
    const initHeading = Math.atan2(dx, -dy);
    drawAUVSprite(ctx, sStart.x, sStart.y, initHeading, 0.0);
  }
}

function drawAUVSprite(ctx, x, y, heading, speed) {
  ctx.save();
  ctx.translate(x, y);
  // Heading in marine radians is clockwise from North (+Y on screen is down, so angle adjustment)
  // heading = 0 is North (-Y). Screen angle = heading - PI/2
  ctx.rotate(heading - Math.PI / 2);

  // Submarine Hull (streamlined yellow/cyan exploration vehicle)
  ctx.fillStyle = "#38bdf8";
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 1.5;

  ctx.beginPath();
  // Nose
  ctx.ellipse(14, 0, 8, 4, 0, -Math.PI / 2, Math.PI / 2);
  // Main Body
  ctx.lineTo(-10, 5);
  ctx.lineTo(-14, 0);
  ctx.lineTo(-10, -5);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Tail Control Fins
  ctx.fillStyle = "#0284c7";
  ctx.beginPath();
  ctx.moveTo(-10, -5); ctx.lineTo(-15, -9); ctx.lineTo(-12, -4);
  ctx.moveTo(-10, 5); ctx.lineTo(-15, 9); ctx.lineTo(-12, 4);
  ctx.fill();

  // Heading Vector Arrow
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(14, 0);
  ctx.lineTo(28, 0);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(28, 0); ctx.lineTo(24, -3); ctx.lineTo(24, 3);
  ctx.closePath();
  ctx.fillStyle = "#10b981";
  ctx.fill();

  // Propeller Wake Particles
  if (speed > 0.2) {
    ctx.fillStyle = "rgba(56, 189, 248, 0.6)";
    ctx.beginPath();
    ctx.arc(-18, (Math.random() - 0.5) * 4, 2, 0, 2 * Math.PI);
    ctx.arc(-22, (Math.random() - 0.5) * 6, 1.5, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.restore();
}

// Side-View Depth Profile Renderer
function renderDepth(fracStep) {
  const ctx = depthCtx;
  const w = depthCanvas.width;
  const h = depthCanvas.height;
  ctx.clearRect(0, 0, w, h);

  const b = getBounds();
  const dMin = 0.0; // Sea surface
  const dMax = 70.0; // Seabed reference

  const marginX = 45;
  const marginY = 20;

  function toDepthScreen(depth, normX) {
    const scaleY = (h - 2 * marginY) / (dMax - dMin);
    return {
      x: marginX + normX * (w - 2 * marginX),
      y: marginY + (depth - dMin) * scaleY
    };
  }

  // Sea Surface Line (cyan water line)
  const surf = toDepthScreen(0.0, 0);
  ctx.strokeStyle = "#0284c7";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(marginX, surf.y);
  ctx.lineTo(w - marginX, surf.y);
  ctx.stroke();
  ctx.fillStyle = "#0284c7";
  ctx.font = "9px ui-monospace, Consolas, monospace";
  ctx.fillText("0m (SURFACE)", 4, surf.y + 3);

  // Depth Geofence Window (10m to 60m)
  const dTop = toDepthScreen(b.depth_min, 0);
  const dBot = toDepthScreen(b.depth_max, 0);
  ctx.fillStyle = "rgba(14, 165, 233, 0.05)";
  ctx.fillRect(marginX, dTop.y, w - 2 * marginX, dBot.y - dTop.y);
  ctx.strokeStyle = "rgba(56, 189, 248, 0.3)";
  ctx.setLineDash([4, 4]);
  ctx.strokeRect(marginX, dTop.y, w - 2 * marginX, dBot.y - dTop.y);
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(100, 116, 139, 0.7)";
  ctx.fillText(`${b.depth_min}m`, 8, dTop.y + 3);
  ctx.fillText(`${b.depth_max}m`, 8, dBot.y + 3);

  // Plot Depth Curve
  if (simData && simData.trajectory && simData.trajectory.length > 0) {
    const traj = simData.trajectory;
    const nPts = traj.length;
    const fStep = (fracStep !== undefined) ? Math.max(0, Math.min(nPts - 1, fracStep)) : currentFractionalStep;
    const i0 = Math.floor(fStep);
    const alpha = fStep - i0;
    const p0 = traj[i0];
    const p1 = traj[Math.min(nPts - 1, i0 + 1)];

    // Interpolate continuous depth and pitch
    const curDepth = p0.true_depth + (p1.true_depth - p0.true_depth) * alpha;
    const curPitch = p0.true_pitch + (p1.true_pitch - p0.true_pitch) * alpha;

    const checkDepthL2 = (p) => (p.leg === 2 || p.phase === 'RETURNING' || p.phase === 'ROTATING_BACK' || p.phase === 'ROTATING_AT_A');

    // Full depth trajectory faint trace
    for (let i = 1; i < nPts; i++) {
      const pPrev = toDepthScreen(traj[i - 1].true_depth, (i - 1) / (nPts - 1));
      const pCurr = toDepthScreen(traj[i].true_depth, i / (nPts - 1));
      const isL2 = checkDepthL2(traj[i]);
      ctx.strokeStyle = isL2 ? "rgba(251, 191, 36, 0.70)" : "rgba(56, 189, 248, 0.35)";
      ctx.lineWidth = isL2 ? 2.0 : 1.5;
      if (isL2) ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(pPrev.x, pPrev.y);
      ctx.lineTo(pCurr.x, pCurr.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Solid depth trace up to continuous fractional position distinguished by leg
    for (let i = 1; i <= i0; i++) {
      const pPrev = toDepthScreen(traj[i - 1].true_depth, (i - 1) / (nPts - 1));
      const pCurr = toDepthScreen(traj[i].true_depth, i / (nPts - 1));
      const isL2 = checkDepthL2(traj[i]);
      ctx.strokeStyle = isL2 ? "#fbbf24" : "#0284c7";
      ctx.lineWidth = isL2 ? 3.0 : 2.5;
      if (isL2) ctx.setLineDash([5, 3]); else ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(pPrev.x, pPrev.y);
      ctx.lineTo(pCurr.x, pCurr.y);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    // Draw milestone marker at Point B in depth plot if reached
    if (simData.metrics && simData.metrics.step_reached_b) {
      const stepB = simData.metrics.step_reached_b;
      const pBDepth = toDepthScreen(traj[stepB].true_depth, stepB / (nPts - 1));
      ctx.strokeStyle = "#fbbf24";
      ctx.fillStyle = "#fbbf24";
      ctx.beginPath();
      ctx.arc(pBDepth.x, pBDepth.y, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.font = "bold 9px monospace";
      ctx.fillText("Point B (Turnaround)", pBDepth.x - 40, Math.max(16, pBDepth.y - 8));
    }
    const curNormX = fStep / Math.max(1, nPts - 1);
    const curScreen = toDepthScreen(curDepth, curNormX);
    if (i0 < nPts - 1) {
      const pLast = toDepthScreen(traj[i0].true_depth, i0 / (nPts - 1));
      const isLeg2 = (traj[i0].leg === 2 || traj[i0].phase === 'RETURNING');
      ctx.strokeStyle = isLeg2 ? "#f59e0b" : "#38bdf8";
      ctx.lineWidth = 2.5;
      if (isLeg2) ctx.setLineDash([4, 2]);
      ctx.beginPath();
      ctx.moveTo(pLast.x, pLast.y);
      ctx.lineTo(curScreen.x, curScreen.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Current Depth Vehicle Node
    ctx.fillStyle = "#10b981";
    ctx.beginPath();
    ctx.arc(curScreen.x, curScreen.y, 5, 0, 2 * Math.PI);
    ctx.fill();

    // Pitch Angle Vector
    const pitchLen = 14;
    const dx = Math.cos(curPitch) * pitchLen;
    const dy = Math.sin(curPitch) * pitchLen;
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(curScreen.x, curScreen.y);
    ctx.lineTo(curScreen.x + dx, curScreen.y + dy);
    ctx.stroke();
  }
}

// Auto-Download Preference State & Toggles
function getAutoDownloadPref() {
  return localStorage.getItem('auto_download_report') === 'true';
}

function initAutoDownloadPref() {
  const pref = getAutoDownloadPref();
  const cb1 = document.getElementById('autoDownloadCheckbox');
  const cb2 = document.getElementById('fastAutoDownloadChk') || document.getElementById('autoDownloadSidebarCheckbox');
  if (cb1) cb1.checked = pref;
  if (cb2) cb2.checked = pref;
}

function toggleAutoDownload(enabled) {
  localStorage.setItem('auto_download_report', enabled ? 'true' : 'false');
  const cb1 = document.getElementById('autoDownloadCheckbox');
  const cb2 = document.getElementById('fastAutoDownloadChk') || document.getElementById('autoDownloadSidebarCheckbox');
  if (cb1) cb1.checked = enabled;
  if (cb2) cb2.checked = enabled;
}

// Post-Simulation Completion Modal Logic
function showSimCompleteModal() {
  if (!simData || !simData.metrics) return;
  const m = simData.metrics;
  const modal = document.getElementById('simCompleteModal');
  if (!modal) return;

  const outEl = document.getElementById('modalOutcome');
  if (m.success_5m) {
    outEl.innerText = "TARGET REACHED (≤5m)";
    outEl.style.color = "#34d399";
  } else {
    outEl.innerText = "CLOSEST APPROACH";
    outEl.style.color = "#fbbf24";
  }

  document.getElementById('modalFinalDist').innerText = `${m.final_distance_m.toFixed(2)} m`;
  document.getElementById('modalPathLen').innerText = `${m.total_path_length_m.toFixed(1)} m`;
  document.getElementById('modalPathEff').innerText = `${m.path_efficiency_pct.toFixed(1)} %`;
  document.getElementById('modalDuration').innerText = `${m.time_to_goal_sec.toFixed(1)} s (${simData.trajectory.length} steps)`;

  const safEl = document.getElementById('modalSafety');
  if (m.physical_collisions === 0 && m.buffer_violations === 0) {
    safEl.innerText = "0 Collisions (Safe)";
    safEl.style.color = "#34d399";
  } else if (m.physical_collisions === 0) {
    safEl.innerText = `${m.buffer_violations} Buffer Breaches`;
    safEl.style.color = "#fbbf24";
  } else {
    safEl.innerText = `${m.physical_collisions} Hull Collisions!`;
    safEl.style.color = "#f87171";
  }

  const cb = document.getElementById('autoDownloadCheckbox');
  if (cb) cb.checked = getAutoDownloadPref();

  modal.style.display = 'flex';
}

function closeSimCompleteModal() {
  const modal = document.getElementById('simCompleteModal');
  if (modal) modal.style.display = 'none';
}

// Category Page Navigation
function switchCategoryPage(pageId) {
  const pages = ['simulation', 'mathematics', 'telemetry_audit', 'reports'];
  pages.forEach(p => {
    const el = document.getElementById('page_' + p);
    const btn = document.getElementById('pageTab_' + p);
    if (el) el.classList.remove('active');
    if (btn) btn.classList.remove('active');
  });

  const targetPage = document.getElementById('page_' + pageId);
  const targetBtn = document.getElementById('pageTab_' + pageId);
  if (targetPage) targetPage.classList.add('active');
  if (targetBtn) targetBtn.classList.add('active');

  if (pageId === 'simulation') {
    setTimeout(resizeCanvases, 40);
  }
}

// Light / Slate Theme Toggle
function toggleTheme() {
  const body = document.body;
  const current = body.getAttribute('data-theme') || 'slate';
  const next = current === 'slate' ? 'light' : 'slate';
  body.setAttribute('data-theme', next);
  try {
    localStorage.setItem('autosub_theme', next);
  } catch (e) {}

  updateThemeToggleUI(next);
  renderAll();
}

function initTheme() {
  let saved = 'slate';
  try {
    saved = localStorage.getItem('autosub_theme') || 'slate';
  } catch (e) {}
  document.body.setAttribute('data-theme', saved);
  updateThemeToggleUI(saved);
}

function updateThemeToggleUI(theme) {
  const icon = document.getElementById('themeToggleIcon');
  const text = document.getElementById('themeToggleText');
  if (icon && text) {
    if (theme === 'light') {
      icon.innerText = '🌙';
      text.innerText = 'Slate Mode';
    } else {
      icon.innerText = '☀️';
      text.innerText = 'Light Mode';
    }
  }
}

// Tab Switching for Mathematical Proof Section (backwards compatibility)
function switchProofTab(tabId) {
  const btnIds = ['btnTabEquations', 'btnTabClearance', 'btnTabTable', 'btnTabMarkdown'];
  const paneIds = ['tabEquations', 'tabClearance', 'tabTable', 'tabMarkdownPreview'];

  btnIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });
  paneIds.forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.remove('active');
  });

  if (tabId === 'equations') {
    const btn = document.getElementById('btnTabEquations');
    const pane = document.getElementById('tabEquations');
    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
  } else if (tabId === 'clearance') {
    const btn = document.getElementById('btnTabClearance');
    const pane = document.getElementById('tabClearance');
    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
  } else if (tabId === 'table') {
    const btn = document.getElementById('btnTabTable');
    const pane = document.getElementById('tabTable');
    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
  } else if (tabId === 'markdown_preview') {
    const btn = document.getElementById('btnTabMarkdown');
    const pane = document.getElementById('tabMarkdownPreview');
    if (btn) btn.classList.add('active');
    if (pane) pane.classList.add('active');
  }
}

function copyProofMarkdown() {
  if (!simData || !simData.report_markdown) {
    alert("Please run a simulation first to generate calculations.");
    return;
  }
  navigator.clipboard.writeText(simData.report_markdown).then(() => {
    const btn = document.getElementById('copyBtnText');
    if (btn) {
      btn.innerText = "✓ Copied!";
      setTimeout(() => { btn.innerText = "📋 Copy Raw .MD"; }, 2000);
    }
  }).catch(() => {
    alert("Failed to copy to clipboard.");
  });
}

function renderMathProofSection(data) {
  if (!data || !data.metrics || !data.trajectory) return;
  const m = data.metrics;
  const proof = data.proof_data || {};
  const traj = data.trajectory;

  // 1. KPI Strip
  const statusEl = document.getElementById('proofKpiStatus');
  const badgeEl = document.getElementById('proofHeaderBadge');
  if (m.physical_collisions === 0) {
    if (statusEl) {
      statusEl.innerText = "PROVEN NON-COLLIDING (PASS)";
      statusEl.style.color = "#34d399";
    }
    if (badgeEl) {
      badgeEl.innerText = "PROVEN ZERO COLLISIONS";
      badgeEl.style.borderColor = "rgba(16, 185, 129, 0.4)";
      badgeEl.style.color = "#34d399";
    }
  } else {
    if (statusEl) {
      statusEl.innerText = `${m.physical_collisions} HULL COLLISION(S)`;
      statusEl.style.color = "#f43f5e";
    }
    if (badgeEl) {
      badgeEl.innerText = "COLLISION VIOLATION";
      badgeEl.style.borderColor = "rgba(244, 63, 94, 0.4)";
      badgeEl.style.color = "#f43f5e";
    }
  }

  const clearanceEl = document.getElementById('proofKpiMinClearance');
  if (clearanceEl) {
    if (m.min_obstacle_distance_m < 900) {
      clearanceEl.innerText = `${m.min_obstacle_distance_m.toFixed(2)} m`;
    } else {
      clearanceEl.innerText = "Clear (>100m)";
    }
  }

  const finalDistEl = document.getElementById('proofKpiFinalDist');
  if (finalDistEl) {
    finalDistEl.innerText = `${m.final_distance_m.toFixed(2)} m (${m.success_5m ? 'Converged' : 'Approach'})`;
  }
  
  const workEl = document.getElementById('proofKpiWork');
  if (workEl) {
    const work = proof.summary ? proof.summary.total_energy_proxy : 0.0;
    workEl.innerText = `${work.toFixed(2)} (m/s)²·s`;
  }

  const effEl = document.getElementById('proofKpiEfficiency');
  if (effEl) {
    effEl.innerText = `${m.path_efficiency_pct.toFixed(1)} %`;
  }

  // 2. Obstacles Clearance Table (Tab 2)
  const obsTbody = document.getElementById('proofObstaclesTableBody');
  if (obsTbody) {
    const obsAudit = proof.obstacles_audit || [];
    if (obsAudit.length === 0) {
      obsTbody.innerHTML = `
        <tr>
          <td colspan="8" style="text-align: center; color: var(--cyan); padding: 16px;">
            ✓ Open Water Corridor: No active hazards registered inside the operational bounds.
          </td>
        </tr>
      `;
    } else {
      obsTbody.innerHTML = obsAudit.map(oa => `
        <tr>
          <td><strong>#${oa.obs_id}</strong></td>
          <td>(${oa.north.toFixed(1)}N, ${oa.east.toFixed(1)}E)</td>
          <td>${oa.radius.toFixed(1)} m</td>
          <td>${oa.safe_radius.toFixed(1)} m</td>
          <td style="color: ${oa.min_distance_m <= oa.radius ? '#f43f5e' : (oa.min_distance_m <= oa.safe_radius ? '#fbbf24' : '#34d399')}; font-weight: 700;">
            ${oa.min_distance_m.toFixed(2)} m
          </td>
          <td style="color: ${oa.clearance_margin_m > 0 ? '#34d399' : '#f43f5e'}; font-weight: 700;">
            ${oa.clearance_margin_m >= 0 ? '+' : ''}${oa.clearance_margin_m.toFixed(2)} m
          </td>
          <td>Step ${oa.min_step} (${(oa.min_step * 2.0).toFixed(1)}s)</td>
          <td>
            <span class="${oa.has_collision ? 'proof-badge-fail' : (oa.has_buffer_breach ? 'proof-badge-warning' : 'proof-badge-pass')}">
              ${oa.proof_verdict}
            </span>
          </td>
        </tr>
      `).join('');
    }
  }

  // 3. Step-by-Step Computational Values Table (Tab 3)
  const fullTbody = document.getElementById('proofFullTableBody');
  const countEl = document.getElementById('proofTableStepCount');
  if (countEl) countEl.innerText = traj.length;

  if (fullTbody) {
    fullTbody.innerHTML = traj.map((r, idx) => {
      const tSec = r.time_sec !== undefined ? r.time_sec : idx * 2.0;
      const tn = r.true_north !== undefined ? r.true_north.toFixed(1) : '0.0';
      const te = r.true_east !== undefined ? r.true_east.toFixed(1) : '0.0';
      const td = r.true_depth !== undefined ? r.true_depth.toFixed(1) : '0.0';
      const spd = r.true_speed !== undefined ? r.true_speed.toFixed(2) : '1.20';
      
      const roll = (r.true_roll * 180 / Math.PI).toFixed(0);
      const pitch = (r.true_pitch * 180 / Math.PI).toFixed(1);
      const yaw = ((r.true_heading * 180 / Math.PI + 360) % 360).toFixed(1);

      const cmdU = r.cmd_speed !== null && !isNaN(r.cmd_speed) ? r.cmd_speed.toFixed(2) : '-';
      const cmdP = r.cmd_pitch !== null && !isNaN(r.cmd_pitch) ? (r.cmd_pitch * 180 / Math.PI).toFixed(1) : '-';
      const cmdH = r.cmd_heading !== null && !isNaN(r.cmd_heading) ? ((r.cmd_heading * 180 / Math.PI + 360) % 360).toFixed(1) : '-';
      const cmdStr = cmdU !== '-' ? `[${cmdU}, ${cmdP}°, ${cmdH}°]` : '[Terminal]';

      const gDist = r.dist_to_goal !== undefined ? r.dist_to_goal.toFixed(2) : '0.00';
      const obsD = r.min_obstacle_dist !== undefined && r.min_obstacle_dist < 900 ? r.min_obstacle_dist.toFixed(2) : '>100';
      
      const clearanceVal = obsD !== '>100' ? (r.min_obstacle_dist - 8.0).toFixed(2) : '>90';

      const jg = r.cost_j_goal !== undefined ? r.cost_j_goal.toFixed(1) : '0.0';
      const jt = r.cost_j_tracking !== undefined ? r.cost_j_tracking.toFixed(1) : '0.0';
      const jc = r.cost_j_control !== undefined ? r.cost_j_control.toFixed(2) : '0.0';
      const js = r.cost_j_smooth !== undefined ? r.cost_j_smooth.toFixed(1) : '0.0';
      const jb = r.cost_j_constraint !== undefined ? r.cost_j_constraint.toFixed(1) : '0.0';
      const jtot = r.cost_total_cost !== undefined ? r.cost_total_cost.toFixed(1) : '0.0';
      const latMs = r.solve_time_sec !== undefined ? (r.solve_time_sec * 1000).toFixed(0) : '0';

      let verdict = '<span class="proof-badge-pass">SAFE</span>';
      if (r.collision) verdict = '<span class="proof-badge-fail">COLLISION</span>';
      else if (r.buffer_violation) verdict = '<span class="proof-badge-warning">BUFFER</span>';
      else if (r.is_in_bounds === false) verdict = '<span class="proof-badge-warning">OOB</span>';

      return `
        <tr>
          <td>${r.step !== undefined ? r.step : idx}</td>
          <td>${tSec.toFixed(1)}</td>
          <td>[${tn}, ${te}, ${td}]</td>
          <td>${spd}</td>
          <td>[${roll}°, ${pitch}°, ${yaw}°]</td>
          <td>${cmdStr}</td>
          <td>${gDist}</td>
          <td>${obsD}</td>
          <td style="color: ${parseFloat(clearanceVal) >= 0 ? '#34d399' : '#f43f5e'}; font-weight: 600;">${clearanceVal}</td>
          <td>${jg}</td>
          <td>${jt}</td>
          <td>${jc}</td>
          <td>${js}</td>
          <td>${jb}</td>
          <td style="color: var(--cyan); font-weight: 600;">${jtot}</td>
          <td>${latMs}</td>
          <td>${verdict}</td>
        </tr>
      `;
    }).join('');
  }

  // 4. Markdown File Preview (Tab 4)
  const codeEl = document.getElementById('proofMarkdownCode');
  if (codeEl && data.report_markdown) {
    codeEl.innerText = data.report_markdown;
  }
}

// Direct File Download Utility (Zero Blob UUIDs, Guaranteed Filenames & MIME Types)
function downloadFileDirect(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  if (filename) a.setAttribute('download', filename);
  a.style.display = 'none';
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    if (a.parentNode) a.parentNode.removeChild(a);
  }, 200);
}

// Download Documentation & Telemetry
function downloadCalculationDoc() {
  if (!simData) {
    alert("Please run a simulation first before downloading the calculation report.");
    return;
  }
  downloadFileDirect('/api/download_report_md', 'mission_calculation_report.md');
}

function downloadCalculationTxt() {
  if (!simData) {
    alert("Please run a simulation first before downloading the text report.");
    return;
  }
  downloadFileDirect('/api/download_report_txt', 'mission_calculation_report.txt');
}

function downloadCalculationHTML() {
  if (!simData) {
    alert("Please run a simulation first before downloading the publication report.");
    return;
  }
  downloadFileDirect('/api/download_report_html', 'mission_calculation_report.html');
}

function downloadTelemetryCSV() {
  if (!simData) {
    alert("Please run a simulation first before downloading the telemetry CSV.");
    return;
  }
  downloadFileDirect('/api/download_telemetry_csv', 'mission_telemetry_calculations.csv');
}

function downloadMetricsJSON() {
  if (!simData) {
    alert("Please run a simulation first before downloading the metrics JSON.");
    return;
  }
  downloadFileDirect('/api/download_metrics_json', 'autosub6000_mission_metrics.json');
}
