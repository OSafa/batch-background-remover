// State
const state = {
  items: [],
  selectedIndex: -1,
  viewMode: 'side',
  reprocessTimeout: null,
  activeDetectAbortController: null, // Cancels previous YOLO requests when clicking through thumbnails
  cropState: {
    isDragging: false,
    dragAction: null,
    startX: 0,
    startY: 0,
    boxStartLeft: 0,
    boxStartTop: 0,
    boxStartWidth: 0,
    boxStartHeight: 0,
  },
};

// DOM Elements
const dropzone = document.getElementById('dropzone');
const previewStage = document.getElementById('previewStage');
const fileInput = document.getElementById('fileInput');
const folderInput = document.getElementById('folderInput');
const thumbnailTrack = document.getElementById('thumbnailTrack');
const batchStatusText = document.getElementById('batchStatusText');
const batchCountBadge = document.getElementById('batchCountBadge');
const processAllBtn = document.getElementById('processAllBtn');
const downloadZipBtn = document.getElementById('downloadZipBtn');
const downloadCurrentBtn = document.getElementById('downloadCurrentBtn');
const clearBatchBtn = document.getElementById('clearBatchBtn');

// View elements
const sideBySideContainer = document.getElementById('sideBySideContainer');
const sideOrigImg = document.getElementById('sideOrigImg');
const sideCutoutImg = document.getElementById('sideCutoutImg');
const processedSideLabel = document.getElementById('processedSideLabel');
const cropHost = document.getElementById('cropHost');
const cropBox = document.getElementById('cropBox');
const cropBoxLabel = document.getElementById('cropBoxLabel');
const cropBadge = document.getElementById('cropBadge');

const singleViewContainer = document.getElementById('singleViewContainer');
const singleImgBox = document.getElementById('singleImgBox');
const singleViewImg = document.getElementById('singleViewImg');

// Badges
const dimensionsBadge = document.getElementById('dimensionsBadge');
const timeBadge = document.getElementById('timeBadge');

// Controls
const maxHeightInput = document.getElementById('maxHeight');
const maxHeightNum = document.getElementById('maxHeightNum');
const featherRadiusInput = document.getElementById('featherRadius');
const featherRadiusVal = document.getElementById('featherRadiusVal');
const cropMarginInput = document.getElementById('cropMargin');
const cropMarginVal = document.getElementById('cropMarginVal');
const defringeToggle = document.getElementById('defringeToggle');
const mipmapsToggle = document.getElementById('mipmapsToggle');
const outputFolderInput = document.getElementById('outputFolder');

const autoTorsoToggle = document.getElementById('autoTorsoToggle');
const torsoPresetSelect = document.getElementById('torsoPreset');

// Initialize
function init() {
  setupSliderAndNumberControls();
  setupDragAndDrop();
  setupViewModeButtons();
  setupPresets();
  setupCropInteraction();
  setupTorsoControls();
  setupProcessActions();
}

// Sliders and Number Inputs
function setupSliderAndNumberControls() {
  const triggerLiveUpdate = () => {
    if (state.reprocessTimeout) clearTimeout(state.reprocessTimeout);
    state.reprocessTimeout = setTimeout(() => {
      liveReprocessActiveItem();
    }, 120);
  };

  maxHeightInput.addEventListener('input', (e) => {
    const val = parseInt(e.target.value, 10);
    maxHeightNum.value = val;
    updatePresetButtons(val);
    triggerLiveUpdate();
  });

  maxHeightNum.addEventListener('input', (e) => {
    let val = parseInt(e.target.value, 10);
    if (isNaN(val) || val < 16) val = 16;
    if (val > 4096) val = 4096;
    if (val <= parseInt(maxHeightInput.max, 10) && val >= parseInt(maxHeightInput.min, 10)) {
      maxHeightInput.value = val;
    }
    updatePresetButtons(val);
    triggerLiveUpdate();
  });

  maxHeightNum.addEventListener('change', (e) => {
    let val = parseInt(e.target.value, 10);
    if (isNaN(val) || val < 16) val = 340;
    maxHeightNum.value = val;
    if (val <= parseInt(maxHeightInput.max, 10) && val >= parseInt(maxHeightInput.min, 10)) {
      maxHeightInput.value = val;
    }
    updatePresetButtons(val);
    triggerLiveUpdate();
  });

  featherRadiusInput.addEventListener('input', (e) => {
    featherRadiusVal.textContent = `${parseFloat(e.target.value).toFixed(1)} px`;
    triggerLiveUpdate();
  });

  cropMarginInput.addEventListener('input', (e) => {
    cropMarginVal.textContent = `${e.target.value} px`;
    triggerLiveUpdate();
  });

  defringeToggle.addEventListener('change', triggerLiveUpdate);
  mipmapsToggle.addEventListener('change', triggerLiveUpdate);
  outputFolderInput.addEventListener('change', triggerLiveUpdate);
}

function updatePresetButtons(currentVal) {
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.classList.toggle('active', parseInt(btn.dataset.val, 10) === currentVal);
  });
}

function setupPresets() {
  document.querySelectorAll('.preset-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const val = parseInt(btn.dataset.val, 10);
      maxHeightInput.value = val;
      maxHeightNum.value = val;
      updatePresetButtons(val);
      liveReprocessActiveItem();
    });
  });
}

// Torso Controls
function setupTorsoControls() {
  autoTorsoToggle.addEventListener('change', () => {
    const isAuto = autoTorsoToggle.checked;
    if (!isAuto) {
      cropBox.classList.remove('visible');
      cropBadge.textContent = 'Pre-Crop Off';
      liveReprocessActiveItem(null);
    } else {
      detectAndApplyTorsoCrop();
    }
  });

  torsoPresetSelect.addEventListener('change', () => {
    const preset = torsoPresetSelect.value;
    updateCropBoxLabel(preset);
    if (autoTorsoToggle.checked) {
      detectAndApplyTorsoCrop();
    }
  });
}

function updateCropBoxLabel(preset) {
  const labels = {
    waist: 'Waist Crop',
    mid_thigh: 'Mid-Thigh Crop',
    bust: 'Bust Crop',
    custom: 'Custom Crop',
    none: 'Full Image',
  };
  const txt = labels[preset] || 'Torso Crop';
  if (cropBoxLabel) cropBoxLabel.textContent = txt;
  if (cropBadge) {
    cropBadge.textContent = preset === 'custom' ? 'Custom Crop (Adjusted)' : `Preset: ${txt}`;
  }
}

// Runs YOLO ONLY for the currently selected single image
async function detectAndApplyTorsoCrop(targetItem = null) {
  const item = targetItem || (state.selectedIndex >= 0 ? state.items[state.selectedIndex] : null);
  if (!item) return;

  const preset = torsoPresetSelect.value;
  updateCropBoxLabel(preset);

  if (preset === 'none') {
    cropBox.classList.remove('visible');
    item.cropBox = null;
    item.normalizedCropBox = [0, 0, 1, 1];
    if (item.status === 'done') {
      liveReprocessActiveItem(null);
    }
    return;
  }

  // Abort any prior in-flight YOLO detection request (e.g. if user quickly clicked through thumbnails)
  if (state.activeDetectAbortController) {
    state.activeDetectAbortController.abort();
  }
  state.activeDetectAbortController = new AbortController();

  try {
    const formData = new FormData();
    formData.append('preset', preset);
    formData.append('item_id', item.id);
    if (item.file) {
      formData.append('file', item.file);
    }

    const res = await fetch('/api/detect_crop', {
      method: 'POST',
      body: formData,
      signal: state.activeDetectAbortController.signal,
    });

    if (!res.ok) return;
    const data = await res.json();
    item.cropBox = data.crop_box;
    item.normalizedCropBox = data.normalized_box;

    // Only update UI if this item is STILL the currently selected item
    if (state.selectedIndex >= 0 && state.items[state.selectedIndex].id === item.id) {
      updateCropBoxDisplay();
    }
    
    // Only reprocess if the item was already processed before
    if (item.status === 'done') {
      liveReprocessActiveItem(item.cropBox);
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.warn('Torso detect error:', err);
    }
  } finally {
    state.activeDetectAbortController = null;
  }
}

// Interactive Draggable Crop Box
function setupCropInteraction() {
  cropBox.addEventListener('mousedown', (e) => {
    const handle = e.target.getAttribute('data-handle');
    startDrag(e.clientX, e.clientY, handle || 'move');
    e.stopPropagation();
  });

  window.addEventListener('mousemove', (e) => {
    if (!state.cropState.isDragging) return;
    handleDrag(e.clientX, e.clientY);
  });

  window.addEventListener('mouseup', () => {
    if (state.cropState.isDragging) {
      stopDrag();
    }
  });

  // Touch Support
  cropBox.addEventListener('touchstart', (e) => {
    if (e.touches.length > 0) {
      const t = e.touches[0];
      const handle = e.target.getAttribute('data-handle');
      startDrag(t.clientX, t.clientY, handle || 'move');
    }
    e.stopPropagation();
  });

  window.addEventListener('touchmove', (e) => {
    if (state.cropState.isDragging && e.touches.length > 0) {
      handleDrag(e.touches[0].clientX, e.touches[0].clientY);
    }
  });

  window.addEventListener('touchend', () => {
    if (state.cropState.isDragging) {
      stopDrag();
    }
  });

  sideOrigImg.addEventListener('load', () => {
    updateCropBoxDisplay();
  });
  window.addEventListener('resize', () => {
    updateCropBoxDisplay();
  });
}

function startDrag(clientX, clientY, action) {
  state.cropState.isDragging = true;
  state.cropState.dragAction = action;
  state.cropState.startX = clientX;
  state.cropState.startY = clientY;
  state.cropState.boxStartLeft = parseFloat(cropBox.style.left) || 0;
  state.cropState.boxStartTop = parseFloat(cropBox.style.top) || 0;
  state.cropState.boxStartWidth = parseFloat(cropBox.style.width) || cropBox.offsetWidth;
  state.cropState.boxStartHeight = parseFloat(cropBox.style.height) || cropBox.offsetHeight;
}

function handleDrag(clientX, clientY) {
  const hostRect = sideOrigImg.getBoundingClientRect();
  const maxW = hostRect.width;
  const maxH = hostRect.height;
  if (maxW <= 0 || maxH <= 0) return;

  const dx = clientX - state.cropState.startX;
  const dy = clientY - state.cropState.startY;
  const action = state.cropState.dragAction;

  let newLeft = state.cropState.boxStartLeft;
  let newTop = state.cropState.boxStartTop;
  let newW = state.cropState.boxStartWidth;
  let newH = state.cropState.boxStartHeight;

  const minSize = 25;

  if (action === 'move') {
    newLeft = Math.max(0, Math.min(maxW - newW, newLeft + dx));
    newTop = Math.max(0, Math.min(maxH - newH, newTop + dy));
  } else {
    if (action.includes('w')) {
      const right = newLeft + newW;
      newLeft = Math.max(0, Math.min(right - minSize, newLeft + dx));
      newW = right - newLeft;
    }
    if (action.includes('e')) {
      newW = Math.max(minSize, Math.min(maxW - newLeft, newW + dx));
    }
    if (action.includes('n')) {
      const bottom = newTop + newH;
      newTop = Math.max(0, Math.min(bottom - minSize, newTop + dy));
      newH = bottom - newTop;
    }
    if (action.includes('s')) {
      newH = Math.max(minSize, Math.min(maxH - newTop, newH + dy));
    }
  }

  cropBox.style.left = `${newLeft}px`;
  cropBox.style.top = `${newTop}px`;
  cropBox.style.width = `${newW}px`;
  cropBox.style.height = `${newH}px`;

  torsoPresetSelect.value = 'custom';
  updateCropBoxLabel('custom');
}

function stopDrag() {
  state.cropState.isDragging = false;
  state.cropState.dragAction = null;

  if (state.selectedIndex < 0 || state.selectedIndex >= state.items.length) return;
  const item = state.items[state.selectedIndex];
  if (!item) return;

  const hostRect = sideOrigImg.getBoundingClientRect();
  const maxW = hostRect.width;
  const maxH = hostRect.height;
  if (maxW <= 0 || maxH <= 0) return;

  const boxL = parseFloat(cropBox.style.left) || 0;
  const boxT = parseFloat(cropBox.style.top) || 0;
  const boxW = parseFloat(cropBox.style.width) || 0;
  const boxH = parseFloat(cropBox.style.height) || 0;

  const nx1 = Math.max(0, Math.min(1, boxL / maxW));
  const ny1 = Math.max(0, Math.min(1, boxT / maxH));
  const nx2 = Math.max(0, Math.min(1, (boxL + boxW) / maxW));
  const ny2 = Math.max(0, Math.min(1, (boxT + boxH) / maxH));

  item.normalizedCropBox = [nx1, ny1, nx2, ny2];

  if (item.origWidth && item.origHeight) {
    item.cropBox = [
      Math.round(nx1 * item.origWidth),
      Math.round(ny1 * item.origHeight),
      Math.round(nx2 * item.origWidth),
      Math.round(ny2 * item.origHeight),
    ];
  }

  torsoPresetSelect.value = 'custom';
  updateCropBoxLabel('custom');

  if (item.status === 'done') {
    liveReprocessActiveItem(item.cropBox);
  }
}

function updateCropBoxDisplay() {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.items.length) {
    cropBox.classList.remove('visible');
    return;
  }
  const item = state.items[state.selectedIndex];
  if (!item || !autoTorsoToggle.checked || torsoPresetSelect.value === 'none') {
    cropBox.classList.remove('visible');
    return;
  }

  const hostW = sideOrigImg.offsetWidth;
  const hostH = sideOrigImg.offsetHeight;
  if (hostW <= 0 || hostH <= 0) return;

  const norm = item.normalizedCropBox || [0, 0, 1, 1];
  const boxL = norm[0] * hostW;
  const boxT = norm[1] * hostH;
  const boxW = (norm[2] - norm[0]) * hostW;
  const boxH = (norm[3] - norm[1]) * hostH;

  cropBox.style.left = `${boxL}px`;
  cropBox.style.top = `${boxT}px`;
  cropBox.style.width = `${boxW}px`;
  cropBox.style.height = `${boxH}px`;
  cropBox.classList.add('visible');
  updateCropBoxLabel(torsoPresetSelect.value);
}

// Live real-time re-processing
async function liveReprocessActiveItem(customCropBox = undefined) {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.items.length) return;
  const item = state.items[state.selectedIndex];
  if (!item || item.status !== 'done') return;

  const targetHeight = parseInt(maxHeightNum.value, 10) || 340;
  const boxToSend = customCropBox !== undefined ? customCropBox : item.cropBox;

  try {
    timeBadge.textContent = 'Updating...';
    const startTime = performance.now();

    const res = await fetch('/api/reprocess', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        item_id: item.id,
        max_height: targetHeight,
        margin: parseInt(cropMarginInput.value, 10),
        feather_radius: parseFloat(featherRadiusInput.value),
        defringe: defringeToggle.checked,
        generate_mipmaps: mipmapsToggle.checked,
        output_folder: outputFolderInput.value.trim(),
        crop_box: boxToSend,
      }),
    });

    if (!res.ok) return;

    const data = await res.json();
    item.cutoutUrl = data.cutout_data_url;
    item.maskUrl = data.mask_data_url;
    item.ddsFilename = data.dds_filename;
    item.ddsDownloadUrl = data.dds_download_url;
    item.width = data.width;
    item.height = data.height;
    item.cropBox = data.crop_box;
    item.normalizedCropBox = data.normalized_crop_box;
    item.timeElapsed = (performance.now() - startTime) / 1000;

    renderActivePreview();
    renderThumbnails();
    updateCropBoxDisplay();
  } catch (err) {
    console.warn('Live reprocess error:', err);
  }
}

// Drag & Drop
function setupDragAndDrop() {
  ['dragenter', 'dragover'].forEach(name => {
    window.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    window.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  window.addEventListener('drop', (e) => {
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      handleFilesAdded(Array.from(e.dataTransfer.files));
    }
  });

  fileInput.addEventListener('change', (e) => {
    handleFilesAdded(Array.from(e.target.files));
    fileInput.value = '';
  });

  folderInput.addEventListener('change', (e) => {
    handleFilesAdded(Array.from(e.target.files));
    folderInput.value = '';
  });

  clearBatchBtn.addEventListener('click', () => {
    state.items = [];
    state.selectedIndex = -1;
    updateBatchUI();
  });
}

function handleFilesAdded(files) {
  const validExtensions = /\.(jpe?g|png|webp|bmp|tiff?)$/i;
  const imageFiles = files.filter(f => validExtensions.test(f.name));

  if (imageFiles.length === 0) return;

  const isInitialBatch = state.items.length === 0;

  imageFiles.forEach(file => {
    const item = {
      id: Math.random().toString(36).substring(2, 9),
      file: file,
      name: file.name,
      originalUrl: URL.createObjectURL(file),
      cutoutUrl: null,
      maskUrl: null,
      cropBox: null, // Null until clicked or processed (LAZY)
      normalizedCropBox: [0, 0, 1, 1],
      origWidth: 0,
      origHeight: 0,
      ddsFilename: null,
      ddsDownloadUrl: null,
      width: 0,
      height: 0,
      status: 'queued',
      timeElapsed: null,
    };

    // Preload image dimensions locally in browser without sending network request
    const probe = new Image();
    probe.src = item.originalUrl;
    probe.onload = () => {
      item.origWidth = probe.naturalWidth;
      item.origHeight = probe.naturalHeight;
    };

    state.items.push(item);
  });

  if (state.selectedIndex === -1 && state.items.length > 0) {
    state.selectedIndex = 0;
  }

  updateBatchUI();

  // ONLY run YOLO for the SINGLE image that is currently selected in the preview!
  // All other batch images remain idle in the queue without running YOLO.
  if (isInitialBatch && state.items.length > 0 && autoTorsoToggle.checked) {
    detectAndApplyTorsoCrop(state.items[0]);
  }
}

function updateBatchUI() {
  const count = state.items.length;
  batchCountBadge.textContent = count;
  batchStatusText.textContent = `${count} item${count === 1 ? '' : 's'} in queue`;
  processAllBtn.disabled = count === 0;

  const hasDoneItems = state.items.some(it => it.status === 'done');
  downloadZipBtn.disabled = !hasDoneItems;

  if (count === 0) {
    dropzone.classList.remove('hidden');
    previewStage.classList.add('hidden');
    thumbnailTrack.innerHTML = '';
    downloadCurrentBtn.disabled = true;
    dimensionsBadge.textContent = '0 × 0 px';
    timeBadge.textContent = 'Ready';
    return;
  }

  dropzone.classList.add('hidden');
  previewStage.classList.remove('hidden');

  renderThumbnails();
  renderActivePreview();
}

function renderThumbnails() {
  thumbnailTrack.innerHTML = '';
  state.items.forEach((item, index) => {
    const card = document.createElement('div');
    card.className = `thumb-card ${index === state.selectedIndex ? 'active' : ''}`;
    
    const statusDot = document.createElement('span');
    statusDot.className = `thumb-status ${item.status}`;
    
    const img = document.createElement('img');
    img.src = item.cutoutUrl || item.originalUrl;
    img.alt = item.name;

    card.appendChild(img);
    card.appendChild(statusDot);

    card.addEventListener('click', () => {
      state.selectedIndex = index;
      renderThumbnails();
      renderActivePreview();
      
      // LAZY EVALUATION: ONLY run YOLO for this single image when clicked (if not already detected)
      if (!item.cropBox && autoTorsoToggle.checked) {
        detectAndApplyTorsoCrop(item);
      } else {
        updateCropBoxDisplay();
      }
    });

    thumbnailTrack.appendChild(card);
  });
}

function renderActivePreview() {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.items.length) return;
  const item = state.items[state.selectedIndex];

  sideOrigImg.src = item.originalUrl;

  if (item.cutoutUrl) {
    sideCutoutImg.src = item.cutoutUrl;
    downloadCurrentBtn.disabled = false;
    processedSideLabel.textContent = `Processed Cutout (${item.height}px, BC7)`;
  } else {
    sideCutoutImg.src = item.originalUrl;
    downloadCurrentBtn.disabled = true;
    processedSideLabel.textContent = 'Processed Cutout (Pending)';
  }

  if (item.width && item.height) {
    dimensionsBadge.textContent = `${item.width} × ${item.height} px`;
  } else {
    dimensionsBadge.textContent = 'Original';
  }

  if (item.timeElapsed) {
    timeBadge.textContent = `${item.timeElapsed.toFixed(2)} s`;
  } else {
    timeBadge.textContent = 'Ready';
  }

  applyViewMode();
}

// View Modes: side, cutout, orig, mask
function setupViewModeButtons() {
  document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      state.viewMode = btn.dataset.mode;
      applyViewMode();
    });
  });
}

function applyViewMode() {
  if (state.selectedIndex < 0 || state.selectedIndex >= state.items.length) return;
  const item = state.items[state.selectedIndex];
  const mode = state.viewMode;

  if (mode === 'side') {
    sideBySideContainer.classList.remove('hidden');
    singleViewContainer.classList.add('hidden');
    updateCropBoxDisplay();
  } else {
    sideBySideContainer.classList.add('hidden');
    singleViewContainer.classList.remove('hidden');

    if (mode === 'cutout') {
      singleImgBox.className = 'single-img-box checkerboard-bg';
      singleViewImg.src = item.cutoutUrl || item.originalUrl;
    } else if (mode === 'orig') {
      singleImgBox.className = 'single-img-box';
      singleViewImg.src = item.originalUrl;
    } else if (mode === 'mask') {
      singleImgBox.className = 'single-img-box';
      singleViewImg.src = item.maskUrl || item.originalUrl;
    }
  }
}

// Processing Actions: Runs sequentially one-by-one during Process Batch
function setupProcessActions() {
  processAllBtn.addEventListener('click', async () => {
    processAllBtn.disabled = true;
    for (let i = 0; i < state.items.length; i++) {
      const item = state.items[i];
      if (item.status === 'done') continue;

      item.status = 'processing';
      renderThumbnails();

      try {
        await processSingleItem(item);
        item.status = 'done';
      } catch (err) {
        console.error('Process error:', err);
        item.status = 'error';
      }

      renderThumbnails();
      if (i === state.selectedIndex) {
        renderActivePreview();
        updateCropBoxDisplay();
      }
    }
    processAllBtn.disabled = false;
    updateBatchUI();
  });

  downloadCurrentBtn.addEventListener('click', () => {
    const item = state.items[state.selectedIndex];
    if (item && item.ddsDownloadUrl) {
      const a = document.createElement('a');
      a.href = item.ddsDownloadUrl;
      a.download = item.ddsFilename || `${item.name.split('.')[0]}.dds`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  });

  downloadZipBtn.addEventListener('click', async () => {
    const completedItems = state.items.filter(it => it.status === 'done' && it.ddsFilename);
    if (completedItems.length === 0) return;

    downloadZipBtn.disabled = true;
    downloadZipBtn.textContent = '⏳ Zipping...';

    try {
      const response = await fetch('/api/export_batch_zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filenames: completedItems.map(it => it.ddsFilename)
        }),
      });

      if (!response.ok) throw new Error('Zip generation failed');

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `portraits_bc7_${Date.now()}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert('Failed to download ZIP: ' + err.message);
    } finally {
      downloadZipBtn.disabled = false;
      downloadZipBtn.textContent = '📦 Download All (.ZIP)';
    }
  });
}

async function processSingleItem(item) {
  const targetHeight = parseInt(maxHeightNum.value, 10) || 340;

  const formData = new FormData();
  formData.append('file', item.file);
  formData.append('item_id', item.id);
  formData.append('max_height', targetHeight);
  formData.append('margin', cropMarginInput.value);
  formData.append('feather_radius', featherRadiusInput.value);
  formData.append('defringe', defringeToggle.checked);
  formData.append('generate_mipmaps', mipmapsToggle.checked);
  formData.append('output_folder', outputFolderInput.value.trim());
  formData.append('auto_torso_crop', autoTorsoToggle.checked);
  formData.append('torso_preset', torsoPresetSelect.value);

  // If user adjusted this image's crop in the preview, send its exact crop box
  if (item.cropBox) {
    formData.append('crop_box_json', JSON.stringify(item.cropBox));
  }

  const startTime = performance.now();
  const res = await fetch('/api/process_single', {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const errData = await res.json().catch(() => ({}));
    throw new Error(errData.detail || 'Processing failed');
  }

  const data = await res.json();
  item.cutoutUrl = data.cutout_data_url;
  item.maskUrl = data.mask_data_url;
  item.ddsFilename = data.dds_filename;
  item.ddsDownloadUrl = data.dds_download_url;
  item.width = data.width;
  item.height = data.height;
  item.origWidth = data.orig_width;
  item.origHeight = data.orig_height;
  item.cropBox = data.crop_box;
  item.normalizedCropBox = data.normalized_crop_box;
  item.timeElapsed = (performance.now() - startTime) / 1000;
}

// Start app
window.addEventListener('DOMContentLoaded', init);
