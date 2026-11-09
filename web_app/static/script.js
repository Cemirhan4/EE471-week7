/**
 * EE471 Week 7 — AI Classifier Studio Controller
 * Handles Dataset Switching, Drag & Drop, Side-by-Side Inference, and Metrics
 */

document.addEventListener('DOMContentLoaded', () => {
  let currentDataset = 'fashion'; // 'fashion' or 'cifar'
  let currentFile = null;
  let selectedSampleName = null;

  // DOM Elements
  const datasetButtons = document.querySelectorAll('.segment-btn');
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const dropContentIdle = document.getElementById('drop-content-idle');
  const dropPreviewContainer = document.getElementById('drop-preview-container');
  const imagePreview = document.getElementById('image-preview');
  const previewFilename = document.getElementById('preview-filename');
  const removeImageBtn = document.getElementById('remove-image-btn');
  const samplesChips = document.getElementById('samples-chips');
  const classifyBtn = document.getElementById('classify-btn');
  const btnText = classifyBtn.querySelector('.btn-text');
  const btnSpinner = classifyBtn.querySelector('.btn-spinner');

  const resultsIdle = document.getElementById('results-idle');
  const resultsActive = document.getElementById('results-active-container');
  const engineText = document.getElementById('engine-text');

  // Side-by-Side UI Elements
  const bannerStatus = document.getElementById('banner-status');
  const bannerIcon = document.getElementById('banner-icon');
  const bannerText = document.getElementById('banner-text');
  const gainValue = document.getElementById('gain-value');

  const baselineModelName = document.getElementById('baseline-model-name');
  const baselineClass = document.getElementById('baseline-class');
  const baselineConf = document.getElementById('baseline-conf');
  const baselineBars = document.getElementById('baseline-bars');

  const optimizedModelName = document.getElementById('optimized-model-name');
  const optimizedClass = document.getElementById('optimized-class');
  const optimizedConf = document.getElementById('optimized-conf');
  const optimizedBars = document.getElementById('optimized-bars');

  // Modal Elements
  const metricsModalBtn = document.getElementById('metrics-modal-btn');
  const metricsModal = document.getElementById('metrics-modal');
  const modalCloseBtn = document.getElementById('modal-close-btn');
  const modalContent = document.getElementById('modal-content');
  const modalTitle = document.getElementById('modal-title');

  // --------------------------------------------------------------------------
  // 1. DATASET SWITCHING
  // --------------------------------------------------------------------------
  datasetButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      datasetButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      currentDataset = btn.dataset.dataset;
      resetUpload();
      loadSamples();
      resetResults();
    });
  });

  // --------------------------------------------------------------------------
  // 2. LOAD SAMPLE CHIPS
  // --------------------------------------------------------------------------
  async function loadSamples() {
    samplesChips.innerHTML = '<span class="loading-chips">Loading samples...</span>';
    try {
      const res = await fetch(`/api/samples?dataset=${currentDataset}`);
      const data = await res.json();
      samplesChips.innerHTML = '';

      if (!data.samples || data.samples.length === 0) {
        samplesChips.innerHTML = '<span class="loading-chips">No test samples found. Train models first.</span>';
        return;
      }

      data.samples.forEach(sample => {
        const chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'sample-chip';
        chip.textContent = sample.name;
        chip.dataset.filename = sample.filename;

        chip.addEventListener('click', () => {
          document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));
          chip.classList.add('selected');
          selectSample(sample.filename, sample.name);
        });

        samplesChips.appendChild(chip);
      });
    } catch (err) {
      samplesChips.innerHTML = '<span class="loading-chips">Samples will appear after running train script.</span>';
    }
  }

  function selectSample(filename, displayName) {
    currentFile = null;
    selectedSampleName = filename;

    // Show preview directly from test folder
    const folder = currentDataset === 'fashion' ? 'FashionMNIST' : 'CIFAR10';
    // Load as local image
    imagePreview.src = `/${folder}/test_images/${filename}`;
    // Fallback if not directly routed
    imagePreview.onerror = () => {
      // In Flask, static route or standard data
      imagePreview.src = `/api/classify?sample_view=1`;
    };

    previewFilename.textContent = `Sample: ${displayName}`;
    dropContentIdle.style.display = 'none';
    dropPreviewContainer.style.display = 'flex';
    classifyBtn.disabled = false;
  }

  // --------------------------------------------------------------------------
  // 3. DRAG & DROP AND FILE PICKER
  // --------------------------------------------------------------------------
  dropZone.addEventListener('click', (e) => {
    if (e.target !== removeImageBtn && !e.target.closest('.remove-image-btn')) {
      fileInput.click();
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file (PNG, JPG, WEBP).');
      return;
    }

    currentFile = file;
    selectedSampleName = null;
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));

    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      previewFilename.textContent = file.name;
      dropContentIdle.style.display = 'none';
      dropPreviewContainer.style.display = 'flex';
      classifyBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  removeImageBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    resetUpload();
  });

  function resetUpload() {
    currentFile = null;
    selectedSampleName = null;
    fileInput.value = '';
    imagePreview.src = '';
    previewFilename.textContent = '';
    dropPreviewContainer.style.display = 'none';
    dropContentIdle.style.display = 'flex';
    classifyBtn.disabled = true;
    document.querySelectorAll('.sample-chip').forEach(c => c.classList.remove('selected'));
  }

  function resetResults() {
    resultsActive.style.display = 'none';
    resultsIdle.style.display = 'flex';
    engineText.textContent = 'Ready for inference';
  }

  // --------------------------------------------------------------------------
  // 4. CLASSIFICATION REQUEST & SIDE-BY-SIDE RENDERING
  // --------------------------------------------------------------------------
  classifyBtn.addEventListener('click', async () => {
    if (!currentFile && !selectedSampleName) return;

    classifyBtn.disabled = true;
    btnText.textContent = 'Inferencing Models...';
    btnSpinner.style.display = 'inline-block';
    engineText.textContent = 'Evaluating neural networks...';

    const formData = new FormData();
    formData.append('dataset', currentDataset);

    if (currentFile) {
      formData.append('image', currentFile);
    } else if (selectedSampleName) {
      formData.append('sample_name', selectedSampleName);
    }

    try {
      const res = await fetch('/api/classify', {
        method: 'POST',
        body: formData
      });

      const data = await res.json();
      if (!res.ok) {
        alert(`Classification error: ${data.error || 'Server error'}`);
        return;
      }

      renderSideBySideResults(data);
    } catch (err) {
      alert(`Network error: ${err.message}`);
    } finally {
      classifyBtn.disabled = false;
      btnText.textContent = 'Run Side-by-Side Classification';
      btnSpinner.style.display = 'none';
    }
  });

  function renderSideBySideResults(data) {
    resultsIdle.style.display = 'none';
    resultsActive.style.display = 'flex';

    if (data.source) {
      engineText.textContent = data.source;
    }

    const base = data.baseline;
    const opt = data.optimized;
    const comp = data.comparison;

    // Comparison Banner
    if (comp.classes_agree) {
      bannerIcon.textContent = '🎯';
      bannerText.textContent = `Both models agree on "${opt.predicted_class}"`;
      bannerStatus.style.color = 'var(--accent-emerald)';
    } else {
      bannerIcon.textContent = '⚡';
      bannerText.textContent = `Model Disagreement: Baseline said "${base.predicted_class}", Optimized said "${opt.predicted_class}"`;
      bannerStatus.style.color = 'var(--accent-amber)';
    }

    const gain = comp.confidence_gain;
    gainValue.textContent = (gain >= 0 ? `+${gain}%` : `${gain}%`);

    // Baseline Card
    baselineModelName.textContent = base.model;
    baselineClass.textContent = base.predicted_class;
    baselineConf.textContent = base.confidence;
    renderBars(baselineBars, base.top_5);

    // Optimized Card
    optimizedModelName.textContent = opt.model;
    optimizedClass.textContent = opt.predicted_class;
    optimizedConf.textContent = opt.confidence;
    renderBars(optimizedBars, opt.top_5);
  }

  function renderBars(container, top5Obj) {
    container.innerHTML = '';
    for (const [className, pctStr] of Object.entries(top5Obj)) {
      const numPct = parseFloat(pctStr);

      const item = document.createElement('div');
      item.className = 'prob-item';
      item.innerHTML = `
        <div class="prob-meta">
          <span class="prob-name">${className}</span>
          <span class="prob-pct">${pctStr}</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width: 0%"></div>
        </div>
      `;

      container.appendChild(item);

      // Animate bar width smoothly
      setTimeout(() => {
        const fill = item.querySelector('.prob-fill');
        if (fill) fill.style.width = `${Math.min(numPct, 100)}%`;
      }, 50);
    }
  }

  // --------------------------------------------------------------------------
  // 5. METRICS MODAL
  // --------------------------------------------------------------------------
  metricsModalBtn.addEventListener('click', async () => {
    metricsModal.style.display = 'flex';
    modalTitle.textContent = `${currentDataset === 'fashion' ? 'Fashion-MNIST' : 'CIFAR-10'} Performance Report`;
    modalContent.innerHTML = '<p class="modal-loading">Fetching training logs and hyperparameter table...</p>';

    try {
      const res = await fetch(`/api/metrics?dataset=${currentDataset}`);
      const data = await res.json();

      if (!res.ok) {
        modalContent.innerHTML = `<p style="color: var(--accent-amber);">${data.error}</p>`;
        return;
      }

      const b = data.baseline;
      const o = data.optimized;

      modalContent.innerHTML = `
        <table class="metrics-table">
          <thead>
            <tr>
              <th>Metric / Parameter</th>
              <th>Baseline Model (Quickstart)</th>
              <th>Optimized Model (Tuned CNN)</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Architecture</strong></td>
              <td>${b.architecture}</td>
              <td>${o.architecture}</td>
            </tr>
            <tr>
              <td><strong>Optimizer</strong></td>
              <td>${b.optimizer} (lr=${b.learning_rate})</td>
              <td>${o.optimizer} (lr=${o.learning_rate})</td>
            </tr>
            <tr>
              <td><strong>Epochs / Batch Size</strong></td>
              <td>${b.epochs} epochs / ${b.batch_size} batch</td>
              <td>${o.epochs} epochs / ${o.batch_size} batch</td>
            </tr>
            <tr>
              <td><strong>Final Test Loss</strong></td>
              <td>${b.final_test_loss}</td>
              <td class="gain-cell">${o.final_test_loss} (Lower is better)</td>
            </tr>
            <tr>
              <td><strong>Final Test Accuracy</strong></td>
              <td>${b.final_test_accuracy}%</td>
              <td class="gain-cell">${o.final_test_accuracy}% (+${data.accuracy_gain_percent}%)</td>
            </tr>
            <tr>
              <td><strong>Training Duration</strong></td>
              <td>${b.training_time_seconds}s</td>
              <td>${o.training_time_seconds}s</td>
            </tr>
          </tbody>
        </table>
        <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 10px;">
          💡 <em>Hyperparameter Optimization Impact:</em> Transitioning from a flat MLP with vanilla SGD to a Deep CNN with 2D Convolutions, Batch Normalization, and the Adam optimizer yielded a <strong>+${data.accuracy_gain_percent}%</strong> test accuracy gain.
        </p>
      `;
    } catch (err) {
      modalContent.innerHTML = `<p style="color: var(--accent-rose);">Failed to load metrics: ${err.message}</p>`;
    }
  });

  modalCloseBtn.addEventListener('click', () => {
    metricsModal.style.display = 'none';
  });

  metricsModal.addEventListener('click', (e) => {
    if (e.target === metricsModal) metricsModal.style.display = 'none';
  });

  // Initial load
  loadSamples();
});
