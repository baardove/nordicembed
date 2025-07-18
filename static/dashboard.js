// Dashboard JavaScript for Norwegian Embeddings Service

// API Configuration
const API_BASE = window.location.origin;

// State
let serviceStatus = {
    healthy: false,
    modelLoaded: false,
    metrics: {
        totalRequests: 0,
        avgResponseTime: 0
    }
};

// Update intervals
let statusInterval;
let metricsInterval;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', () => {
    initializeDashboard();
    setupEventListeners();
    startStatusUpdates();
});

function initializeDashboard() {
    // Set API endpoint display
    document.getElementById('api-endpoint').textContent = `${API_BASE}/api/embed`;
    
    // Initial status check
    checkServiceStatus();
    
    // Load initial data
    loadServiceInfo();
}

function setupEventListeners() {
    // Test form submission
    document.getElementById('test-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await testEmbeddings();
    });
    
    // Rerank form submission
    const rerankForm = document.getElementById('rerank-form');
    if (rerankForm) {
        rerankForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await testReranking();
        });
    }
    
    // Device save button
    document.getElementById('save-device-btn').addEventListener('click', async () => {
        await saveDeviceSetting();
    });
}

function startStatusUpdates() {
    // Update status every 5 seconds
    statusInterval = setInterval(checkServiceStatus, 5000);
    
    // Update metrics every 10 seconds
    metricsInterval = setInterval(loadMetrics, 10000);
}

async function checkServiceStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/health`);
        const data = await response.json();
        
        updateHealthStatus(data);
        updateLastUpdate();
    } catch (error) {
        updateHealthStatus({ status: 'unhealthy', model_loaded: false });
        console.error('Health check failed:', error);
    }
}

function updateHealthStatus(data) {
    const indicator = document.getElementById('health-indicator');
    const statusText = document.getElementById('health-status');
    
    if (data.status === 'healthy' && data.model_loaded) {
        indicator.className = 'status-indicator status-healthy';
        statusText.textContent = 'Healthy';
        statusText.className = 'success';
        serviceStatus.healthy = true;
        serviceStatus.modelLoaded = true;
    } else if (data.status === 'healthy' && !data.model_loaded) {
        indicator.className = 'status-indicator status-loading';
        statusText.textContent = 'Loading Model...';
        statusText.className = '';
        serviceStatus.healthy = true;
        serviceStatus.modelLoaded = false;
    } else {
        indicator.className = 'status-indicator status-unhealthy';
        statusText.textContent = 'Unhealthy';
        statusText.className = 'error';
        serviceStatus.healthy = false;
        serviceStatus.modelLoaded = false;
    }
}

function updateLastUpdate() {
    const now = new Date();
    document.getElementById('last-update').textContent = now.toLocaleTimeString();
}

async function loadServiceInfo() {
    try {
        // Get service info
        const response = await fetch(`${API_BASE}/api/info`);
        if (response.ok) {
            const data = await response.json();
            
            // Update device info with null checks
            const deviceEl = document.getElementById('device-type');
            if (deviceEl) deviceEl.textContent = data.device || 'Unknown';
            
            // Set current device in select dropdown
            const deviceSelect = document.getElementById('device-select');
            if (deviceSelect && data.device) {
                deviceSelect.value = data.device.toLowerCase();
            }
            
            // Update configuration
            const batchEl = document.getElementById('batch-size');
            if (batchEl) batchEl.textContent = data.max_batch_size || '32';
            
            const lengthEl = document.getElementById('max-length');
            if (lengthEl) lengthEl.textContent = data.max_length || '512';
            
            // Update available models
            if (data.available_models) {
                const modelsEl = document.getElementById('available-models');
                if (modelsEl) modelsEl.textContent = data.available_models.join(', ');
            }
            
            // Update available rerankers
            if (data.available_rerankers) {
                const rerankersEl = document.getElementById('available-rerankers');
                if (rerankersEl) rerankersEl.textContent = data.available_rerankers.join(', ');
            }
        } else {
            console.error('Service info request failed:', response.status);
            setErrorState();
        }
    } catch (error) {
        console.error('Failed to load service info:', error);
        setErrorState();
    }
}

function setErrorState() {
    const deviceEl = document.getElementById('device-type');
    if (deviceEl) deviceEl.textContent = 'Error loading';
}

async function loadMetrics() {
    try {
        const response = await fetch(`${API_BASE}/api/metrics`);
        if (response.ok) {
            const data = await response.json();
            
            // Update metrics display
            document.getElementById('requests-count').textContent = data.total_requests || '0';
            document.getElementById('avg-response-time').textContent = 
                data.avg_response_time ? `${data.avg_response_time.toFixed(1)}ms` : '0ms';
            
            // Update last request info
            if (data.last_request && data.last_request.timestamp) {
                const lastReq = data.last_request;
                const timestamp = new Date(lastReq.timestamp);
                const timeAgo = getTimeAgo(timestamp);
                
                let lastRequestHtml = `
                    <div class="info-item">
                        <span class="info-label">Last Request:</span>
                        <span class="info-value">${timeAgo}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Endpoint:</span>
                        <span class="info-value">${lastReq.endpoint}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Texts Processed:</span>
                        <span class="info-value">${lastReq.text_count}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Response Time:</span>
                        <span class="info-value">${(lastReq.response_time * 1000).toFixed(1)}ms</span>
                    </div>`;
                
                if (lastReq.source_ip) {
                    lastRequestHtml += `
                    <div class="info-item">
                        <span class="info-label">Source IP:</span>
                        <span class="info-value">${lastReq.source_ip}</span>
                    </div>`;
                }
                
                // Add to page if element exists
                const lastRequestEl = document.getElementById('last-request-info');
                if (lastRequestEl) {
                    lastRequestEl.innerHTML = lastRequestHtml;
                }
            }
            
            // Update additional metrics
            if (data.uptime_seconds) {
                const uptimeEl = document.getElementById('uptime');
                if (uptimeEl) {
                    uptimeEl.textContent = formatUptime(data.uptime_seconds);
                }
            }
            
            if (data.requests_per_minute !== undefined) {
                const rpmEl = document.getElementById('requests-per-minute');
                if (rpmEl) {
                    rpmEl.textContent = data.requests_per_minute.toFixed(2);
                }
            }
            
            // Update utilization indicator
            if (data.utilization_percentage !== undefined) {
                const utilization = data.utilization_percentage;
                const utilizationText = document.getElementById('utilization-text');
                const utilizationBar = document.getElementById('utilization-bar');
                const utilizationBarText = document.getElementById('utilization-bar-text');
                const utilizationDetails = document.getElementById('utilization-details');
                
                if (utilizationText) {
                    utilizationText.textContent = `${utilization}%`;
                }
                
                if (utilizationBar) {
                    // Set width
                    utilizationBar.style.width = `${Math.max(utilization, 5)}%`; // Min 5% for visibility
                    
                    // Set color class based on utilization
                    utilizationBar.classList.remove('utilization-low', 'utilization-medium', 'utilization-high');
                    if (utilization <= 50) {
                        utilizationBar.classList.add('utilization-low');
                    } else if (utilization <= 80) {
                        utilizationBar.classList.add('utilization-medium');
                    } else {
                        utilizationBar.classList.add('utilization-high');
                    }
                    
                    // Set bar text (only show if bar is wide enough)
                    if (utilizationBarText) {
                        utilizationBarText.textContent = utilization > 10 ? `${utilization}%` : '';
                    }
                }
                
                if (utilizationDetails && data.busy_seconds_10min !== undefined) {
                    const busyTime = data.busy_seconds_10min;
                    utilizationDetails.textContent = `Busy time: ${busyTime.toFixed(1)}s / 600s`;
                }
            }
            
            // Update model statistics
            if (data.model_requests !== undefined) {
                // Update individual model counters
                const models = [
                    // Norwegian models
                    'norbert2', 'nb-bert-base', 'nb-bert-large',
                    'simcse-nb-bert-large', 'norbert3-base', 'norbert3-large',
                    'xlm-roberta-base', 'electra-small-nordic', 'sentence-bert-base',
                    // Swedish models
                    'kb-sbert-swedish', 'kb-bert-swedish', 'bert-large-swedish', 'albert-swedish',
                    // Danish models
                    'dabert', 'aelaectra-danish', 'da-bert-ner', 'electra-base-danish',
                    // Multilingual Nordic
                    'multilingual-e5-base', 'paraphrase-multilingual-minilm',
                    // Finnish models
                    'finbert-base', 'finbert-sbert', 'finbert-large',
                    // Icelandic model
                    'icebert'
                ];
                
                models.forEach(model => {
                    const count = data.model_requests[model] || 0;
                    
                    // Update model stat grid counter
                    const countEl = document.getElementById(`count-${model}`);
                    if (countEl) {
                        countEl.textContent = count.toString();
                    }
                    
                    // Update table counter
                    const tableCountEl = document.getElementById(`table-count-${model}`);
                    if (tableCountEl) {
                        tableCountEl.textContent = count.toString();
                    }
                });
            }
            
            serviceStatus.metrics = data;
        }
    } catch (error) {
        console.error('Failed to load metrics:', error);
    }
}

function getTimeAgo(date) {
    const seconds = Math.floor((new Date() - date) / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

function formatUptime(seconds) {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
}

async function testEmbeddings() {
    const text = document.getElementById('test-text').value.trim();
    const model = document.getElementById('test-model').value;
    const poolingStrategy = document.getElementById('pooling-strategy').value;
    const button = document.getElementById('test-button');
    const loading = document.getElementById('test-loading');
    const resultBox = document.getElementById('test-result');
    
    if (!text) {
        showResult('Please enter some text to embed.', 'error');
        return;
    }
    
    // UI state
    button.disabled = true;
    loading.style.display = 'inline-block';
    resultBox.style.display = 'none';
    
    try {
        const startTime = performance.now();
        
        const response = await fetch(`${API_BASE}/api/test-embed`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts: [text],
                model: model,
                pooling_strategy: poolingStrategy
            })
        });
        
        const endTime = performance.now();
        const responseTime = Math.round(endTime - startTime);
        
        if (response.ok) {
            const data = await response.json();
            
            // Format result
            const embedding = data.embeddings[0];
            const dimensions = embedding.length;
            const preview = embedding.slice(0, 10).map(v => v.toFixed(6)).join(', ');
            
            const result = `Success! Generated ${dimensions}-dimensional embedding in ${responseTime}ms\n\n` +
                          `Model: ${model}\n` +
                          `Pooling Strategy: ${poolingStrategy}\n` +
                          `Text Length: ${text.length} characters\n\n` +
                          `Embedding Preview (first 10 values):\n[${preview}, ...]`;
            
            showResult(result, 'success');
        } else {
            const error = await response.json();
            showResult(`Error: ${error.detail || 'Failed to generate embeddings'}`, 'error');
        }
    } catch (error) {
        showResult(`Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        loading.style.display = 'none';
    }
}

function showResult(message, type = 'info') {
    const resultBox = document.getElementById('test-result');
    resultBox.textContent = message;
    resultBox.className = `result-box ${type}`;
    resultBox.style.display = 'block';
}

async function saveDeviceSetting() {
    const deviceSelect = document.getElementById('device-select');
    const saveButton = document.getElementById('save-device-btn');
    const resultBox = document.getElementById('device-save-result');
    
    const newDevice = deviceSelect.value;
    
    // Disable button during save
    saveButton.disabled = true;
    resultBox.style.display = 'none';
    
    try {
        const response = await fetch(`${API_BASE}/api/update-device`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                device: newDevice
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            resultBox.textContent = `Success! Device setting updated to ${newDevice.toUpperCase()}. Please restart the container for changes to take effect.`;
            resultBox.className = 'result-box success';
            resultBox.style.display = 'block';
            
            // Update the display to reflect the new setting
            const deviceEl = document.getElementById('device-type');
            if (deviceEl) {
                deviceEl.textContent = `${newDevice.toUpperCase()} (restart required)`;
                deviceEl.style.color = '#e67e22'; // Orange color to indicate restart needed
            }
        } else {
            const error = await response.json();
            resultBox.textContent = `Error: ${error.detail || 'Failed to update device setting'}`;
            resultBox.className = 'result-box error';
            resultBox.style.display = 'block';
        }
    } catch (error) {
        resultBox.textContent = `Error: ${error.message}`;
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
    } finally {
        saveButton.disabled = false;
    }
}

async function testReranking() {
    const query = document.getElementById('rerank-query').value.trim();
    const documentsText = document.getElementById('rerank-documents').value.trim();
    const model = document.getElementById('rerank-model').value;
    const topK = parseInt(document.getElementById('rerank-topk').value) || null;
    const button = document.getElementById('rerank-button');
    const loading = document.getElementById('rerank-loading');
    const resultBox = document.getElementById('rerank-result');
    
    if (!query) {
        showRerankResult('Please enter a search query.', 'error');
        return;
    }
    
    if (!documentsText) {
        showRerankResult('Please enter documents to rerank.', 'error');
        return;
    }
    
    // Split documents by newline and filter empty lines
    const documents = documentsText.split('\n').filter(doc => doc.trim());
    
    if (documents.length === 0) {
        showRerankResult('Please enter at least one document.', 'error');
        return;
    }
    
    // UI state
    button.disabled = true;
    loading.style.display = 'inline-block';
    resultBox.style.display = 'none';
    
    try {
        const startTime = performance.now();
        
        const response = await fetch(`${API_BASE}/api/rerank`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                documents: documents,
                model: model,
                top_k: topK
            })
        });
        
        const endTime = performance.now();
        const responseTime = Math.round(endTime - startTime);
        
        if (response.ok) {
            const data = await response.json();
            
            // Format results
            let resultHtml = `<strong>Reranking Results</strong> (${responseTime}ms)<br><br>`;
            resultHtml += `<strong>Query:</strong> "${query}"<br><br>`;
            resultHtml += `<strong>Top ${data.results.length} Results:</strong><br><br>`;
            
            data.results.forEach((result, idx) => {
                resultHtml += `<div style="margin-bottom: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 4px;">`;
                resultHtml += `<strong>${idx + 1}. Score: ${result.score.toFixed(4)}</strong><br>`;
                resultHtml += `<span style="color: #666;">Original index: ${result.index}</span><br>`;
                resultHtml += `<div style="margin-top: 5px;">${escapeHtml(result.document)}</div>`;
                resultHtml += `</div>`;
            });
            
            resultHtml += `<br><strong>Model:</strong> ${data.model}<br>`;
            resultHtml += `<strong>Documents processed:</strong> ${data.documents_count}`;
            
            showRerankResult(resultHtml, 'success');
        } else {
            const error = await response.json();
            showRerankResult(`Error: ${error.detail || 'Failed to rerank documents'}`, 'error');
        }
    } catch (error) {
        showRerankResult(`Error: ${error.message}`, 'error');
    } finally {
        button.disabled = false;
        loading.style.display = 'none';
    }
}

function showRerankResult(message, type = 'info') {
    const resultBox = document.getElementById('rerank-result');
    resultBox.innerHTML = message;
    resultBox.className = `result-box ${type}`;
    resultBox.style.display = 'block';
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (statusInterval) clearInterval(statusInterval);
    if (metricsInterval) clearInterval(metricsInterval);
});