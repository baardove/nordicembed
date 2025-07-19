// Dashboard JavaScript for Nordic Embeddings Service

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
    setupEndpointUrls();
});

function setupEndpointUrls() {
    // Set up full URLs for endpoints
    const baseUrl = window.location.origin;
    
    // Update the RAGFlow base URL (without /v1/)
    const ragflowBaseUrl = document.getElementById('ragflow-base-url');
    if (ragflowBaseUrl) {
        ragflowBaseUrl.textContent = `${baseUrl}/`;
    }
    
    // Update the main OpenAI base URL at the top
    const openAIBaseUrl = document.getElementById('openai-base-url');
    if (openAIBaseUrl) {
        openAIBaseUrl.textContent = `${baseUrl}/v1/`;
    }
    
    // Update embedding endpoints
    const embedOpenAIEndpoint = document.getElementById('embed-openai-endpoint');
    if (embedOpenAIEndpoint) {
        embedOpenAIEndpoint.textContent = `${baseUrl}/v1/embeddings`;
    }
    
    const embedBasicEndpoint = document.getElementById('embed-basic-endpoint');
    if (embedBasicEndpoint) {
        embedBasicEndpoint.textContent = `${baseUrl}/api/embed`;
    }
    
    // Update reranking endpoints
    const rerankOpenAIEndpoint = document.getElementById('rerank-openai-endpoint');
    if (rerankOpenAIEndpoint) {
        rerankOpenAIEndpoint.textContent = `${baseUrl}/v1/rerank`;
    }
    
    const rerankBasicEndpoints = document.getElementById('rerank-basic-endpoints');
    if (rerankBasicEndpoints) {
        rerankBasicEndpoints.textContent = `${baseUrl}/api/rerank, ${baseUrl}/api/score-pairs`;
    }
}

function initializeDashboard() {
    // Set API endpoint display (if element exists)
    const apiEndpointEl = document.getElementById('api-endpoint');
    if (apiEndpointEl) {
        apiEndpointEl.textContent = `${API_BASE}/api/embed`;
    }
    
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
    
    // Restart container button
    document.getElementById('restart-container-btn').addEventListener('click', async () => {
        await restartContainer();
    });
    
    // Trust remote code toggle
    const trustToggle = document.getElementById('trust-remote-code-toggle');
    if (trustToggle) {
        trustToggle.addEventListener('change', () => {
            // Enable save button when changed
            document.getElementById('save-trust-btn').disabled = false;
        });
    }
    
    // Save trust remote code button
    document.getElementById('save-trust-btn').addEventListener('click', async () => {
        await saveTrustRemoteCode();
    });
    
    // Restart container button for trust settings
    const restartTrustBtn = document.getElementById('restart-container-trust-btn');
    if (restartTrustBtn) {
        restartTrustBtn.addEventListener('click', async () => {
            await restartContainer();
        });
    }
    
    // View logs button
    const viewLogsBtn = document.getElementById('view-logs-btn');
    if (viewLogsBtn) {
        viewLogsBtn.addEventListener('click', () => {
            openLogModal();
        });
    }
    
    // Authentication event listeners
    setupAuthenticationListeners();
}

function setupAuthenticationListeners() {
    // Dashboard auth toggle
    const dashboardAuthToggle = document.getElementById('dashboard-auth-toggle');
    if (dashboardAuthToggle) {
        dashboardAuthToggle.addEventListener('change', (e) => {
            const passwordSection = document.getElementById('dashboard-password-section');
            passwordSection.style.display = e.target.checked ? 'block' : 'none';
            document.getElementById('save-dashboard-auth-btn').disabled = false;
        });
    }
    
    // Password inputs
    const dashboardPassword = document.getElementById('dashboard-password');
    const dashboardPasswordConfirm = document.getElementById('dashboard-password-confirm');
    if (dashboardPassword && dashboardPasswordConfirm) {
        [dashboardPassword, dashboardPasswordConfirm].forEach(input => {
            input.addEventListener('input', () => {
                document.getElementById('save-dashboard-auth-btn').disabled = false;
            });
        });
    }
    
    // Save dashboard auth button
    document.getElementById('save-dashboard-auth-btn').addEventListener('click', async () => {
        await saveDashboardAuth();
    });
    
    // API auth toggle
    const apiAuthToggle = document.getElementById('api-auth-toggle');
    if (apiAuthToggle) {
        apiAuthToggle.addEventListener('change', (e) => {
            const warning = document.getElementById('api-auth-warning');
            warning.style.display = e.target.checked ? 'block' : 'none';
            document.getElementById('save-api-auth-btn').disabled = false;
        });
    }
    
    // Save API auth button
    document.getElementById('save-api-auth-btn').addEventListener('click', async () => {
        await saveApiAuth();
    });
    
    // Create API key button
    document.getElementById('create-api-key-btn').addEventListener('click', async () => {
        await createApiKey();
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
    const lastUpdateEl = document.getElementById('last-update');
    if (lastUpdateEl) {
        lastUpdateEl.textContent = now.toLocaleTimeString();
    }
}

async function loadServiceInfo() {
    try {
        // Prepare headers with internal API key if available
        const headers = {};
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            console.log('Internal key response:', keyData);
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        } else {
            console.error('Failed to get internal key:', keyResponse.status);
        }
        
        // Get service info
        const response = await fetch(`${API_BASE}/api/info`, {
            headers: headers
        });
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
            
            // Update available models with copy buttons
            if (data.available_models && data.available_models.length > 0) {
                const modelsEl = document.getElementById('available-models');
                if (modelsEl) {
                    let modelsHtml = '';
                    data.available_models.forEach((model, index) => {
                        const modelId = `model-${index}`;
                        modelsHtml += `<span style="display: inline-flex; align-items: center; margin-right: 10px; margin-bottom: 5px;">
                            <code id="${modelId}" style="background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 13px;">${model}</code>
                            <button onclick="copyToClipboard('${modelId}', event)" class="btn-small btn-copy" style="padding: 2px 6px; font-size: 11px; margin-left: 5px;">Copy</button>
                        </span>`;
                    });
                    modelsEl.innerHTML = modelsHtml;
                }
            }
            
            // Update available rerankers with copy buttons
            if (data.available_rerankers && data.available_rerankers.length > 0) {
                const rerankersEl = document.getElementById('available-rerankers');
                if (rerankersEl) {
                    let rerankersHtml = '';
                    data.available_rerankers.forEach((model, index) => {
                        const modelId = `reranker-${index}`;
                        rerankersHtml += `<span style="display: inline-flex; align-items: center; margin-right: 10px; margin-bottom: 5px;">
                            <code id="${modelId}" style="background: #e9ecef; padding: 4px 8px; border-radius: 4px; font-size: 13px;">${model}</code>
                            <button onclick="copyToClipboard('${modelId}', event)" class="btn-small btn-copy" style="padding: 2px 6px; font-size: 11px; margin-left: 5px;">Copy</button>
                        </span>`;
                    });
                    rerankersEl.innerHTML = rerankersHtml;
                }
            }
            
            // Update trust remote code status
            if (data.allow_trust_remote_code !== undefined) {
                const trustToggle = document.getElementById('trust-remote-code-toggle');
                const trustStatus = document.getElementById('trust-remote-code-status');
                
                if (trustToggle) {
                    trustToggle.checked = data.allow_trust_remote_code;
                }
                
                if (trustStatus) {
                    trustStatus.textContent = data.allow_trust_remote_code ? 'Enabled' : 'Disabled';
                    trustStatus.style.color = data.allow_trust_remote_code ? '#27ae60' : '#e74c3c';
                }
            }
        } else {
            console.error('Service info request failed:', response.status);
            setErrorState();
        }
    } catch (error) {
        console.error('Error loading service info:', error);
        setErrorState();
    }
    
    // Load authentication status
    await loadAuthStatus();
    await loadApiKeys();
}

function setErrorState() {
    const deviceEl = document.getElementById('device-type');
    if (deviceEl) deviceEl.textContent = 'Error loading';
}

async function loadMetrics() {
    try {
        // Prepare headers with internal API key if available
        const headers = {};
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        }
        
        const response = await fetch(`${API_BASE}/api/metrics`, {
            headers: headers
        });
        if (response.ok) {
            const data = await response.json();
            
            // Update metrics display
            const requestsCountEl = document.getElementById('requests-count');
            if (requestsCountEl) {
                requestsCountEl.textContent = data.total_requests || '0';
            }
            
            const avgResponseEl = document.getElementById('avg-response-time');
            if (avgResponseEl) {
                avgResponseEl.textContent = data.avg_response_time ? `${data.avg_response_time.toFixed(1)}ms` : '0ms';
            }
            
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
                    
                    // Update response time
                    if (data.model_response_times && data.model_response_times[model]) {
                        const responseTime = data.model_response_times[model].avg;
                        const responseEl = document.getElementById(`table-response-${model}`);
                        if (responseEl) {
                            responseEl.textContent = responseTime ? `${responseTime}ms` : '-';
                        }
                    }
                });
                
                // Update reranker statistics
                const rerankers = [
                    'mmarco-minilm-l12', 'ms-marco-minilm-l6', 'ms-marco-minilm-l12',
                    'jina-reranker-multilingual', 'nordic-reranker'
                ];
                
                rerankers.forEach(model => {
                    const count = data.model_requests[model] || 0;
                    
                    // Update reranker table counter
                    const rerankCountEl = document.getElementById(`rerank-count-${model}`);
                    if (rerankCountEl) {
                        rerankCountEl.textContent = count.toString();
                    }
                    
                    // Update reranker response time
                    if (data.model_response_times && data.model_response_times[model]) {
                        const responseTime = data.model_response_times[model].avg;
                        const rerankResponseEl = document.getElementById(`rerank-response-${model}`);
                        if (rerankResponseEl) {
                            rerankResponseEl.textContent = responseTime ? `${responseTime}ms` : '-';
                        }
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
    const apiKey = document.getElementById('test-api-key').value.trim();
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
        
        // Build headers
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Add API key if provided
        if (apiKey) {
            headers['X-API-Key'] = apiKey;
        }
        
        const response = await fetch(`${API_BASE}/api/test-embed`, {
            method: 'POST',
            headers: headers,
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
        // Get internal API key if needed
        const headers = {
            'Content-Type': 'application/json'
        };
        
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        }
        
        const response = await fetch(`${API_BASE}/api/update-device`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({
                device: newDevice
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            resultBox.textContent = `Success! Device setting updated to ${newDevice.toUpperCase()}. Configuration saved to ${data.config_location}. Please restart the container for changes to take effect.`;
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
    const apiKey = document.getElementById('rerank-api-key').value.trim();
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
        
        // Build headers
        const headers = {
            'Content-Type': 'application/json'
        };
        
        // Add API key if provided
        if (apiKey) {
            headers['X-API-Key'] = apiKey;
        }
        
        const response = await fetch(`${API_BASE}/api/rerank`, {
            method: 'POST',
            headers: headers,
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

async function saveTrustRemoteCode() {
    const trustToggle = document.getElementById('trust-remote-code-toggle');
    const resultBox = document.getElementById('trust-save-result');
    const restartBtn = document.getElementById('restart-container-trust-btn');
    
    const enabled = trustToggle.checked;
    
    try {
        // Get internal API key if needed
        const headers = {
            'Content-Type': 'application/json'
        };
        
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        }
        
        const response = await fetch(`${API_BASE}/api/config/trust-remote-code`, {
            method: 'POST',
            headers: headers,
            body: JSON.stringify({ enabled: enabled })
        });
        
        if (response.ok) {
            const data = await response.json();
            resultBox.textContent = `Success! Trust remote code ${enabled ? 'enabled' : 'disabled'}. Configuration saved to ${data.config_location}.`;
            resultBox.className = 'result-box success';
            resultBox.style.display = 'block';
            
            // Show restart button
            restartBtn.style.display = 'block';
            
            // Update status display
            const trustStatus = document.getElementById('trust-remote-code-status');
            if (trustStatus) {
                trustStatus.textContent = enabled ? 'Enabled (restart required)' : 'Disabled (restart required)';
                trustStatus.style.color = '#e67e22'; // Orange to indicate restart needed
            }
            
            // Disable save button until next change
            document.getElementById('save-trust-btn').disabled = true;
        } else {
            const error = await response.json();
            resultBox.textContent = `Error: ${error.detail || 'Failed to update trust remote code setting'}`;
            resultBox.className = 'result-box error';
            resultBox.style.display = 'block';
        }
    } catch (error) {
        resultBox.textContent = `Error: ${error.message}`;
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
    }
}

async function restartContainer() {
    if (!confirm('Are you sure you want to restart the container? The service will be temporarily unavailable.')) {
        return;
    }
    
    // Find all result boxes to show status
    const resultBoxes = document.querySelectorAll('.result-box');
    resultBoxes.forEach(box => {
        box.textContent = 'Restarting container...';
        box.className = 'result-box';
        box.style.display = 'block';
    });
    
    try {
        // First, get the internal API key if API auth is enabled
        const headers = {};
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        }
        
        const response = await fetch(`${API_BASE}/api/restart`, {
            method: 'POST',
            headers: headers
        });
        
        if (response.ok) {
            resultBoxes.forEach(box => {
                box.textContent = 'Container restart initiated. The service will be back online in a few seconds...';
                box.className = 'result-box success';
            });
            
            // Wait a bit then start checking if service is back
            setTimeout(() => {
                checkServiceRestart();
            }, 5000);
        } else {
            const error = await response.json();
            resultBoxes.forEach(box => {
                box.textContent = `Error: ${error.detail || 'Failed to restart container'}`;
                box.className = 'result-box error';
            });
        }
    } catch (error) {
        resultBoxes.forEach(box => {
            box.textContent = `Error: ${error.message}`;
            box.className = 'result-box error';
        });
    }
}

function checkServiceRestart() {
    const checkInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/api/health`);
            if (response.ok) {
                // Service is back online
                clearInterval(checkInterval);
                window.location.reload();
            }
        } catch (error) {
            // Still offline, keep checking
        }
    }, 2000);
    
    // Stop checking after 60 seconds
    setTimeout(() => {
        clearInterval(checkInterval);
    }, 60000);
}

// Clean up on page unload
window.addEventListener('beforeunload', () => {
    if (statusInterval) clearInterval(statusInterval);
    if (metricsInterval) clearInterval(metricsInterval);
});

// Authentication functions
async function loadAuthStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/auth/status`);
        if (response.ok) {
            const data = await response.json();
            
            // Update dashboard auth status
            const dashboardAuthToggle = document.getElementById('dashboard-auth-toggle');
            const dashboardAuthStatus = document.getElementById('dashboard-auth-status');
            if (dashboardAuthToggle) {
                dashboardAuthToggle.checked = data.dashboard_auth_enabled;
            }
            if (dashboardAuthStatus) {
                dashboardAuthStatus.textContent = data.dashboard_auth_enabled ? 'Enabled' : 'Disabled';
                dashboardAuthStatus.style.color = data.dashboard_auth_enabled ? '#27ae60' : '#e74c3c';
            }
            
            // Show/hide logout button based on dashboard auth status
            const logoutBtn = document.getElementById('logout-btn');
            if (logoutBtn) {
                logoutBtn.style.display = data.dashboard_auth_enabled ? 'block' : 'none';
            }
            
            // Update API auth status
            const apiAuthToggle = document.getElementById('api-auth-toggle');
            const apiAuthStatus = document.getElementById('api-auth-status');
            if (apiAuthToggle) {
                apiAuthToggle.checked = data.api_auth_enabled;
            }
            if (apiAuthStatus) {
                apiAuthStatus.textContent = data.api_auth_enabled ? 'Enabled' : 'Disabled';
                apiAuthStatus.style.color = data.api_auth_enabled ? '#27ae60' : '#e74c3c';
            }
            
            // Show warning if API auth is enabled but no keys exist
            if (data.api_auth_enabled && data.api_keys_count === 0) {
                document.getElementById('api-auth-warning').style.display = 'block';
            }
        }
    } catch (error) {
        console.error('Failed to load auth status:', error);
    }
}

async function saveDashboardAuth() {
    const toggle = document.getElementById('dashboard-auth-toggle');
    const password = document.getElementById('dashboard-password').value;
    const confirmPassword = document.getElementById('dashboard-password-confirm').value;
    const resultBox = document.getElementById('dashboard-auth-result');
    
    // Validate passwords if enabling auth
    if (toggle.checked && password !== confirmPassword) {
        resultBox.textContent = 'Error: Passwords do not match';
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
        return;
    }
    
    if (toggle.checked && password.length < 6) {
        resultBox.textContent = 'Error: Password must be at least 6 characters';
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/auth/dashboard`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                enabled: toggle.checked,
                password: toggle.checked ? password : null
            })
        });
        
        if (response.ok) {
            resultBox.textContent = `Dashboard authentication ${toggle.checked ? 'enabled' : 'disabled'} successfully`;
            resultBox.className = 'result-box success';
            resultBox.style.display = 'block';
            
            // Clear password fields
            document.getElementById('dashboard-password').value = '';
            document.getElementById('dashboard-password-confirm').value = '';
            
            // Update status
            await loadAuthStatus();
        } else {
            const error = await response.json();
            resultBox.textContent = `Error: ${error.detail || 'Failed to update dashboard authentication'}`;
            resultBox.className = 'result-box error';
            resultBox.style.display = 'block';
        }
    } catch (error) {
        resultBox.textContent = `Error: ${error.message}`;
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
    }
}

async function saveApiAuth() {
    const toggle = document.getElementById('api-auth-toggle');
    const resultBox = document.getElementById('api-auth-result');
    
    try {
        const response = await fetch(`${API_BASE}/api/auth/api`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                enabled: toggle.checked
            })
        });
        
        if (response.ok) {
            resultBox.textContent = `API authentication ${toggle.checked ? 'enabled' : 'disabled'} successfully`;
            resultBox.className = 'result-box success';
            resultBox.style.display = 'block';
            
            // Update status
            await loadAuthStatus();
        } else {
            const error = await response.json();
            resultBox.textContent = `Error: ${error.detail || 'Failed to update API authentication'}`;
            resultBox.className = 'result-box error';
            resultBox.style.display = 'block';
        }
    } catch (error) {
        resultBox.textContent = `Error: ${error.message}`;
        resultBox.className = 'result-box error';
        resultBox.style.display = 'block';
    }
}

async function createApiKey() {
    const nameInput = document.getElementById('new-api-key-name');
    const name = nameInput.value.trim();
    
    if (!name) {
        alert('Please enter a name for the API key');
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/auth/keys`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ name: name })
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Show the new API key with copy button
            const keyId = `new-api-key-${Date.now()}`;
            const message = `
                <div style="margin-bottom: 10px; font-size: 16px; font-weight: 600;">✅ API Key created successfully!</div>
                <div style="background: #f8f9fa; padding: 15px; border-radius: 4px; margin-bottom: 10px; display: flex; align-items: center; gap: 10px;">
                    <code id="${keyId}" style="font-family: monospace; font-size: 14px; background: white; padding: 8px 12px; border: 1px solid #dee2e6; border-radius: 4px; flex: 1; word-break: break-all;">${data.api_key}</code>
                    <button onclick="copyToClipboard('${keyId}', event)" class="btn" style="padding: 8px 16px; background-color: #3498db; white-space: nowrap;">📋 Copy Key</button>
                </div>
                <div style="color: #e74c3c; font-weight: bold; font-size: 14px;">⚠️ Save this key now! It won't be shown again.</div>
            `;
            
            // Show in a temporary alert div
            const alertDiv = document.createElement('div');
            alertDiv.className = 'result-box success';
            alertDiv.innerHTML = message;
            alertDiv.style.marginTop = '15px';
            
            const container = document.getElementById('api-keys-list');
            container.insertBefore(alertDiv, container.firstChild);
            
            // Clear input
            nameInput.value = '';
            
            // Reload API keys list
            await loadApiKeys();
            
            // Remove alert after 30 seconds
            setTimeout(() => {
                alertDiv.remove();
            }, 30000);
        } else {
            const error = await response.json();
            alert(`Error: ${error.detail || 'Failed to create API key'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

async function loadApiKeys() {
    try {
        const response = await fetch(`${API_BASE}/api/auth/keys`);
        if (response.ok) {
            const data = await response.json();
            const container = document.getElementById('api-keys-container');
            const statsContainer = document.getElementById('api-stats-container');
            
            const visibleKeys = Object.keys(data).filter(name => name !== '_internal_system_key');
            if (visibleKeys.length === 0) {
                container.innerHTML = '<p style="color: #7f8c8d;">No API keys created yet</p>';
                document.getElementById('api-key-stats').style.display = 'none';
            } else {
                // Build API keys table
                let keysHtml = `
                    <table class="models-table" style="width: 100%;">
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>API Key</th>
                                <th>Created</th>
                                <th>Last Used</th>
                                <th>Requests</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                `;
                
                let totalRequests = 0;
                
                for (const [name, info] of Object.entries(data)) {
                    // Skip internal system key from display
                    if (name === '_internal_system_key') {
                        totalRequests += info.request_count;
                        continue;
                    }
                    
                    const createdDate = new Date(info.created_at).toLocaleDateString();
                    const lastUsed = info.last_used ? new Date(info.last_used).toLocaleString() : 'Never';
                    totalRequests += info.request_count;
                    
                    const keyElementId = `api-key-${name.replace(/[^a-zA-Z0-9]/g, '-')}-${Date.now()}`;
                    const hasFullKey = info.full_key !== undefined;
                    
                    keysHtml += `
                        <tr>
                            <td class="model-name-cell">${escapeHtml(name)}</td>
                            <td>
                                <div style="display: flex; align-items: center; gap: 8px;">
                                    <code id="${keyElementId}" style="background: #f8f9fa; padding: 4px 8px; border-radius: 4px; font-size: 12px; ${hasFullKey ? 'font-family: monospace;' : ''}">${hasFullKey ? info.full_key : info.key_preview}</code>
                                    ${hasFullKey ? 
                                        `<button onclick="copyToClipboard('${keyElementId}', event)" class="btn-small btn-copy" style="padding: 2px 8px; font-size: 11px;">Copy</button>` : 
                                        `<span class="text-muted" style="font-size: 11px; color: #7f8c8d;">(partial)</span>`
                                    }
                                </div>
                            </td>
                            <td>${createdDate}</td>
                            <td style="font-size: 13px;">${lastUsed}</td>
                            <td style="text-align: center;"><strong>${info.request_count}</strong></td>
                            <td style="text-align: center;">
                                <button class="btn-small btn-delete" onclick="deleteApiKey('${escapeHtml(name)}')" style="padding: 4px 12px;">Delete</button>
                            </td>
                        </tr>
                    `;
                }
                
                keysHtml += `
                        </tbody>
                    </table>
                `;
                
                container.innerHTML = keysHtml;
                
                // Show stats
                document.getElementById('api-key-stats').style.display = 'block';
                
                // Build stats grid
                let statsHtml = '<div class="stats-grid">';
                let visibleKeyCount = Object.keys(data).filter(name => name !== '_internal_system_key').length;
                statsHtml += `
                    <div class="stat-card">
                        <div class="stat-value">${visibleKeyCount}</div>
                        <div class="stat-label">Active Keys</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">${totalRequests}</div>
                        <div class="stat-label">Total Requests</div>
                    </div>
                `;
                
                // Add per-key stats
                for (const [name, info] of Object.entries(data)) {
                    // Skip internal key from stats display
                    if (name === '_internal_system_key') continue;
                    
                    if (info.request_count > 0) {
                        statsHtml += `
                            <div class="stat-card">
                                <div class="stat-value">${info.request_count}</div>
                                <div class="stat-label">${escapeHtml(name)}</div>
                            </div>
                        `;
                    }
                }
                
                statsHtml += '</div>';
                statsContainer.innerHTML = statsHtml;
            }
        }
    } catch (error) {
        console.error('Failed to load API keys:', error);
    }
}

// Make these functions global so they can be called from onclick handlers
window.copyToClipboard = function(elementId, event) {
    // Get the element containing the text to copy
    const element = document.getElementById(elementId);
    if (!element) {
        console.error('Element not found:', elementId);
        return;
    }
    
    const text = element.textContent.trim();
    
    // Get the button that was clicked
    const btn = event ? event.target : document.activeElement;
    
    // Check if clipboard API is available
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
        // Show temporary success message
        const originalText = btn.textContent;
        const originalBg = btn.style.backgroundColor;
        const originalColor = btn.style.color;
        
        btn.textContent = 'Copied!';
        btn.style.backgroundColor = '#27ae60';
        btn.style.color = 'white';
        
        setTimeout(() => {
            btn.textContent = originalText;
            btn.style.backgroundColor = originalBg;
            btn.style.color = originalColor;
        }, 1500);
        }).catch(err => {
            console.error('Failed to copy with clipboard API:', err);
            copyUsingFallback(text, btn);
        });
    } else {
        // Use fallback method if clipboard API is not available
        copyUsingFallback(text, btn);
    }
}

function copyUsingFallback(text, btn) {
    try {
        const textarea = document.createElement('textarea');
        textarea.value = text;
        textarea.style.position = 'fixed';
        textarea.style.opacity = '0';
        textarea.style.pointerEvents = 'none';
        document.body.appendChild(textarea);
        textarea.select();
        const success = document.execCommand('copy');
        document.body.removeChild(textarea);
        
        if (success) {
            // Show success message
            const originalText = btn.textContent;
            const originalBg = btn.style.backgroundColor;
            const originalColor = btn.style.color;
            
            btn.textContent = 'Copied!';
            btn.style.backgroundColor = '#27ae60';
            btn.style.color = 'white';
            
            setTimeout(() => {
                btn.textContent = originalText;
                btn.style.backgroundColor = originalBg;
                btn.style.color = originalColor;
            }, 1500);
        } else {
            alert('Failed to copy to clipboard. Please copy manually.');
        }
    } catch (fallbackErr) {
        console.error('Fallback copy failed:', fallbackErr);
        alert('Failed to copy to clipboard. Please copy manually.');
    }
}

window.copyApiKey = function(elementId) {
    const element = document.getElementById(elementId);
    const text = element.textContent;
    
    navigator.clipboard.writeText(text).then(() => {
        alert('API key copied to clipboard!');
    }).catch(err => {
        console.error('Failed to copy:', err);
    });
}

window.deleteApiKey = async function(name) {
    if (!confirm(`Are you sure you want to delete the API key "${name}"?`)) {
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/api/auth/keys/${encodeURIComponent(name)}`, {
            method: 'DELETE'
        });
        
        if (response.ok) {
            await loadApiKeys();
        } else {
            const error = await response.json();
            alert(`Error: ${error.detail || 'Failed to delete API key'}`);
        }
    } catch (error) {
        alert(`Error: ${error.message}`);
    }
}

// Log viewer functions
let logRefreshInterval = null;

window.openLogModal = function() {
    document.getElementById('log-modal').style.display = 'block';
    refreshLogs();
    // Auto-refresh logs every 5 seconds while modal is open
    logRefreshInterval = setInterval(refreshLogs, 5000);
}

window.closeLogModal = function() {
    document.getElementById('log-modal').style.display = 'none';
    // Stop auto-refresh when modal is closed
    if (logRefreshInterval) {
        clearInterval(logRefreshInterval);
        logRefreshInterval = null;
    }
}

window.refreshLogs = async function() {
    const lines = document.getElementById('log-lines-select').value;
    const logContent = document.getElementById('log-content');
    const logTimestamp = document.getElementById('log-timestamp');
    
    try {
        // Get internal API key if needed
        const headers = {};
        const keyResponse = await fetch(`${API_BASE}/api/auth/internal-key`);
        if (keyResponse.ok) {
            const keyData = await keyResponse.json();
            if (keyData.api_key) {
                headers['X-API-Key'] = keyData.api_key;
            }
        }
        
        const response = await fetch(`${API_BASE}/api/logs?lines=${lines}`, {
            headers: headers
        });
        
        if (response.ok) {
            const data = await response.json();
            
            // Format logs with syntax highlighting
            let formattedLogs = data.logs;
            
            // Highlight log levels
            formattedLogs = formattedLogs.replace(/\bERROR\b/g, '<span style="color: #f44336;">ERROR</span>');
            formattedLogs = formattedLogs.replace(/\bWARNING\b/g, '<span style="color: #ff9800;">WARNING</span>');
            formattedLogs = formattedLogs.replace(/\bINFO\b/g, '<span style="color: #4caf50;">INFO</span>');
            formattedLogs = formattedLogs.replace(/\bDEBUG\b/g, '<span style="color: #2196f3;">DEBUG</span>');
            
            // Highlight timestamps (ISO format)
            formattedLogs = formattedLogs.replace(/\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}/g, '<span style="color: #9e9e9e;">$&</span>');
            
            // Highlight HTTP status codes
            formattedLogs = formattedLogs.replace(/\b[1-5]\d{2}\b/g, '<span style="color: #00bcd4;">$&</span>');
            
            logContent.innerHTML = formattedLogs || 'No logs available';
            logTimestamp.textContent = `Last updated: ${new Date().toLocaleTimeString()}`;
        } else {
            logContent.textContent = 'Failed to load logs';
        }
    } catch (error) {
        logContent.textContent = `Error loading logs: ${error.message}`;
    }
}

// Close modal when clicking outside of it
window.onclick = function(event) {
    const modal = document.getElementById('log-modal');
    if (event.target === modal) {
        closeLogModal();
    }
}