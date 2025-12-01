/**
 * EL AMANECER - Gemma-2 LLM + Consciousness System
 */

const API_BASE = 'http://localhost:8000';
let latencyChart = null;
let throughputChart = null;

function switchTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(nav => nav.classList.remove('active'));

    const selectedTab = document.getElementById(tabName + '-tab');
    if (selectedTab) selectedTab.classList.add('active');

    document.querySelectorAll('.nav-item').forEach(item => {
        if (item.getAttribute('onclick')?.includes(tabName)) item.classList.add('active');
    });

    if (tabName === 'analytics' && !latencyChart) setTimeout(initAnalytics, 100);
    if (tabName === 'consciousness') setTimeout(initConsciousness, 100);
}

window.switchTab = switchTab;

// CONSCIOUSNESS
const EMOTIONS = [
    'alegría', 'tristeza', 'miedo', 'ira', 'sorpresa', 'asco', 'amor', 'orgullo',
    'vergüenza', 'culpa', 'envidia', 'celos', 'gratitud', 'admiración', 'desprecio',
    'ansiedad', 'esperanza', 'alivio', 'decepción', 'frustración', 'satisfacción',
    'curiosidad', 'aburrimiento', 'confusión', 'interés', 'diversión', 'serenidad',
    'nostalgia', 'melancolía', 'euforia', 'pánico', 'calma', 'excitación', 'ternura', 'compasión'
];

function initConsciousness() {
    const grid = document.getElementById('emotions-grid');
    if (!grid || grid.children.length > 0) return;

    EMOTIONS.forEach(emotion => {
        const cell = document.createElement('div');
        cell.className = 'emotion-cell';
        cell.innerHTML = `<div class="emotion-intensity">0</div><div class="emotion-name">${emotion}</div>`;
        cell.onclick = () => activateEmotion(emotion, cell);
        grid.appendChild(cell);
    });

    updateConsciousnessMetrics();
    setInterval(updateConsciousnessMetrics, 3000);
}

function activateEmotion(emotion, cell) {
    document.querySelectorAll('.emotion-cell').forEach(c => c.classList.remove('active'));
    cell.classList.add('active');
    cell.querySelector('.emotion-intensity').textContent = '1';
}

async function updateConsciousnessMetrics() {
    try {
        const query = `query { consciousness(consciousnessId: "main") { phiValue emotionalDepth mindfulnessLevel currentEmotion } }`;
        const response = await fetch(`${API_BASE}/graphql`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.data?.consciousness) {
                const c = data.data.consciousness;
                const phiEl = document.getElementById('phi-value');
                const emotionEl = document.getElementById('cons-emotion');
                const arousalEl = document.getElementById('cons-arousal');
                const valenceEl = document.getElementById('cons-valence');

                if (phiEl) phiEl.textContent = c.phiValue.toFixed(2);
                if (emotionEl) emotionEl.textContent = c.currentEmotion;
                if (arousalEl) arousalEl.textContent = c.emotionalDepth.toFixed(2);
                if (valenceEl) valenceEl.textContent = (c.mindfulnessLevel - 0.5).toFixed(2);
            }
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function processExperience() {
    updateConsciousnessMetrics();
}

function regulateEmotion() {
    document.querySelectorAll('.emotion-cell').forEach(cell => {
        cell.classList.remove('active');
        cell.querySelector('.emotion-intensity').textContent = '0';
    });
}

// CHAT
async function sendMessage() {
    const input = document.getElementById('chat-input');
    const message = input.value.trim();

    if (!message) return;

    addMessage('user', message);
    input.value = '';

    try {
        const mutation = `mutation SendMessage($content: String!, $conversationId: String!) { sendMessage(input: { conversationId: $conversationId content: $content consciousnessEnhanced: true }) { id content } }`;
        const response = await fetch(`${API_BASE}/graphql`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: mutation,
                variables: { content: message, conversationId: 'web-' + Date.now() }
            })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.data?.sendMessage) {
                addMessage('assistant', data.data.sendMessage.content);
            } else if (data.errors) {
                addMessage('assistant', 'Error: ' + data.errors.map(e => e.message).join('; '));
            }
        }
    } catch (error) {
        addMessage('assistant', 'Backend offline.');
    }
}

function addMessage(role, content) {
    const container = document.getElementById('chat-messages');
    const group = document.createElement('div');
    group.classList.add('message-group', role);

    const avatar = document.createElement('div');
    avatar.classList.add('message-avatar');
    avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-brain"></i>';

    const bubble = document.createElement('div');
    bubble.classList.add('message-bubble');
    bubble.innerHTML = `<p>${content}</p>`;

    group.appendChild(avatar);
    group.appendChild(bubble);
    container.appendChild(group);
    container.scrollTop = container.scrollHeight;
}

// ANALYTICS
function initAnalytics() {
    initCharts();
    updateMetrics();
    startClock();
    setInterval(updateMetrics, 2000);
    setInterval(updateCharts, 2000);
}

function initCharts() {
    const latencyCtx = document.getElementById('latency-chart')?.getContext('2d');
    if (latencyCtx) {
        latencyChart = new Chart(latencyCtx, {
            type: 'line',
            data: { labels: [], datasets: [{ label: 'Latency', data: [], borderColor: '#00f0ff', backgroundColor: 'rgba(0,240,255,0.1)', borderWidth: 3, tension: 0.4, fill: true, pointRadius: 0 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } } } }
        });
    }

    const throughputCtx = document.getElementById('throughput-chart')?.getContext('2d');
    if (throughputCtx) {
        throughputChart = new Chart(throughputCtx, {
            type: 'bar',
            data: { labels: [], datasets: [{ label: 'Throughput', data: [], backgroundColor: 'rgba(188,19,254,0.6)', borderColor: '#bc13fe', borderWidth: 2, borderRadius: 8 }] },
            options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }, x: { grid: { display: false }, ticks: { color: '#94a3b8', maxTicksLimit: 8 } } } }
        });
    }
}

function updateCharts() {
    if (!latencyChart || !throughputChart) return;

    const now = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const latency = Math.floor(Math.random() * 50 + 100);
    const throughput = Math.floor(Math.random() * 200 + 300);

    latencyChart.data.labels.push(now);
    latencyChart.data.datasets[0].data.push(latency);
    if (latencyChart.data.labels.length > 15) {
        latencyChart.data.labels.shift();
        latencyChart.data.datasets[0].data.shift();
    }
    latencyChart.update('none');

    throughputChart.data.labels.push(now);
    throughputChart.data.datasets[0].data.push(throughput);
    if (throughputChart.data.labels.length > 15) {
        throughputChart.data.labels.shift();
        throughputChart.data.datasets[0].data.shift();
    }
    throughputChart.update('none');
}

function updateMetrics() {
    const metrics = {
        latency: Math.floor(Math.random() * 50 + 100) + 'ms',
        throughput: Math.floor(Math.random() * 200 + 300) + '/s',
        cpu: Math.floor(Math.random() * 30 + 30) + '%',
        memory: (Math.random() * 1 + 2).toFixed(1) + 'GB'
    };

    Object.keys(metrics).forEach(key => {
        const el = document.getElementById('metric-' + key);
        if (el) el.textContent = metrics[key];
    });
}

function startClock() {
    const updateClock = () => {
        const el = document.getElementById('current-time');
        if (el) el.textContent = new Date().toLocaleTimeString('en-US', { hour12: false });
    };
    updateClock();
    setInterval(updateClock, 1000);
}

function autoResizeTextarea() {
    const textarea = document.getElementById('chat-input');
    if (!textarea) return;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 EL AMANECER');

    const sendBtn = document.getElementById('send-btn');
    if (sendBtn) sendBtn.addEventListener('click', sendMessage);

    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.addEventListener('input', autoResizeTextarea);
        chatInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    }

    switchTab('chat');
});
