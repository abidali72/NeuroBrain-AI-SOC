/**
 * 🧠 Neuro Brain - Chat UI Logic
 * Handles real-time messaging, command parsing, and alert polling.
 */

document.addEventListener('DOMContentLoaded', () => {
    const chatContainer = document.getElementById('chat-container');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // ─── CHAT ENGINE ───
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Add User Message to UI
        appendMessage('user', text);
        userInput.value = '';

        // 2. Mock AI "Processing"
        const typingId = appendMessage('ai', '...', true);

        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await response.json();

            // 3. Update Typing Message with actual AI Response
            updateMessage(typingId, data.response, data.card);
        } catch (error) {
            updateMessage(typingId, "Error connecting to Neuro Brain Core. Please check backend.");
        }
    }

    function appendMessage(role, text, isTyping = false) {
        const msgId = 'msg-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        msgDiv.id = msgId;

        const avatar = role === 'ai' ? '🤖' : '👤';
        const avatarClass = role === 'ai' ? 'ai-avatar' : 'user-avatar';

        msgDiv.innerHTML = `
            <div class="avatar ${avatarClass}">${avatar}</div>
            <div class="content ${role}-content">${text}</div>
        `;

        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
        return msgId;
    }

    function updateMessage(id, text, card = null) {
        const msgDiv = document.getElementById(id);
        if (!msgDiv) return;

        const contentDiv = msgDiv.querySelector('.content');
        contentDiv.innerHTML = text;

        if (card) {
            const cardDiv = document.createElement('div');
            cardDiv.className = 'threat-card';
            cardDiv.innerHTML = `
                <div class="threat-header">
                    <span class="threat-title">${card.title}</span>
                    <span class="severity-tag">${card.severity}</span>
                </div>
                <div class="threat-details">${card.details}</div>
            `;
            contentDiv.appendChild(cardDiv);
        }

        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // ─── EVENT LISTENERS ───
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendMessage();
    });

    // ─── BACKGROUND ALERT POLLING ───
    async function pollAlerts() {
        try {
            const response = await fetch('/alerts');
            const data = await response.json();

            if (data.alert) {
                appendAlertMessage(data.alert);
                // Update Sidebar Stats
                updateStats(data.stats);
            }
        } catch (e) { }
        setTimeout(pollAlerts, 3000);
    }

    function appendAlertMessage(alert) {
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message';
        msgDiv.innerHTML = `
            <div class="avatar ai-avatar">🚨</div>
            <div class="content ai-content">
                <div class="threat-card">
                    <div class="threat-header">
                        <span class="threat-title">${alert.type} DETECTED</span>
                        <span class="severity-tag">${alert.severity}</span>
                    </div>
                    <div class="threat-details">
                        <b>Source:</b> ${alert.source}<br>
                        <b>Detail:</b> ${alert.message}
                    </div>
                </div>
            </div>
        `;
        chatContainer.appendChild(msgDiv);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function updateStats(stats) {
        if (stats.health) document.getElementById('stat-health').innerText = stats.health + '%';
        if (stats.detections) document.getElementById('stat-detections').innerText = stats.detections;
        if (stats.blocked) document.getElementById('stat-blocked').innerText = stats.blocked;
    }

    pollAlerts(); // Start polling
});
