/* =========================================================
   DOMAINQA — FRONTEND APPLICATION (v2 Next-Level)
========================================================= */

const API_BASE =
    window.__APP_CONFIG__ && window.__APP_CONFIG__.apiBase !== undefined
        ? window.__APP_CONFIG__.apiBase
        : "http://127.0.0.1:8010";

/* =========================================================
   STATE
========================================================= */

let selectedPdf = null;
let currentSessionId = "session-" + Math.random().toString(36).substring(2, 11);
let conversationHistory = [];
let currentAnswerText = "";
let currentSources = [];
let currentFeedback = null; // null | "up" | "down"
let isStreaming = false;

/* =========================================================
   DOM ELEMENTS
========================================================= */

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const selectedFile = document.getElementById("selectedFile");
const uploadButton = document.getElementById("uploadButton");
const uploadProgress = document.getElementById("uploadProgress");
const progressBar = document.getElementById("progressBar");
const progressText = document.getElementById("progressText");
const progressPercent = document.getElementById("progressPercent");
const uploadNotification = document.getElementById("uploadNotification");

const questionInput = document.getElementById("questionInput");
const characterCount = document.getElementById("characterCount");
const askButton = document.getElementById("askButton");

const toast = document.getElementById("toast");
const toastTitle = document.getElementById("toastTitle");
const toastMessage = document.getElementById("toastMessage");
const answerState = document.getElementById("answerState");

const dashboardView = document.getElementById("dashboardView");
const documentsView = document.getElementById("documentsView");
const documentsContainer = document.getElementById("documentsContainer");
const documentsList = document.getElementById("documentsList");
const documentCount = document.getElementById("documentCount");
const historyView = document.getElementById("historyView");
const historyContainer = document.getElementById("historyContainer");
const analyticsView = document.getElementById("analyticsView");
const analyticsContainer = document.getElementById("analyticsContainer");

const navDashboard = document.getElementById("navDashboard");
const navDocuments = document.getElementById("navDocuments");
const navHistory = document.getElementById("navHistory");
const navAnalytics = document.getElementById("navAnalytics");

const refreshDocuments = document.getElementById("refreshDocuments");
const refreshAnalytics = document.getElementById("refreshAnalytics");
const resetAnalyticsButton = document.getElementById("resetAnalytics");
const clearHistory = document.getElementById("clearHistory");
const exportHistory = document.getElementById("exportHistory");
const toggleUpload = document.getElementById("toggleUpload");
const clearChatBtn = document.getElementById("clearChatBtn");
const clearEvidenceButton = document.getElementById("clearEvidence");
const chatArea = document.getElementById("chatArea");
const chunkModal = document.getElementById("chunkModal");
const closeChunkModal = document.getElementById("closeChunkModal");

function setupEventListeners() {
    if (fileInput) {
        fileInput.addEventListener("change", () => {
            if (fileInput.files.length) handleFile(fileInput.files[0]);
        });
    }

    if (dropZone) {
        dropZone.addEventListener("click", event => {
            if (event.target !== fileInput) fileInput?.click();
        });
        dropZone.addEventListener("dragover", event => {
            event.preventDefault();
            dropZone.classList.add("drag-over");
        });
        dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
        dropZone.addEventListener("drop", event => {
            event.preventDefault();
            dropZone.classList.remove("drag-over");
            const file = event.dataTransfer?.files?.[0];
            if (file) handleFile(file);
        });
    }

    if (uploadButton) uploadButton.addEventListener("click", uploadPdf);
    if (askButton) askButton.addEventListener("click", askQuestion);

    if (questionInput) {
        questionInput.addEventListener("input", updateCharacterCount);
        questionInput.addEventListener("keydown", event => {
            if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                askQuestion();
            }
        });
        questionInput.addEventListener("input", () => {
            questionInput.style.height = "auto";
            questionInput.style.height = `${Math.min(questionInput.scrollHeight, 120)}px`;
        });
    }

    document.querySelectorAll(".suggestion-chip").forEach(button => {
        button.addEventListener("click", () => {
            questionInput.value = button.textContent.trim();
            updateCharacterCount();
            questionInput.focus();
        });
    });

    navDashboard?.addEventListener("click", () => navigateTo("dashboard"));
    navDocuments?.addEventListener("click", () => navigateTo("documents"));
    navHistory?.addEventListener("click", () => navigateTo("history"));
    navAnalytics?.addEventListener("click", () => navigateTo("analytics"));
    refreshDocuments?.addEventListener("click", loadDocuments);
    refreshAnalytics?.addEventListener("click", loadAnalytics);
    resetAnalyticsButton?.addEventListener("click", resetAnalytics);
    toggleUpload?.addEventListener("click", () => {
        document.querySelector(".upload-section")?.classList.toggle("collapsed");
    });

    clearHistory?.addEventListener("click", () => {
        conversationHistory = [];
        saveHistory();
        renderHistoryView();
        showToast("History Cleared", "Conversation history was cleared.");
    });
    exportHistory?.addEventListener("click", exportHistoryAsMarkdown);
    clearChatBtn?.addEventListener("click", () => {
        if (chatArea) chatArea.innerHTML = createWelcomeMessage();
        conversationHistory = [];
        saveHistory();
    });
    clearEvidenceButton?.addEventListener("click", clearEvidence);
    closeChunkModal?.addEventListener("click", closeChunkViewer);
    chunkModal?.addEventListener("click", event => {
        if (event.target === chunkModal) closeChunkViewer();
    });
}

setupEventListeners();
checkSystemStatus();
loadDocuments();

/* =========================================================
   NAVIGATION
========================================================= */

function navigateTo(view) {
    const views = [dashboardView, documentsView, historyView, analyticsView];
    const navs = [navDashboard, navDocuments, navHistory, navAnalytics];

    views.forEach(v => v && v.classList.add("hidden"));
    navs.forEach(n => n && n.classList.remove("active"));

    switch (view) {
        case "dashboard":
            dashboardView.classList.remove("hidden");
            navDashboard.classList.add("active");
            break;
        case "documents":
            documentsView.classList.remove("hidden");
            navDocuments.classList.add("active");
            loadDocuments();
            break;
        case "history":
            historyView.classList.remove("hidden");
            navHistory.classList.add("active");
            renderHistoryView();
            break;
        case "analytics":
            analyticsView.classList.remove("hidden");
            navAnalytics.classList.add("active");
            loadAnalytics();
            break;
    }
}

/* =========================================================
   SYSTEM STATUS
========================================================= */

async function checkSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/api/status`);
        if (!response.ok) throw new Error("API unavailable");
        const data = await response.json();
        console.log("DomainQA connected:", data);
    } catch (error) {
        console.error("API status error:", error);
        showToast("Backend Offline", "Unable to connect to the FastAPI server.");
    }
}

/* =========================================================
   FILE HANDLING
========================================================= */

const ACCEPTED_FILE_TYPES = [
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
];
const ACCEPTED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg", ".webp"];

function handleFile(file) {
    const nameLower = file.name.toLowerCase();
    const typeOk = ACCEPTED_FILE_TYPES.includes(file.type);
    const extOk  = ACCEPTED_EXTENSIONS.some(ext => nameLower.endsWith(ext));

    if (!typeOk && !extOk) {
        showNotification("Please select a PDF or image file (PNG, JPG, WEBP).", "error");
        return;
    }

    selectedPdf = file;
    const isPdf = nameLower.endsWith(".pdf");
    if (selectedFile) selectedFile.textContent = `Selected: ${file.name}`;
    if (uploadButton) uploadButton.disabled = false;
    showNotification(
        isPdf
            ? "PDF selected and ready for indexing."
            : "Image selected. Gemini Vision will analyze and index it.",
        "success"
    );
}

/* =========================================================
   UPLOAD PDF
========================================================= */

async function uploadPdf() {
    if (!selectedPdf) {
        showNotification("Please choose a PDF first.", "error");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedPdf);

    uploadButton.disabled = true;
    if (uploadProgress) uploadProgress.classList.add("active");
    setProgress(10, "Uploading document...");

    // Simulate progress while the server processes the file
    let progress = 10;
    const progressMessages = [
        { pct: 25, text: "Extracting PDF text..." },
        { pct: 45, text: "Splitting into chunks..." },
        { pct: 65, text: "Generating embeddings..." },
        { pct: 80, text: "Storing in knowledge base..." },
        { pct: 90, text: "Almost done..." },
    ];
    let msgIndex = 0;
    const progressTimer = setInterval(() => {
        if (msgIndex < progressMessages.length) {
            progress = progressMessages[msgIndex].pct;
            setProgress(progress, progressMessages[msgIndex].text);
            msgIndex++;
        }
    }, 5000);

    try {
        const response = await fetch(`${API_BASE}/api/upload`, {
            method: "POST",
            body: formData
        });

        clearInterval(progressTimer);
        setProgress(100, "Indexing complete...");

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || data.message || "Upload failed.");
        }

        if (data.duplicate) {
            showNotification("This document has already been indexed.", "success");
            showToast("Already Indexed", "This document is already in the knowledge base.");
        } else {
            const pages  = data.pages  ?? 0;
            const chunks = data.chunks ?? 0;
            const tables = data.tables ?? 0;
            const images = data.images ?? 0;
            let details = `${pages} pages · ${chunks} chunks`;
            if (tables > 0) details += ` · ${tables} table${tables !== 1 ? "s" : ""}`;
            if (images > 0) details += ` · ${images} image${images !== 1 ? "s" : ""} analyzed`;
            showNotification(`Successfully indexed: ${details}`, "success");
            showToast("Document Indexed", `${chunks} chunks added to the knowledge base.`);
            setTimeout(() => {
                const uploadSection = document.querySelector(".upload-section");
                if (uploadSection) uploadSection.classList.add("collapsed");
            }, 1000);
        }

        await loadDocuments();
    } catch (error) {
        clearInterval(progressTimer);
        console.error("Upload error:", error);
        showNotification(error.message || "Unable to upload the document.", "error");
        showToast("Upload Failed", error.message || "Something went wrong.");
    } finally {
        setTimeout(() => {
            if (uploadProgress) uploadProgress.classList.remove("active");
            setProgress(0, "Processing...");
        }, 1500);
        uploadButton.disabled = false;
    }
}

function setProgress(percent, text) {
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressPercent) progressPercent.textContent = `${percent}%`;
    if (progressText) progressText.textContent = text;
}

/* =========================================================
   ASK QUESTION (with SSE streaming)
========================================================= */

async function askQuestion() {
    const question = questionInput.value.trim();
    if (!question) {
        showToast("Question Required", "Please enter a question.");
        questionInput.focus();
        return;
    }

    if (isStreaming) return;
    isStreaming = true;
    askButton.disabled = true;
    askButton.setAttribute("aria-busy", "true");
    const askButtonLabel = askButton.querySelector("span");
    if (askButtonLabel) askButtonLabel.textContent = "Generating...";
    if (answerState) {
        answerState.textContent = "Generating";
        answerState.classList.add("is-loading");
    }
    currentFeedback = null;
    currentSources = [];
    renderSources([]);

    // Remove welcome message if present
    const welcomeMsg = chatArea.querySelector(".welcome-message");
    if (welcomeMsg) welcomeMsg.remove();

    // Render user bubble
    appendChatBubble("user", question);
    questionInput.value = "";
    questionInput.style.height = "auto";
    updateCharacterCount();

    // Show streaming placeholder
    currentAnswerText = "";
    let lastDoneAnswer = "";
    let lastDoneSources = [];
    const streamingBubble = appendChatBubble("assistant", "", null, true, question);

    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE}/api/ask/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: question,
                session_id: currentSessionId,
                history: conversationHistory.slice(-5).map(h => ({
                    role: h.role,
                    content: h.content
                })),
                use_hybrid_search: true,
                use_query_rewriting: false,
                use_guard: false,
                evaluate: false
            })
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.detail || errData?.message || `HTTP ${response.status}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let errorSeen = false;
        let streamError = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (line.startsWith("event: error")) {
                    // Next data line has the error detail
                    errorSeen = true;
                    continue;
                }
                if (line.startsWith("data: ")) {
                    const payload = line.slice(6).trim();
                    if (payload === "[DONE]") continue;

                    try {
                        const chunk = JSON.parse(payload);
                        console.log("Packet received:", chunk); // DEBUG

                        if (chunk.detail && errorSeen) {
                            // Error event data
                            currentAnswerText = "";
                            streamError = chunk.detail;
                            streamingBubble.querySelector(".bubble-content").innerHTML =
                                `<span style="color: var(--red);">${escapeHtml(streamError)}</span>`;
                        } else {
                            if (chunk.token) {
                                currentAnswerText += chunk.token;
                                renderStreamingInBubble(streamingBubble, currentAnswerText);
                            }
                            if (chunk.sources) {
                                currentSources = chunk.sources;
                            }
                            if (chunk.answer) {
                                lastDoneAnswer = chunk.answer;
                                lastDoneSources = chunk.sources || currentSources;
                                if (!currentAnswerText) currentAnswerText = lastDoneAnswer;
                                currentSources = lastDoneSources;
                            }
                        }
                    } catch (e) {
                        console.warn("SSE parse error:", e, "payload:", payload);
                    }
                }
            }
        }

        // Process remaining buffer
        if (buffer.startsWith("data: ")) {
            const payload = buffer.slice(6).trim();
            if (payload && payload !== "[DONE]") {
                try {
                    const chunk = JSON.parse(payload);
                    if (chunk.detail && errorSeen) {
                        streamError = chunk.detail;
                    } else {
                        if (chunk.token) {
                            currentAnswerText += chunk.token;
                            renderStreamingInBubble(streamingBubble, currentAnswerText);
                        }
                        if (chunk.sources) currentSources = chunk.sources;
                        if (chunk.answer) {
                            lastDoneAnswer = chunk.answer;
                            lastDoneSources = chunk.sources || currentSources;
                            if (!currentAnswerText) currentAnswerText = lastDoneAnswer;
                            currentSources = lastDoneSources;
                        }
                    }
                } catch (e) {
                    currentAnswerText += payload;
                    renderStreamingInBubble(streamingBubble, currentAnswerText);
                }
            }
        }

        if (streamError) {
            throw new Error(streamError);
        }

        const elapsed = Math.round(performance.now() - startTime);

        // Fallback: if streaming tokens were empty, try to use the done event answer
        if (!currentAnswerText && lastDoneAnswer) {
            currentAnswerText = lastDoneAnswer;
            currentSources = lastDoneSources || currentSources;
        }

        // Finalize - remove cursor and set final content
        finalizeBubble(streamingBubble, currentAnswerText, currentSources, elapsed);

        // Store in conversation history
        conversationHistory.push({
            role: "user",
            content: question,
            timestamp: Date.now()
        });
        conversationHistory.push({
            role: "assistant",
            content: currentAnswerText,
            sources: currentSources,
            latency_ms: elapsed,
            timestamp: Date.now()
        });
        saveHistory();

        scrollChatToBottom();

    } catch (error) {
        console.error("Question error:", error);
        currentAnswerText = "";
        if (answerState) {
            answerState.textContent = "Failed";
            answerState.classList.remove("is-loading");
        }

        // Show error in bubble
        const content = streamingBubble?.querySelector(".bubble-content");
        if (content) {
            content.innerHTML = `<span class="answer-error">Unable to generate answer: ${escapeHtml(error.message)}</span>`;
        }

        showToast("Question Failed", error.message || "Something went wrong.");

    } finally {
        isStreaming = false;
        askButton.disabled = false;
        askButton.setAttribute("aria-busy", "false");
        if (askButtonLabel) askButtonLabel.textContent = "Ask Question";
        if (answerState?.classList.contains("is-loading")) {
            answerState.textContent = "Ready";
            answerState.classList.remove("is-loading");
        }
    }
}

function appendChatBubble(role, content, sources, isStreaming = false, question = null) {
    if (!chatArea) return null;

    const bubble = document.createElement("div");
    bubble.className = `chat-bubble chat-bubble-${role}`;
    if (question) bubble.dataset.question = question;

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.textContent = role === "user" ? "You" : "DomainQA";
    bubble.appendChild(meta);

    const contentElement = document.createElement("div");
    contentElement.className = "bubble-content";
    if (role === "assistant") {
        contentElement.innerHTML = isStreaming
            ? '<div class="thinking-indicator" role="status" aria-label="Generating answer"><span></span><span></span><span></span></div>'
            : formatAnswer(content);
    } else {
        contentElement.textContent = content;
    }

    bubble.appendChild(contentElement);
    chatArea.appendChild(bubble);
    scrollChatToBottom();
    return bubble;
}

function scrollChatToBottom() {
    if (chatArea) chatArea.scrollTop = chatArea.scrollHeight;
}

/* =========================================================
   STREAMING RENDER
========================================================= */

function renderStreamingInBubble(bubble, text) {
    const contentEl = bubble.querySelector(".bubble-content");
    if (contentEl) {
        contentEl.innerHTML = `${formatAnswer(text)}<span class="streaming-cursor"></span>`;
    }
}

function finalizeBubble(bubble, text, sources, latencyMs) {
    const contentEl = bubble.querySelector(".bubble-content");
    if (contentEl) {
        let html = formatAnswer(text);
        if (sources && sources.length > 0) {
            html += `<div class="citation-summary" style="margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border-light); font-size: 11px; color: var(--text-3);">
                ${sources.length} source${sources.length === 1 ? "" : "s"} cited
            </div>`;
        }
        if (latencyMs) {
            html += `<div style="font-size: 11px; color: var(--text-3); margin-top: 4px;">
                ${latencyMs < 1000 ? latencyMs + "ms" : (latencyMs / 1000).toFixed(1) + "s"}
            </div>`;
        }

        // Add feedback buttons
        html += `
            <div class="feedback-actions">
                <button class="feedback-btn" data-rating="up" onclick="sendFeedback('up', '${escapeHtml(bubble.dataset.question || '')}', '${escapeHtml(text.substring(0, 50))}', this.closest('.chat-bubble'))">👍 Like</button>
                <button class="feedback-btn" data-rating="down" onclick="sendFeedback('down', '${escapeHtml(bubble.dataset.question || '')}', '${escapeHtml(text.substring(0, 50))}', this.closest('.chat-bubble'))">👎 Dislike</button>
            </div>
        `;
        contentEl.innerHTML = html;
    }
    renderSources(sources || []);
}

// ---------------------------------------------------------------------------
// Markdown table → HTML table converter
// ---------------------------------------------------------------------------

function markdownTableToHtml(mdTable) {
    const lines = mdTable.trim().split("\n").filter(l => l.trim());
    if (lines.length < 2) return `<pre>${escapeHtml(mdTable)}</pre>`;

    const parseRow = (line) =>
        line.replace(/^\|/, "").replace(/\|$/, "")
            .split("|")
            .map(cell => cell.trim());

    const headers = parseRow(lines[0]);
    const bodyLines = lines.slice(2); // skip separator line

    let html = `<div class="table-wrapper"><table class="retrieved-table">`;
    html += `<thead><tr>${headers.map(h => `<th>${escapeHtml(h)}</th>`).join("")}</tr></thead>`;
    html += `<tbody>`;
    for (const line of bodyLines) {
        if (!line.trim() || /^\|[-|\s]+\|$/.test(line)) continue;
        const cells = parseRow(line);
        html += `<tr>${cells.map(c => `<td>${escapeHtml(c)}</td>`).join("")}</tr>`;
    }
    html += `</tbody></table></div>`;
    return html;
}

function formatAnswer(text) {
    // First, extract and convert Markdown tables before general processing
    let html = text;

    // Replace Markdown tables (lines starting and ending with |)
    html = html.replace(
        /((\|[^\n]+\|\n)(\|[-: |]+\|\n)((?:\|[^\n]+\|\n?)*))/g,
        (match) => markdownTableToHtml(match)
    );

    // Escape remaining non-table HTML
    // We need to escape only non-table parts — use a placeholder approach
    const tablePlaceholders = [];
    html = html.replace(/<div class="table-wrapper">[\s\S]*?<\/div>/g, (tableHtml) => {
        tablePlaceholders.push(tableHtml);
        return `%%TABLE_PLACEHOLDER_${tablePlaceholders.length - 1}%%`;
    });

    html = escapeHtml(html);

    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/^### (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^## (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^# (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^\s*[-*]\s+(.*?)$/gm, "<li>$1</li>");
    html = html.replace(/^\s*\d+\.\s+(.*?)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*?<\/li>\s*)+)/gs, "<ul>$1</ul>");
    html = html.replace(/\n\n+/g, "</p><p>");
    html = `<p>${html}</p>`;

    // Restore table placeholders
    tablePlaceholders.forEach((tableHtml, i) => {
        html = html.replace(`%%TABLE_PLACEHOLDER_${i}%%`, tableHtml);
    });

    return html;
}

/* =========================================================
   SOURCES
========================================================= */

function renderSources(sources) {
    const sourcesList = document.getElementById("sourcesList");
    const sourceCount = document.getElementById("sourceCount");
    if (!sourcesList || !sourceCount) return;

    const visibleSources = Array.isArray(sources) ? sources : [];
    sourceCount.textContent = `${visibleSources.length} source${visibleSources.length === 1 ? "" : "s"}`;

    if (!visibleSources.length) {
        sourcesList.innerHTML = '<div class="source-empty">Sources will appear after a question.</div>';
        return;
    }

    sourcesList.innerHTML = visibleSources.map(source => {
        const ct = source.content_type || "text";
        const filename = source.filename || "Unknown document";
        const page = source.page ?? "?";

        if (ct === "table") {
            // Render as HTML table with expandable details
            const tableNum = source.table_number ? `Table ${source.table_number}` : "Table";
            const headers = source.table_headers ? `<div class="source-table-headers">Columns: ${escapeHtml(source.table_headers)}</div>` : "";
            let tableHtml = "";
            if (source.table_markdown) {
                tableHtml = markdownTableToHtml(source.table_markdown);
            }
            return `
                <div class="source-item source-item-table">
                    <div class="source-header">
                        <span class="source-type-badge source-type-table">TABLE</span>
                        <div class="source-details">
                            <strong>${escapeHtml(filename)}</strong>
                            <span>Page ${escapeHtml(String(page))} · ${tableNum}</span>
                        </div>
                    </div>
                    ${headers}
                    ${tableHtml}
                </div>
            `;
        } else if (ct === "image") {
            // Render with image preview
            const imgDesc = source.image_description || "";
            const imgUrl  = source.image_url || "";
            let imgHtml = "";
            if (imgUrl) {
                imgHtml = `
                    <div class="source-image-preview">
                        <img src="${escapeHtml(imgUrl)}"
                             alt="Extracted image from ${escapeHtml(filename)} page ${escapeHtml(String(page))}"
                             class="source-image"
                             loading="lazy"
                             onerror="this.parentElement.innerHTML='<span class=source-image-error>Image could not be loaded</span>'">
                    </div>`;
            }
            const descHtml = imgDesc
                ? `<div class="source-image-desc">${escapeHtml(imgDesc.substring(0, 200))}${imgDesc.length > 200 ? "…" : ""}</div>`
                : "";
            return `
                <div class="source-item source-item-image">
                    <div class="source-header">
                        <span class="source-type-badge source-type-image">IMAGE</span>
                        <div class="source-details">
                            <strong>${escapeHtml(filename)}</strong>
                            <span>Page ${escapeHtml(String(page))}</span>
                        </div>
                    </div>
                    ${imgHtml}
                    ${descHtml}
                </div>
            `;
        } else {
            // Text / OCR source (original rendering)
            const typeLabel = ct === "ocr" ? "OCR" : "TEXT";
            const typeCls   = ct === "ocr" ? "source-type-ocr" : "source-type-text";
            const chunkInfo = source.chunk ? ` · Chunk ${escapeHtml(String(source.chunk))}` : "";
            return `
                <div class="source-item">
                    <div class="source-header">
                        <span class="source-type-badge ${typeCls}">${typeLabel}</span>
                        <div class="source-details">
                            <strong>${escapeHtml(filename)}</strong>
                            <span>Page ${escapeHtml(String(page))}${chunkInfo}</span>
                        </div>
                    </div>
                </div>
            `;
        }
    }).join("");
}

function clearEvidence() {
    currentSources = [];
    renderSources([]);
    chatArea?.querySelectorAll(".citation-summary").forEach(summary => summary.remove());
}

/* =========================================================
   WELCOME MESSAGE
========================================================= */

function createWelcomeMessage() {
    return `
        <div class="welcome-message">
            <div class="welcome-icon">
                <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/>
                    <path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/>
                </svg>
            </div>
            <h3>Your answer will appear here</h3>
            <p>Ask a question to see a grounded response.</p>
        </div>
    `;
}

async function loadDocuments() {
    try {
        const response = await fetch(`${API_BASE}/api/documents`);
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || data.message || "Unable to load documents.");
        }

        const documents = data.documents || [];
        updateStatistics(documents);
        renderDocumentsList(documents);
        renderDocuments(documents);
    } catch (error) {
        console.error("Document loading error:", error);
        if (documentsList) {
            documentsList.innerHTML = `<div class="empty-state-small">Unable to load documents</div>`;
        }
    }
}

function updateStatistics(documents) {
    if (documentCount) {
        documentCount.textContent = `${documents.length} doc${documents.length === 1 ? "" : "s"}`;
    }
}

function renderDocumentsList(documents) {
    if (!documentsList) return;
    if (!documents.length) {
        documentsList.innerHTML = '<div class="empty-state-small">No documents yet</div>';
        return;
    }

    documentsList.innerHTML = documents.map(doc => `
        <div class="document-item">
            <div class="doc-icon">PDF</div>
            <div class="doc-info">
                <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                <div class="doc-meta">${doc.pages ?? 0} pages · ${doc.chunks ?? 0} chunks</div>
            </div>
            <button class="doc-delete" data-filename="${escapeHtml(doc.filename)}" title="Delete document" aria-label="Delete ${escapeHtml(doc.filename)}">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3,6 5,6 21,6"/><path d="M19 6l-1 14H6L5 6"/></svg>
            </button>
        </div>
    `).join("");

    documentsList.querySelectorAll(".doc-delete").forEach(button => {
        button.addEventListener("click", () => deleteDocument(button.dataset.filename));
    });
}

function renderDocuments(documents) {

    if (!documents.length) {
        documentsContainer.innerHTML = `<div class="loading-state">No documents have been indexed yet.</div>`;
        return;
    }

    documentsContainer.innerHTML = `
        <div class="doc-row doc-row-header">
            <div class="doc-name"><strong>Document</strong></div>
            <div class="doc-stat"><strong>Pages</strong></div>
            <div class="doc-stat"><strong>Chunks</strong></div>
            <div class="doc-stat"><strong>Actions</strong></div>
        </div>
        ${documents.map((doc) => `
            <div class="doc-row">
                <div class="doc-name">
                    <div class="doc-icon">PDF</div>
                    <div class="doc-name-text">
                        <strong title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</strong>
                        <span>Indexed document</span>
                    </div>
                </div>
                <div class="doc-stat"><strong>${doc.pages ?? 0}</strong> pages</div>
                <div class="doc-stat"><strong>${doc.chunks ?? 0}</strong> chunks</div>
                <div style="display:flex;gap:6px;">
                    <button class="btn-view-chunks" data-filename="${escapeHtml(doc.filename)}">View Chunks</button>
                    <button class="delete-btn" data-filename="${escapeHtml(doc.filename)}">Delete</button>
                </div>
            </div>
        `).join("")}
    `;

    document.querySelectorAll(".delete-btn").forEach((btn) => {
        btn.addEventListener("click", () => deleteDocument(btn.dataset.filename));
    });

    document.querySelectorAll(".btn-view-chunks").forEach((btn) => {
        btn.addEventListener("click", () => openChunkViewer(btn.dataset.filename));
    });
}

async function deleteDocument(filename) {
    if (!window.confirm(`Delete "${filename}" from the knowledge base?`)) return;

    try {
        const response = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, {
            method: "DELETE"
        });

        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || data.message || "Unable to delete document.");
        }

        showToast("Document Deleted", `${filename} was removed from the knowledge base.`);
        await loadDocuments();
    } catch (error) {
        console.error("Delete error:", error);
        showToast("Delete Failed", error.message || "Unable to delete document.");
    }
}

/* =========================================================
   CHUNK VIEWER
========================================================= */

async function openChunkViewer(filename) {
    if (!chunkModal) return;

    chunkModalTitle.textContent = "Document Chunks";
    chunkModalSubtitle.textContent = filename;
    chunkModalBody.innerHTML = '<div class="loading-state">Loading chunks...</div>';
    chunkModal.style.display = "flex";

    try {
        const response = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}/chunks`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to load chunks.");
        }

        renderChunkViewer(data);
    } catch (error) {
        console.error("Chunk viewer error:", error);
        chunkModalBody.innerHTML = `<div class="loading-state">Error: ${escapeHtml(error.message)}</div>`;
    }
}

function closeChunkViewer() {
    if (chunkModal) {
        chunkModal.style.display = "none";
    }
}

function renderChunkViewer(data) {
    const { filename, chunk_size, chunk_overlap, total_chunks, chunks } = data;

    // Group chunks by page
    const pageGroups = {};
    chunks.forEach(chunk => {
        const page = chunk.page;
        if (!pageGroups[page]) pageGroups[page] = [];
        pageGroups[page].push(chunk);
    });

    const pageCount = Object.keys(pageGroups).length;

    let html = `
        <div class="chunk-stats">
            <div class="chunk-stat">
                <div class="chunk-stat-value">${pageCount}</div>
                <div class="chunk-stat-label">Pages</div>
            </div>
            <div class="chunk-stat">
                <div class="chunk-stat-value">${total_chunks}</div>
                <div class="chunk-stat-label">Chunks</div>
            </div>
            <div class="chunk-stat">
                <div class="chunk-stat-value">${chunk_size}</div>
                <div class="chunk-stat-label">Chunk Size</div>
            </div>
            <div class="chunk-stat">
                <div class="chunk-stat-value">${chunk_overlap}</div>
                <div class="chunk-stat-label">Overlap</div>
            </div>
        </div>
    `;

    // Render each page group
    Object.keys(pageGroups).sort((a, b) => a - b).forEach(page => {
        const pageChunks = pageGroups[page];
        const pageCount = pageChunks.length;

        html += `
            <div class="chunk-page-group">
                <div class="chunk-page-header">
                    <span class="chunk-page-badge">Page ${page}</span>
                    <span>${pageCount} chunk${pageCount !== 1 ? 's' : ''}</span>
                </div>
        `;

        pageChunks.forEach((chunk, index) => {
            let cardClass = "chunk-card";
            if (pageChunks.length === 1) {
                // Single chunk on page
            } else if (index === 0) {
                cardClass += " chunk-first";
            } else if (index === pageChunks.length - 1) {
                cardClass += " chunk-last";
            } else {
                cardClass += " chunk-middle";
            }

            const charCount = chunk.content.length;
            const isLastChunk = index === pageChunks.length - 1 && chunk.content.length < chunk_size;

            html += `
                <div class="${cardClass}">
                    <div class="chunk-card-header">
                        <span class="chunk-card-id">Chunk ${chunk.chunk_number}</span>
                        <span class="chunk-card-size">${charCount} chars${isLastChunk ? ' (partial)' : ''}</span>
                    </div>
                    <div class="chunk-card-content">${escapeHtml(chunk.content)}</div>
                </div>
            `;
        });

        html += `</div>`;
    });

    chunkModalBody.innerHTML = html;
}

/* =========================================================
   HISTORY VIEW
========================================================= */

function renderHistoryView() {
    if (!historyContainer) return;

    if (!conversationHistory.length) {
        historyContainer.innerHTML = `<div class="empty-state">No history yet. Ask a question to get started.</div>`;
        return;
    }

    historyContainer.innerHTML = conversationHistory.reduce((html, entry, i) => {
        if (entry.role === "user") {
            const nextEntry = conversationHistory[i + 1];
            const answerText = nextEntry ? nextEntry.content : "(no response)";
            const latency = nextEntry && nextEntry.latency_ms
                ? `<span class="history-entry-latency">${nextEntry.latency_ms < 1000 ? nextEntry.latency_ms + "ms" : (nextEntry.latency_ms / 1000).toFixed(1) + "s"}</span>`
                : "";
            const time = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : "";
            const sourcesCount = nextEntry && nextEntry.sources ? nextEntry.sources.length : 0;
            const sourcesNote = sourcesCount > 0
                ? `<span class="history-entry-latency" style="background:var(--purple-soft);color:var(--purple);">${sourcesCount} source${sourcesCount === 1 ? "" : "s"}</span>`
                : "";

            return html + `
                <div class="history-entry">
                    <div class="history-entry-head">
                        <div class="history-entry-time">
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                            </svg>
                            ${time}
                        </div>
                        <div style="display:flex;gap:6px;">${latency}${sourcesNote}</div>
                    </div>
                    <div class="history-q">
                        <div class="history-q-icon">Q</div>
                        <div class="history-q-text">${escapeHtml(entry.content)}</div>
                    </div>
                    <div class="history-a">${formatAnswer(answerText)}</div>
                </div>
            `;
        }
        return html;
    }, "");
}

function saveHistory() {
    try {
        localStorage.setItem("domainqa-history", JSON.stringify(conversationHistory));
    } catch (e) {
        console.warn("Could not save history to localStorage:", e);
    }
}

function loadHistoryFromStorage() {
    try {
        const data = localStorage.getItem("domainqa-history");
        if (data) {
            conversationHistory = JSON.parse(data);
        }
    } catch (e) {
        console.warn("Could not load history from localStorage:", e);
        conversationHistory = [];
    }
}

// Load on startup
loadHistoryFromStorage();

function exportHistoryAsMarkdown() {
    if (!conversationHistory.length) {
        showToast("No History", "There is no conversation history to export.");
        return;
    }

    let md = "# DomainQA Conversation History\n\n";
    md += `Exported: ${new Date().toLocaleString()}\n\n---\n\n`;

    for (let i = 0; i < conversationHistory.length; i++) {
        const entry = conversationHistory[i];
        if (entry.role === "user") {
            const answerEntry = conversationHistory[i + 1];
            md += `## Q: ${entry.content}\n\n`;
            if (answerEntry) {
                md += `**Answer:**\n\n${answerEntry.content}\n\n`;
                if (answerEntry.sources && answerEntry.sources.length) {
                    md += `**Sources:** ${answerEntry.sources.map(s => s.filename || s.file || "unknown").join(", ")}\n\n`;
                }
                if (answerEntry.latency_ms) {
                    md += `*Latency: ${answerEntry.latency_ms}ms*\n\n`;
                }
            }
            md += "---\n\n";
        }
    }

    const blob = new Blob([md], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `domainqa-history-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

async function sendFeedback(rating, question, answer, bubble) {
    try {
        const response = await fetch(`${API_BASE}/api/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, answer, rating })
        });

        if (!response.ok) throw new Error("Feedback failed");

        // Update UI
        const btns = bubble.querySelectorAll(".feedback-btn");
        btns.forEach(b => b.classList.remove("active"));
        bubble.querySelector(`.feedback-btn[data-rating="${rating}"]`).classList.add("active");
        showToast("Feedback Recorded", "Thank you for your feedback!");
    } catch (error) {
        showToast("Error", "Could not send feedback.");
    }
}


/* =========================================================
   ANALYTICS
========================================================= */

async function loadAnalytics() {
    if (!analyticsContainer) return;

    analyticsContainer.innerHTML = `<div class="loading-state">Loading analytics...</div>`;

    try {
        const response = await fetch(`${API_BASE}/api/analytics`);
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.detail || "Unable to load analytics.");
        }

        renderAnalytics(data);
    } catch (error) {
        console.error("Analytics error:", error);

        // Show local stats as fallback
        renderAnalytics({
            total_queries: conversationHistory.filter(h => h.role === "user").length,
            thumbs_up: 0,
            thumbs_down: 0,
            avg_latency_ms: 0,
            popular_documents: [],
            recent_queries: conversationHistory
                .filter(h => h.role === "user")
                .slice(-5)
                .map(h => ({ question: h.content, timestamp: h.timestamp }))
        });

        showToast("Analytics Notice", "Showing local stats. Server analytics unavailable.");
    }
}

async function resetAnalytics() {
    const confirmed = window.confirm(
        "Reset query counts, feedback, and latency metrics? Documents and conversation history will be kept."
    );
    if (!confirmed) return;

    if (resetAnalyticsButton) resetAnalyticsButton.disabled = true;
    try {
        const response = await fetch(`${API_BASE}/api/analytics/reset`, { method: "POST" });
        const data = await response.json();
        if (!response.ok) {
            throw new Error(data.detail || "Unable to reset analytics.");
        }

        await loadAnalytics();
        showToast("Analytics Reset", data.message || "Analytics metrics were reset.");
    } catch (error) {
        console.error("Analytics reset error:", error);
        showToast("Reset Failed", error.message || "Unable to reset analytics.");
    } finally {
        if (resetAnalyticsButton) resetAnalyticsButton.disabled = false;
    }
}

function renderAnalytics(data) {
    const totalQueries = data.total_queries ?? 0;
    const thumbsUp = data.thumbs_up ?? 0;
    const thumbsDown = data.thumbs_down ?? 0;
    const avgLatency = data.avg_latency_ms ?? 0;
    const popularDocs = data.popular_documents ?? [];
    const recentQueries = data.recent_queries ?? [];
    const feedbackTotal = thumbsUp + thumbsDown;
    const positiveRate = feedbackTotal ? Math.round((thumbsUp / feedbackTotal) * 100) : null;
    const latencyLabel = avgLatency > 0
        ? (avgLatency < 1000 ? `${Math.round(avgLatency)} ms` : `${(avgLatency / 1000).toFixed(1)} s`)
        : "—";
    const maxChunks = Math.max(1, ...popularDocs.map(doc => Number(doc.chunks) || 0));
    const formatTime = value => {
        const numericTime = Number(value);
        if (!Number.isFinite(numericTime) || numericTime <= 0) return "Time unavailable";
        const date = new Date(numericTime < 1e12 ? numericTime * 1000 : numericTime);
        return new Intl.DateTimeFormat(undefined, {
            month: "short",
            day: "numeric",
            hour: "numeric",
            minute: "2-digit"
        }).format(date);
    };

    analyticsContainer.innerHTML = `
        <div class="analytics-grid">
            <article class="analytics-card">
                <span class="analytics-card-label">Completed questions</span>
                <strong class="analytics-card-value">${totalQueries.toLocaleString()}</strong>
                <span class="analytics-card-sub">Across all sessions</span>
            </article>
            <article class="analytics-card">
                <span class="analytics-card-label">Average response</span>
                <strong class="analytics-card-value">${latencyLabel}</strong>
                <span class="analytics-card-sub">Generation latency</span>
            </article>
            <article class="analytics-card">
                <span class="analytics-card-label">Positive feedback</span>
                <strong class="analytics-card-value analytics-positive">${thumbsUp.toLocaleString()}</strong>
                <span class="analytics-card-sub">${positiveRate === null ? "No ratings yet" : `${positiveRate}% of submitted ratings`}</span>
            </article>
            <article class="analytics-card">
                <span class="analytics-card-label">Negative feedback</span>
                <strong class="analytics-card-value analytics-negative">${thumbsDown.toLocaleString()}</strong>
                <span class="analytics-card-sub">${feedbackTotal.toLocaleString()} ratings total</span>
            </article>
        </div>

        <div class="analytics-layout">
            <section class="analytics-panel analytics-activity" aria-labelledby="analyticsRecentTitle">
                <div class="analytics-panel-heading">
                    <div>
                        <h3 id="analyticsRecentTitle">Recent questions</h3>
                        <p>Latest completed answers</p>
                    </div>
                    <span class="analytics-count">${recentQueries.length} shown</span>
                </div>
                <div class="analytics-query-list">
                    ${recentQueries.length
                        ? recentQueries.map((query, index) => `
                            <div class="analytics-query-row">
                                <span class="analytics-query-index">${String(index + 1).padStart(2, "0")}</span>
                                <p class="analytics-query-text">${escapeHtml(query.question || query.q || query.text || "Untitled question")}</p>
                                <time class="analytics-query-time">${escapeHtml(formatTime(query.ts ?? query.timestamp))}</time>
                            </div>
                        `).join("")
                        : '<div class="analytics-empty">No completed questions yet.</div>'
                    }
                </div>
            </section>

            <div class="analytics-side-column">
                <section class="analytics-panel" aria-labelledby="analyticsDocsTitle">
                    <div class="analytics-panel-heading">
                        <div>
                            <h3 id="analyticsDocsTitle">Document index</h3>
                            <p>Indexed content by document</p>
                        </div>
                        <span class="analytics-count">${popularDocs.length} listed</span>
                    </div>
                    <div class="analytics-doc-list">
                        ${popularDocs.length
                            ? popularDocs.map(doc => {
                                const name = doc.filename || doc.name || "Unknown document";
                                const chunks = Number(doc.chunks) || 0;
                                const width = Math.round((chunks / maxChunks) * 100);
                                return `
                                    <div class="analytics-doc-row">
                                        <div class="analytics-doc-meta">
                                            <strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong>
                                            <span>${chunks.toLocaleString()} chunks</span>
                                        </div>
                                        <div class="analytics-bar-track" aria-hidden="true">
                                            <div class="analytics-bar-fill" style="width:${width}%;"></div>
                                        </div>
                                    </div>
                                `;
                            }).join("")
                            : '<div class="analytics-empty">No indexed documents.</div>'
                        }
                    </div>
                    <p class="analytics-note">Queries are counted overall; usage is not attributed to individual documents.</p>
                </section>

                <section class="analytics-panel analytics-feedback" aria-labelledby="analyticsFeedbackTitle">
                    <div class="analytics-panel-heading">
                        <div>
                            <h3 id="analyticsFeedbackTitle">Answer feedback</h3>
                            <p>Ratings submitted by users</p>
                        </div>
                        <span class="analytics-count">${feedbackTotal.toLocaleString()} total</span>
                    </div>
                    <div class="analytics-feedback-track" role="img" aria-label="${thumbsUp} positive and ${thumbsDown} negative ratings">
                        <span class="analytics-feedback-up" style="width:${positiveRate ?? 0}%;"></span>
                        <span class="analytics-feedback-down" style="width:${feedbackTotal && positiveRate !== null ? 100 - positiveRate : 0}%;"></span>
                    </div>
                    <div class="analytics-feedback-legend">
                        <span><i class="analytics-legend-dot analytics-positive-dot"></i>Positive <strong>${thumbsUp}</strong></span>
                        <span><i class="analytics-legend-dot analytics-negative-dot"></i>Negative <strong>${thumbsDown}</strong></span>
                    </div>
                </section>
            </div>
        </div>
    `;
}

/* =========================================================
   CHARACTER COUNT
========================================================= */

function updateCharacterCount() {
    if (characterCount && questionInput) {
        characterCount.textContent = questionInput.value.length;
    }
}

/* =========================================================
   NOTIFICATION
========================================================= */

function showNotification(message, type = "success") {
    if (!uploadNotification) return;

    uploadNotification.textContent = message;
    uploadNotification.className = `notification show ${type}`;

    setTimeout(() => {
        uploadNotification.classList.remove("show");
    }, 5000);
}

/* =========================================================
   TOAST
========================================================= */

function showToast(title, message) {
    if (!toast) return;

    if (toastTitle) toastTitle.textContent = title;
    if (toastMessage) toastMessage.textContent = message;

    toast.className = "toast";
    toast.classList.add("show");

    const isError = title.toLowerCase().includes("fail") || title.toLowerCase().includes("error") || title.toLowerCase().includes("offline");
    toast.classList.add(isError ? "toast-error" : "toast-success");

    setTimeout(() => {
        toast.classList.remove("show");
    }, 4000);
}

/* =========================================================
   HTML ESCAPING
========================================================= */

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}
