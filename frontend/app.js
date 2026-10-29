/* =========================================================
   LLM-RAG — CINEMATIC FRONTEND APPLICATION v3.0
   Fully preserves all existing API integration.
   Adds: intro animation, bg canvas, knowledge graph,
         retrieval graph and focused evidence interactions.
========================================================= */

/* =========================================================
   API CONFIGURATION (unchanged from original)
========================================================= */

const API_BASE_CANDIDATES = [
    "http://127.0.0.1:8010",
    "http://127.0.0.1:8011",
    "http://127.0.0.1:8000",
    "http://localhost:8010",
    "http://localhost:8011",
    "http://localhost:8000",
];

let API_BASE =
    window.__APP_CONFIG__ && window.__APP_CONFIG__.apiBase !== undefined
        ? window.__APP_CONFIG__.apiBase
        : "http://127.0.0.1:8010";

async function resolveApiBase() {
    const configured = window.__APP_CONFIG__ && window.__APP_CONFIG__.apiBase !== undefined
        ? String(window.__APP_CONFIG__.apiBase).trim()
        : "";
    if (configured) return configured;

    if (window.location.protocol !== "file:") {
        try {
            const r = await fetch("/api/health", { cache: "no-store" });
            if (r.ok) return "";
        } catch (_) {}
    }

    for (const candidate of API_BASE_CANDIDATES) {
        try {
            const r = await fetch(`${candidate}/api/health`, { cache: "no-store" });
            if (r.ok) return candidate;
        } catch (_) {}
    }

    return "http://127.0.0.1:8010";
}

/* =========================================================
   STATE
========================================================= */

let selectedPdf = null;
let currentSessionId = "session-" + Math.random().toString(36).substring(2, 11);
let conversationHistory = [];
let currentAnswerText = "";
let currentSources = [];
let currentFeedback = null;
let isStreaming = false;
let currentView = "dashboard";
let indexedDocuments = [];
let kgNodes = [];
let kgEdges = [];
let kgAnimFrame = null;
let kgHovered = null;
let kgSelectedNode = null;

function documentLocationLabel(doc) {
    const extension = (doc.filename || "").split(".").pop()?.toUpperCase() || "DOC";
    if (doc.pages != null) return `${doc.pages}p · ${doc.chunks ?? 0}c`;
    return `${extension} · ${doc.chunks ?? 0} chunks`;
}

/* =========================================================
   DOM REFS
========================================================= */

const dom = {
    fileInput:            () => document.getElementById("fileInput"),
    dropZone:             () => document.getElementById("dropZone"),
    selectedFile:         () => document.getElementById("selectedFile"),
    uploadButton:         () => document.getElementById("uploadButton"),
    uploadProgress:       () => document.getElementById("uploadProgress"),
    progressText:         () => document.getElementById("progressText"),
    uploadNotification:   () => document.getElementById("uploadNotification"),
    questionInput:        () => document.getElementById("questionInput"),
    characterCount:       () => document.getElementById("characterCount"),
    askButton:            () => document.getElementById("askButton"),
    toast:                () => document.getElementById("toast"),
    toastTitle:           () => document.getElementById("toastTitle"),
    toastMessage:         () => document.getElementById("toastMessage"),
    answerState:          () => document.getElementById("answerState"),
    chatArea:             () => document.getElementById("chatArea"),
    sourcesList:          () => document.getElementById("sourcesList"),
    sourceCount:          () => document.getElementById("sourceCount"),
    documentsList:        () => document.getElementById("documentsList"),
    documentCount:        () => document.getElementById("documentCount"),
    documentsContainer:   () => document.getElementById("documentsContainer"),
    historyContainer:     () => document.getElementById("historyContainer"),
    analyticsContainer:   () => document.getElementById("analyticsContainer"),
    connectionStatus:     () => document.getElementById("connectionStatus"),
    connectionStatusText: () => document.getElementById("connectionStatusText"),
    sidebarBackendStatus: () => document.getElementById("sidebarBackendStatus"),
    sidebarBackendStatusText: () => document.getElementById("sidebarBackendStatusText"),
    sidebarLlmStatus:     () => document.getElementById("sidebarLlmStatus"),
    assistantLlmStatus:   () => document.getElementById("assistantLlmStatus"),
    chunkModal:           () => document.getElementById("chunkModal"),
    chunkModalTitle:      () => document.getElementById("chunkModalTitle"),
    chunkModalSubtitle:   () => document.getElementById("chunkModalSubtitle"),
    chunkModalBody:       () => document.getElementById("chunkModalBody"),
    closeChunkModal:      () => document.getElementById("closeChunkModal"),
    clearChatBtn:         () => document.getElementById("clearChatBtn"),
    clearEvidence:        () => document.getElementById("clearEvidence"),
    kgCanvas:             () => document.getElementById("kgCanvas"),
    kgSection:            () => document.getElementById("kgSection"),
    kgNote:               () => document.getElementById("kgNote"),
    kgDetailPanel:        () => document.getElementById("kgDetailPanel"),
    kgDetailType:         () => document.getElementById("kgDetailType"),
    kgDetailTitle:        () => document.getElementById("kgDetailTitle"),
    kgDetailMeta:         () => document.getElementById("kgDetailMeta"),
    kgDetailExcerpt:      () => document.getElementById("kgDetailExcerpt"),
    kgDetailSourceButton: () => document.getElementById("kgDetailSourceButton"),
    kgDetailDocumentButton: () => document.getElementById("kgDetailDocumentButton"),
    showKgBtn:            () => document.getElementById("showKgBtn"),
    hideKgBtn:            () => document.getElementById("hideKgBtn"),
    bgCanvas:             () => document.getElementById("bgCanvas"),
    introScreen:          () => document.getElementById("introScreen"),
    pageBreadcrumb:       () => document.getElementById("pageBreadcrumb"),
    pageTitle:            () => document.getElementById("pageTitle"),
    mobileNavToggle:      () => document.getElementById("mobileNavToggle"),
    appSidebar:           () => document.getElementById("appSidebar"),
    sidebarOverlay:       () => document.getElementById("sidebarOverlay"),
};

/* =========================================================
   INTRO ANIMATION
========================================================= */

function runIntroAnimation() {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const intro = dom.introScreen();
    let introAlreadyShown = true;
    try {
        introAlreadyShown = window.sessionStorage.getItem("llmrag-intro-shown") === "true";
        if (!introAlreadyShown) window.sessionStorage.setItem("llmrag-intro-shown", "true");
    } catch (_) {}

    if (intro) {
        if (reduced || introAlreadyShown) {
            intro.style.display = "none";
        } else {
            intro.addEventListener("transitionend", () => {
                intro.style.display = "none";
            }, { once: true });
            window.setTimeout(() => intro.classList.add("fade-out"), 350);
            window.setTimeout(() => { intro.style.display = "none"; }, 800);
        }
    }
    startBgCanvas();
}

/* =========================================================
   BACKGROUND CANVAS
========================================================= */

function startBgCanvas() {
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const canvas = dom.bgCanvas();
    if (!canvas || reduced) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    function drawBackdrop() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let x = 28; x < canvas.width; x += 56) {
            for (let y = 28; y < canvas.height; y += 56) {
                ctx.beginPath();
                ctx.arc(x, y, 1, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(76,125,255,0.09)";
                ctx.fill();
            }
        }
    }

    drawBackdrop();
    window.addEventListener("resize", drawBackdrop, { passive: true });
}

/* =========================================================
   KNOWLEDGE GRAPH
========================================================= */

function initKnowledgeGraph() {
    const canvas = dom.kgCanvas();
    if (!canvas) return;

    canvas.addEventListener("mousemove", onKgMouseMove);
    canvas.addEventListener("click", onKgClick);
    canvas.addEventListener("mouseleave", () => {
        kgHovered = null;
        drawKnowledgeGraph();
    });

    drawKnowledgeGraph();
}

function buildKgFromSources(question, sources) {
    kgNodes = [];
    kgEdges = [];
    kgSelectedNode = null;
    dom.kgDetailPanel()?.classList.add("hidden");

    kgNodes.push({
        id: "query",
        label: question.length > 40 ? question.substring(0, 40) + "…" : question,
        type: "query",
        x: 0.5,
        y: 0.15,
        r: 14,
        color: "#FBBF24",
        metadata: { type: "Query", text: question },
    });

    const docMap = Object.create(null);
    sources.forEach((src, i) => {
        const key = src.filename || `Source ${i+1}`;
        if (!docMap[key]) {
            docMap[key] = { sources: [], filename: key, sourceIndices: [] };
        }
        docMap[key].sources.push({ ...src, sourceIndex: i });
        docMap[key].sourceIndices.push(i);
    });

    const docKeys = Object.keys(docMap);
    const total = docKeys.length;

    docKeys.forEach((key, di) => {
        const angle = (di / total) * Math.PI * 2 - Math.PI / 2;
        const radius = 0.28;
        const cx = 0.5 + Math.cos(angle) * radius;
        const cy = 0.55 + Math.sin(angle) * radius * 0.7;

        const docNode = {
            id: `doc-${di}`,
            label: key,
            type: "document",
            x: cx,
            y: cy,
            r: 12,
            color: "#4C7DFF",
            metadata: {
                filename: key,
                chunks: docMap[key].sources.length,
                sourceIndices: docMap[key].sourceIndices,
            },
        };
        kgNodes.push(docNode);
        kgEdges.push({ from: "query", to: docNode.id, color: "rgba(251,191,36,0.4)", dashed: true });

        // Source chunk nodes
        docMap[key].sources.forEach((src, si) => {
            const sAngle = angle + (si - (docMap[key].sources.length-1)/2) * 0.35;
            const sRadius = 0.14;
            const sx = cx + Math.cos(sAngle) * sRadius;
            const sy = cy + Math.sin(sAngle) * sRadius * 0.7;

            const ctColor = { text: "#4C7DFF", table: "#A78BFA", image: "#52E5C5", ocr: "#FBBF24" };
            const chunkNode = {
                id: `chunk-${di}-${si}`,
                label: `p.${src.page ?? "?"}`,
                type: "chunk",
                x: sx,
                y: sy,
                r: 7,
                color: ctColor[src.content_type] || "#4C7DFF",
                metadata: { ...src, filename: src.filename || key },
                sourceIndex: src.sourceIndex,
            };
            kgNodes.push(chunkNode);
            kgEdges.push({ from: docNode.id, to: chunkNode.id, color: "rgba(76,125,255,0.3)", dashed: false });
        });
    });

    kgNodes.forEach(n => {
        n.animAlpha = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? 1 : 0;
        n.targetAlpha = 1;
    });
    drawKnowledgeGraph();
}

function getKgCanvasPos(canvas, e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) / (rect.width || 1),
        y: (e.clientY - rect.top) / (rect.height || 1),
    };
}

function onKgMouseMove(e) {
    const canvas = dom.kgCanvas();
    if (!canvas) return;
    const pos = getKgCanvasPos(canvas, e);
    const previousHovered = kgHovered;
    kgHovered = null;
    kgNodes.forEach(n => {
        const dx = n.x - pos.x;
        const dy = n.y - pos.y;
        const radiusX = (n.r + 5) / canvas.clientWidth;
        const radiusY = (n.r + 5) / canvas.clientHeight;
        if ((dx / radiusX) ** 2 + (dy / radiusY) ** 2 < 1) {
            kgHovered = n;
        }
    });
    canvas.style.cursor = kgHovered ? "pointer" : "default";
    if (previousHovered !== kgHovered) drawKnowledgeGraph();
}

function onKgClick(e) {
    const canvas = dom.kgCanvas();
    if (!canvas) return;
    const pos = getKgCanvasPos(canvas, e);
    const node = [...kgNodes].reverse().find(n => {
        const dx = (n.x - pos.x) * canvas.clientWidth;
        const dy = (n.y - pos.y) * canvas.clientHeight;
        return Math.hypot(dx, dy) <= n.r + 5;
    });
    if (node) showKgNodeDetails(node);
}

function showKgNodeDetails(node) {
    kgSelectedNode = node;
    const panel = dom.kgDetailPanel();
    const type = dom.kgDetailType();
    const title = dom.kgDetailTitle();
    const meta = dom.kgDetailMeta();
    const excerpt = dom.kgDetailExcerpt();
    const sourceButton = dom.kgDetailSourceButton();
    const documentButton = dom.kgDetailDocumentButton();
    const data = node.metadata || {};

    if (!panel || !type || !title || !meta || !excerpt || !sourceButton || !documentButton) return;
    type.textContent = node.type === "query"
        ? "Question"
        : node.type === "document" ? "Retrieved document" : "Retrieved passage";
    title.textContent = node.type === "query" ? "Current question" : data.filename || node.label;

    if (node.type === "query") {
        meta.textContent = data.text || "";
        excerpt.textContent = "";
    } else if (node.type === "document") {
        const pages = [...new Set((data.sourceIndices || [])
            .map(index => kgNodes.find(item => item.sourceIndex === index)?.metadata.page)
            .filter(page => page !== undefined && page !== null))];
        meta.textContent = `${data.chunks} retrieved passage${data.chunks === 1 ? "" : "s"}${pages.length ? ` · Pages ${pages.join(", ")}` : ""}`;
        excerpt.textContent = "";
    } else {
        const pageText = data.page === undefined || data.page === null ? "Page unavailable" : `Page ${data.page}`;
        const chunkText = data.chunk === undefined || data.chunk === null ? "" : ` · Chunk ${data.chunk}`;
        meta.textContent = `${pageText}${chunkText} · ${(data.content_type || "text").toUpperCase()}`;
        excerpt.textContent = data.snippet || data.table_markdown || data.image_description || "No excerpt is available for this passage.";
    }

    sourceButton.hidden = !Number.isInteger(node.sourceIndex);
    documentButton.hidden = node.type === "query" || !data.filename;
    panel.classList.remove("hidden");
    drawKnowledgeGraph();
}

function focusSource(sourceIndex) {
    const sourceItem = dom.sourcesList()?.querySelector(`.source-item[data-source-index="${sourceIndex}"]`);
    if (!sourceItem) return;

    dom.kgSection()?.style && (dom.kgSection().style.display = "block");
    const showButton = dom.showKgBtn();
    if (showButton) showButton.style.display = "none";
    sourceItem.scrollIntoView({
        block: "nearest",
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth",
    });
    dom.sourcesList()?.querySelectorAll(".source-item.is-selected").forEach(item => item.classList.remove("is-selected"));
    sourceItem.classList.add("is-selected");

    const node = kgNodes.find(item => item.sourceIndex === sourceIndex);
    if (node) showKgNodeDetails(node);
    else drawKnowledgeGraph();
}

function showEvidenceInWorkspace(question, sources, sourceIndex) {
    if (!Array.isArray(sources) || !sources[sourceIndex]) return;
    navigateTo("dashboard");
    currentSources = sources;
    renderSources(sources);
    buildKgFromSources(question, sources);
    const section = dom.kgSection();
    if (section) section.style.display = "block";
    const showButton = dom.showKgBtn();
    if (showButton) showButton.style.display = "none";
    focusSource(sourceIndex);
}

function activateCitation(citation) {
    const sourceIndex = Number(citation.dataset.sourceIndex);
    const bubble = citation.closest(".chat-bubble");
    if (bubble?.sources) {
        showEvidenceInWorkspace(bubble.dataset.question || "", bubble.sources, sourceIndex);
        return;
    }

    const historyEntry = citation.closest(".history-entry");
    if (!historyEntry) return;
    const questionIndex = Number(historyEntry.dataset.historyIndex);
    const question = conversationHistory[questionIndex];
    const answer = conversationHistory[questionIndex + 1];
    if (question?.role === "user" && answer?.role === "assistant") {
        showEvidenceInWorkspace(question.content, answer.sources || [], sourceIndex);
    }
}

function drawKnowledgeGraph() {
    const canvas = dom.kgCanvas();
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const W = canvas.clientWidth || canvas.width;
    const H = canvas.clientHeight || canvas.height;
    canvas.width = W;
    canvas.height = H;

    ctx.clearRect(0, 0, W, H);

    if (!kgNodes.length) {
        ctx.fillStyle = "rgba(148,155,174,0.35)";
        ctx.font = "12px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Ask a question to visualize retrieval", W / 2, H / 2);
        return;
    }

    // Animate alpha
    kgNodes.forEach(n => {
        if (n.animAlpha < n.targetAlpha) {
            n.animAlpha = Math.min(n.animAlpha + 0.04, n.targetAlpha);
        }
    });

    // Draw edges
    kgEdges.forEach(e => {
        const from = kgNodes.find(n => n.id === e.from);
        const to   = kgNodes.find(n => n.id === e.to);
        if (!from || !to) return;

        ctx.beginPath();
        ctx.moveTo(from.x * W, from.y * H);
        ctx.lineTo(to.x * W, to.y * H);
        ctx.strokeStyle = e.color;
        ctx.lineWidth = 1.2;
        if (e.dashed) ctx.setLineDash([5, 4]);
        else ctx.setLineDash([]);
        ctx.globalAlpha = Math.min(from.animAlpha, to.animAlpha);
        ctx.stroke();
        ctx.setLineDash([]);
    });

    ctx.globalAlpha = 1;

    // Draw nodes
    kgNodes.forEach(n => {
        const x = n.x * W;
        const y = n.y * H;
        ctx.globalAlpha = n.animAlpha;

        // Glow
        if (n === kgHovered || n === kgSelectedNode || n.type === "query") {
            ctx.beginPath();
            ctx.arc(x, y, n.r + 6, 0, Math.PI * 2);
            ctx.fillStyle = n.color.replace(")", ",0.15)").replace("rgb", "rgba");
            if (n.color.startsWith("#")) {
                ctx.fillStyle = hexToRgba(n.color, 0.18);
            }
            ctx.fill();
        }

        // Node circle
        ctx.beginPath();
        ctx.arc(x, y, n.r, 0, Math.PI * 2);
        ctx.fillStyle = hexToRgba(n.color, 0.15);
        ctx.fill();
        ctx.strokeStyle = n.color;
        ctx.lineWidth = n === kgHovered || n === kgSelectedNode ? 2 : 1.5;
        ctx.stroke();

        // Label
        if (n.type !== "chunk" || n === kgHovered) {
            ctx.fillStyle = n === kgHovered ? "#F4F6FC" : n.color;
            ctx.font = `${n.type === "query" ? "600 " : ""}${n.type === "chunk" ? "9" : "10"}px 'IBM Plex Mono', monospace`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";

            const label = n.label;
            if (n.type === "chunk") {
                ctx.fillText(label, x, y);
            } else {
                ctx.fillText(label.length > 18 ? label.substring(0, 16) + "…" : label, x, y + n.r + 11);
            }
        }
    });

    ctx.globalAlpha = 1;

    // Tooltip
    if (kgHovered) {
        const n = kgHovered;
        const x = n.x * W;
        const y = n.y * H;
        const meta = n.metadata;

        let lines = [];
        if (n.type === "query") lines = ["Query", meta.text ? meta.text.substring(0, 45) : ""];
        else if (n.type === "document") lines = [meta.filename || "Document", `${meta.chunks || 0} retrieved chunk(s)`];
        else lines = [meta.filename || "?", `Page ${meta.page ?? "?"} · ${meta.content_type || "text"}`];

        const padding = 8;
        const lineH = 14;
        const w = 200;
        const h = padding * 2 + lines.length * lineH;
        let tx = Math.min(x + 16, W - w - 4);
        let ty = Math.max(y - h / 2, 4);

        ctx.fillStyle = "#171D29";
        ctx.strokeStyle = "#252C3A";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tx, ty, w, h, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#F4F6FC";
        ctx.font = "500 11px 'IBM Plex Sans', sans-serif";
        ctx.textAlign = "left";
        ctx.textBaseline = "top";
        ctx.fillText(lines[0], tx + padding, ty + padding);
        if (lines[1]) {
            ctx.fillStyle = "#939BAE";
            ctx.font = "10px 'IBM Plex Mono', monospace";
            ctx.fillText(lines[1], tx + padding, ty + padding + lineH);
        }
    }

    if (kgNodes.some(n => n.animAlpha < n.targetAlpha) && kgAnimFrame === null) {
        kgAnimFrame = requestAnimationFrame(() => {
            kgAnimFrame = null;
            drawKnowledgeGraph();
        });
    }
}

function hexToRgba(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

/* =========================================================
   NAVIGATION
========================================================= */

const PAGE_META = {
    dashboard:  { breadcrumb: "Workspace",    title: "Knowledge Assistant" },
    documents:  { breadcrumb: "Documents",    title: "Document Library" },
    history:    { breadcrumb: "History",      title: "Search History" },
    analytics:  { breadcrumb: "Evaluation",   title: "Evaluation & Analytics" },
};

function navigateTo(view) {
    currentView = view;

    const views = {
        dashboard: document.getElementById("dashboardView"),
        documents: document.getElementById("documentsView"),
        history:   document.getElementById("historyView"),
        analytics: document.getElementById("analyticsView"),
    };

    const navs = {
        dashboard: document.getElementById("navDashboard"),
        documents: document.getElementById("navDocuments"),
        history:   document.getElementById("navHistory"),
        analytics: document.getElementById("navAnalytics"),
    };

    Object.values(views).forEach(v => v && v.classList.add("hidden"));
    Object.values(navs).forEach(n => n && (n.classList.remove("active"), n.removeAttribute("aria-current")));

    const targetView = views[view];
    const targetNav = navs[view];

    if (targetView) {
        targetView.classList.remove("hidden");
        targetView.style.animation = "none";
        void targetView.offsetWidth;
        targetView.style.animation = "";
    }

    if (targetNav) {
        targetNav.classList.add("active");
        targetNav.setAttribute("aria-current", "page");
    }

    const meta = PAGE_META[view] || PAGE_META.dashboard;
    const breadEl = dom.pageBreadcrumb();
    const titleEl = dom.pageTitle();
    if (breadEl) breadEl.textContent = meta.breadcrumb;
    if (titleEl) titleEl.textContent = meta.title;

    switch (view) {
        case "documents": loadDocuments(); break;
        case "history":   renderHistoryView(); break;
        case "analytics": loadAnalytics(); break;
    }

    closeMobileSidebar();
}

/* =========================================================
   MOBILE SIDEBAR
========================================================= */

function openMobileSidebar() {
    dom.appSidebar()?.classList.add("mobile-open");
    dom.sidebarOverlay()?.classList.add("active");
    dom.mobileNavToggle()?.setAttribute("aria-expanded", "true");
}

function closeMobileSidebar() {
    dom.appSidebar()?.classList.remove("mobile-open");
    dom.sidebarOverlay()?.classList.remove("active");
    dom.mobileNavToggle()?.setAttribute("aria-expanded", "false");
}

async function checkBackendStatus() {
    const connection = dom.connectionStatus();
    const connectionText = dom.connectionStatusText();
    const sidebarStatus = dom.sidebarBackendStatus();
    const sidebarText = dom.sidebarBackendStatusText();

    try {
        const response = await fetch(`${API_BASE}/api/status`, { cache: "no-store" });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || "Backend status request failed.");

        connection?.classList.remove("is-offline");
        sidebarStatus?.classList.remove("is-offline");
        if (connectionText) connectionText.textContent = "Backend online";
        if (sidebarText) sidebarText.textContent = "Backend online";

        const llmLabel = data.llm_configured ? "Gemini configured" : "Gemini not configured";
        const sidebarLlm = dom.sidebarLlmStatus();
        const assistantLlm = dom.assistantLlmStatus();
        if (sidebarLlm) sidebarLlm.textContent = llmLabel;
        if (assistantLlm) assistantLlm.textContent = llmLabel;
    } catch (_) {
        connection?.classList.add("is-offline");
        sidebarStatus?.classList.add("is-offline");
        if (connectionText) connectionText.textContent = "Backend offline";
        if (sidebarText) sidebarText.textContent = "Backend offline";
        const sidebarLlm = dom.sidebarLlmStatus();
        const assistantLlm = dom.assistantLlmStatus();
        if (sidebarLlm) sidebarLlm.textContent = "Gemini status unavailable";
        if (assistantLlm) assistantLlm.textContent = "Gemini status unavailable";
    }
}

/* =========================================================
   FILE HANDLING
========================================================= */

const ACCEPTED_FILE_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
];
const ACCEPTED_EXTENSIONS = [".pdf",".docx",".png",".jpg",".jpeg",".webp"];

function handleFile(file) {
    const nameLower = file.name.toLowerCase();
    const typeOk = ACCEPTED_FILE_TYPES.includes(file.type);
    const extOk  = ACCEPTED_EXTENSIONS.some(ext => nameLower.endsWith(ext));
    if (!typeOk && !extOk) {
        selectedPdf = null;
        const input = dom.fileInput();
        if (input) input.value = "";
        const selectedFile = dom.selectedFile();
        if (selectedFile) selectedFile.textContent = "";
        const uploadButton = dom.uploadButton();
        if (uploadButton) uploadButton.disabled = true;
        showNotification("Please select a PDF, DOCX, or image file (PNG, JPG, WEBP).", "error");
        return;
    }
    selectedPdf = file;
    const isPdf = nameLower.endsWith(".pdf");
    const isDocx = nameLower.endsWith(".docx");
    const sf = dom.selectedFile();
    if (sf) sf.textContent = `✓ ${file.name}`;
    const ub = dom.uploadButton();
    if (ub) ub.disabled = false;
    showNotification(
        isPdf
            ? "PDF selected and ready for indexing."
            : isDocx
                ? "Word document selected and ready for indexing."
            : "Image selected. Gemini Vision or local Tesseract OCR required.",
        "success"
    );
}

/* =========================================================
   UPLOAD
========================================================= */

async function uploadPdf() {
    if (!selectedPdf) {
        showNotification("Please choose a file first.", "error");
        return;
    }

    const fileToUpload = selectedPdf;
    const formData = new FormData();
    formData.append("file", fileToUpload);

    const ub = dom.uploadButton();
    const up = dom.uploadProgress();
    if (ub) ub.disabled = true;
    if (up) up.classList.add("active");
    setProgress("Uploading and indexing…");

    try {
        const response = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: formData });
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.detail || data.message || "Upload failed.");

        if (selectedPdf === fileToUpload) {
            selectedPdf = null;
            const fi = dom.fileInput();
            const sf = dom.selectedFile();
            if (fi) fi.value = "";
            if (sf) sf.textContent = "";
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
        }

        await loadDocuments();
    } catch (err) {
        showNotification(err.message || "Unable to upload document.", "error");
        showToast("Upload Failed", err.message || "Something went wrong.");
    } finally {
        if (up) up.classList.remove("active");
        setProgress("");
        const ub2 = dom.uploadButton();
        if (ub2) ub2.disabled = !selectedPdf;
    }
}

function setProgress(text) {
    const ptxt = dom.progressText();
    if (ptxt) ptxt.textContent = text;
    const track = dom.uploadProgress()?.querySelector("[role=progressbar]");
    const bar = dom.uploadProgress()?.querySelector(".progress-bar");
    if (track) track.setAttribute("aria-busy", text ? "true" : "false");
    bar?.classList.toggle("is-indeterminate", Boolean(text));
}

/* =========================================================
   ASK QUESTION — SSE STREAMING (fully preserved)
========================================================= */

async function askQuestion() {
    const qi = dom.questionInput();
    const question = qi ? qi.value.trim() : "";
    if (!question) {
        showToast("Question Required", "Please enter a question.");
        qi?.focus();
        return;
    }

    if (isStreaming) return;
    isStreaming = true;

    const ab = dom.askButton();
    if (ab) {
        ab.disabled = true;
        ab.setAttribute("aria-busy", "true");
        const label = ab.querySelector("span");
        if (label) label.textContent = "Generating…";
    }

    const as = dom.answerState();
    if (as) {
        as.textContent = "Generating";
        as.classList.add("is-loading");
    }

    currentFeedback = null;
    currentSources = [];
    renderSources([]);
    kgNodes = [];
    kgEdges = [];
    kgSelectedNode = null;
    dom.kgDetailPanel()?.classList.add("hidden");
    const kgSection = dom.kgSection();
    if (kgSection) kgSection.style.display = "none";
    const kgButton = dom.showKgBtn();
    if (kgButton) kgButton.style.display = "none";
    drawKnowledgeGraph();

    const chatArea = dom.chatArea();
    const welcomeMsg = chatArea?.querySelector(".welcome-message");
    if (welcomeMsg) welcomeMsg.remove();

    appendChatBubble("user", question);
    if (qi) {
        qi.value = "";
        qi.style.height = "auto";
        updateCharacterCount();
    }

    currentAnswerText = "";
    let lastDoneAnswer = "";
    let lastDoneSources = [];
    let lastDoneLatency = null;
    const streamingBubble = appendChatBubble("assistant", "", null, true, question);

    const startTime = performance.now();

    try {
        const response = await fetch(`${API_BASE}/api/ask/stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                session_id: currentSessionId,
                history: conversationHistory.slice(-5).map(h => ({ role: h.role, content: h.content })),
                use_hybrid_search: true,
                use_query_rewriting: false,
                use_guard: true,
                evaluate: false,
            }),
        });

        if (!response.ok) {
            const errData = await response.json().catch(() => null);
            throw new Error(errData?.detail || errData?.message || `HTTP ${response.status}`);
        }

        const reader  = response.body.getReader();
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
                if (line.startsWith("event: error")) { errorSeen = true; continue; }
                if (line.startsWith("data: ")) {
                    const payload = line.slice(6).trim();
                    if (payload === "[DONE]") continue;
                    try {
                        const chunk = JSON.parse(payload);
                        if (chunk.detail && errorSeen) {
                            currentAnswerText = "";
                            streamError = chunk.detail;
                            const bc = streamingBubble?.querySelector(".bubble-content");
                            if (bc) bc.innerHTML = `<span class="answer-error">${escapeHtml(streamError)}</span>`;
                        } else {
                            if (chunk.token) {
                                currentAnswerText += chunk.token;
                                renderStreamingInBubble(streamingBubble, currentAnswerText, currentSources);
                            }
                            if (chunk.sources) {
                                currentSources = chunk.sources;
                                streamingBubble.sources = currentSources;
                            }
                            if (chunk.answer) {
                                lastDoneAnswer   = chunk.answer;
                                lastDoneSources  = chunk.sources || currentSources;
                                lastDoneLatency  = Number.isFinite(chunk.latency_ms) ? chunk.latency_ms : null;
                                if (!currentAnswerText) currentAnswerText = lastDoneAnswer;
                                currentSources   = lastDoneSources;
                            }
                        }
                    } catch (e) {
                        console.warn("SSE parse error:", e, payload);
                    }
                }
            }
        }

        // Flush buffer
        if (buffer.startsWith("data: ")) {
            const payload = buffer.slice(6).trim();
            if (payload && payload !== "[DONE]") {
                try {
                    const chunk = JSON.parse(payload);
                    if (chunk.sources) {
                        currentSources = chunk.sources;
                        streamingBubble.sources = currentSources;
                    }
                    if (chunk.token) {
                        currentAnswerText += chunk.token;
                        renderStreamingInBubble(streamingBubble, currentAnswerText, currentSources);
                    }
                    if (chunk.answer) {
                        lastDoneAnswer  = chunk.answer;
                        lastDoneSources = chunk.sources || currentSources;
                        lastDoneLatency = Number.isFinite(chunk.latency_ms) ? chunk.latency_ms : null;
                        if (!currentAnswerText) currentAnswerText = lastDoneAnswer;
                        currentSources  = lastDoneSources;
                    }
                } catch (_) {
                    console.warn("Could not parse the final SSE payload:", payload);
                }
            }
        }

        if (streamError) throw new Error(streamError);
        if (!lastDoneAnswer) {
            throw new Error("The response stream ended before a complete answer was received. Please retry.");
        }

        const elapsed = Math.round(performance.now() - startTime);
        if (lastDoneAnswer) {
            currentAnswerText  = lastDoneAnswer;
            currentSources     = lastDoneSources || currentSources;
        }

        const responseLatency = lastDoneLatency ?? elapsed;
        finalizeBubble(streamingBubble, currentAnswerText, currentSources, responseLatency);

        conversationHistory.push({ role: "user",      content: question,           timestamp: Date.now() });
        conversationHistory.push({ role: "assistant",  content: currentAnswerText,  sources: currentSources, latency_ms: responseLatency, timestamp: Date.now() });
        saveHistory();

        scrollChatToBottom();

        if (currentSources && currentSources.length > 0) {
            buildKgFromSources(question, currentSources);
            const kgSec = dom.kgSection();
            if (kgSec) kgSec.style.display = "block";
            const showBtn = dom.showKgBtn();
            if (showBtn) showBtn.style.display = "flex";
            window.requestAnimationFrame(drawKnowledgeGraph);
        } else {
            kgNodes = [];
            kgEdges = [];
            drawKnowledgeGraph();
        }

    } catch (err) {
        currentAnswerText = "";
        if (as) { as.textContent = "Failed"; as.classList.remove("is-loading"); }
        const content = streamingBubble?.querySelector(".bubble-content");
        if (content) content.innerHTML = `<span class="answer-error">Unable to generate answer: ${escapeHtml(err.message)}</span>`;
        showToast("Question Failed", err.message || "Something went wrong.");
    } finally {
        isStreaming = false;
        const ab2 = dom.askButton();
        if (ab2) {
            ab2.disabled = false;
            ab2.setAttribute("aria-busy", "false");
            const label = ab2.querySelector("span");
            if (label) label.textContent = "Ask Question";
        }
        if (as?.classList.contains("is-loading")) {
            as.textContent = "Ready";
            as.classList.remove("is-loading");
        }
    }
}

/* =========================================================
   CHAT BUBBLE RENDERING
========================================================= */

function appendChatBubble(role, content, sources, streaming = false, question = null) {
    const chatArea = dom.chatArea();
    if (!chatArea) return null;

    const bubble = document.createElement("div");
    bubble.className = `chat-bubble chat-bubble-${role}`;
    if (question) bubble.dataset.question = question;

    const meta = document.createElement("div");
    meta.className = "bubble-meta";
    meta.textContent = role === "user" ? "You" : "LLM-RAG";
    bubble.appendChild(meta);

    const contentEl = document.createElement("div");
    contentEl.className = "bubble-content";

    if (role === "assistant") {
        contentEl.innerHTML = streaming
            ? '<div class="thinking-indicator" role="status" aria-label="Generating answer"><span></span><span></span><span></span></div>'
            : formatAnswer(content);
    } else {
        contentEl.textContent = content;
    }

    bubble.appendChild(contentEl);
    chatArea.appendChild(bubble);
    scrollChatToBottom();
    return bubble;
}

function scrollChatToBottom() {
    const ca = dom.chatArea();
    if (ca) ca.scrollTop = ca.scrollHeight;
}

function renderStreamingInBubble(bubble, text, sources = []) {
    const contentEl = bubble?.querySelector(".bubble-content");
    if (contentEl) {
        contentEl.innerHTML = `${formatAnswer(text, sources)}<span class="streaming-cursor"></span>`;
    }
}

function finalizeBubble(bubble, text, sources, latencyMs) {
    const contentEl = bubble?.querySelector(".bubble-content");
    if (!contentEl) return;

    bubble.answerText = text;
    bubble.sources = sources || [];
    let html = formatAnswer(text, sources);

    const citedSources = new Set(
        [...text.matchAll(/\[SOURCE\s*(\d+)\]/gi)]
            .map(match => Number(match[1]))
            .filter(number => number > 0 && number <= (sources?.length || 0)),
    );
    if (citedSources.size > 0) {
        html += `<div class="citation-summary">${citedSources.size} source${citedSources.size === 1 ? "" : "s"} cited</div>`;
    } else if (sources?.length) {
        html += `<div class="citation-summary">${sources.length} retrieved passage${sources.length === 1 ? "" : "s"} available</div>`;
    }
    if (latencyMs) {
        html += `<div style="font-size:11px;color:var(--text-4);margin-top:4px;font-family:var(--font-mono);">${latencyMs < 1000 ? latencyMs + "ms" : (latencyMs / 1000).toFixed(1) + "s"}</div>`;
    }

    html += `
        <div class="feedback-actions">
            <button class="feedback-btn" data-rating="up" type="button">Helpful</button>
            <button class="feedback-btn" data-rating="down" type="button">Not helpful</button>
        </div>
    `;

    contentEl.innerHTML = html;
    renderSources(sources || []);
}

/* =========================================================
   MARKDOWN FORMATTING
========================================================= */

function markdownTableToHtml(mdTable) {
    const lines = mdTable.trim().split("\n").filter(l => l.trim());
    if (lines.length < 2) return `<pre>${escapeHtml(mdTable)}</pre>`;

    const parseRow = line =>
        line.replace(/^\|/, "").replace(/\|$/, "").split("|").map(cell => cell.trim());

    const headers   = parseRow(lines[0]);
    const bodyLines = lines.slice(2);

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

function formatAnswer(text, sources = []) {
    let html = text;

    // Extract markdown tables first
    html = html.replace(
        /((\|[^\n]+\|\n)(\|[-: |]+\|\n)((?:\|[^\n]+\|?\n?)*))/g,
        match => markdownTableToHtml(match)
    );

    // Protect table HTML
    const tablePlaceholders = [];
    html = html.replace(/<div class="table-wrapper">[\s\S]*?<\/div>/g, tableHtml => {
        tablePlaceholders.push(tableHtml);
        return `%%TABLE_${tablePlaceholders.length - 1}%%`;
    });

    html = escapeHtml(html);
    html = html.replace(/\[SOURCE\s*(\d+)\]/gi, (marker, number) => {
        const sourceIndex = Number(number) - 1;
        if (!Number.isInteger(sourceIndex) || sourceIndex < 0 || sourceIndex >= sources.length) return marker;
        return `<button class="answer-citation" type="button" data-source-index="${sourceIndex}" aria-label="View source ${number}">[${number}]</button>`;
    });
    html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/^### (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^## (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^# (.*?)$/gm, "<strong>$1</strong>");
    html = html.replace(/^\s*[-*]\s+(.*?)$/gm, "<li>$1</li>");
    html = html.replace(/^\s*\d+\.\s+(.*?)$/gm, "<li>$1</li>");
    html = html.replace(/((?:<li>.*?<\/li>\s*)+)/gs, "<ul>$1</ul>");
    html = html.replace(/\n\n+/g, "</p><p>");
    html = `<p>${html}</p>`;

    tablePlaceholders.forEach((t, i) => {
        html = html.replace(`%%TABLE_${i}%%`, t);
    });

    return html;
}

/* =========================================================
   SOURCES
========================================================= */

function renderSources(sources) {
    const sl = dom.sourcesList();
    const sc = dom.sourceCount();
    if (!sl || !sc) return;

    const visibleSources = Array.isArray(sources) ? sources : [];
    sc.textContent = `${visibleSources.length} source${visibleSources.length === 1 ? "" : "s"}`;

    if (!visibleSources.length) {
        sl.innerHTML = '<div class="source-empty">Sources will appear after a question.</div>';
        return;
    }

    sl.innerHTML = visibleSources.map((source, index) => {
        const ct       = source.content_type || "text";
        const filename = source.filename || "Unknown document";
        const page     = source.page ?? "?";
        const traceButton = `<button class="source-focus" type="button" data-source-index="${index}" aria-label="Trace source ${index + 1} in retrieval map">Trace</button>`;

        if (ct === "table") {
            const tableNum = source.table_number ? `Table ${source.table_number}` : "Table";
            const headers  = source.table_headers ? `<div class="source-table-headers">Columns: ${escapeHtml(source.table_headers)}</div>` : "";
            const tableHtml = source.table_markdown ? markdownTableToHtml(source.table_markdown) : "";
            return `
                <div class="source-item source-item-table" data-source-index="${index}">
                    <div class="source-header">
                        <span class="source-type-badge source-type-table">TABLE</span>
                        <div class="source-details">
                            <strong>${escapeHtml(filename)}</strong>
                            <span>Page ${escapeHtml(String(page))} · ${tableNum}</span>
                        </div>
                        ${traceButton}
                    </div>
                    ${headers}${tableHtml}
                </div>`;
        }

        if (ct === "image") {
            const imgDesc = source.image_description || "";
            const imgUrl  = source.image_url || "";
            const imgHtml = imgUrl ? `
                <div class="source-image-preview">
                    <img src="${escapeHtml(imgUrl)}"
                         alt="Extracted image from ${escapeHtml(filename)} page ${escapeHtml(String(page))}"
                         class="source-image" loading="lazy"
                         onerror="this.parentElement.innerHTML='<span class=source-image-error>Image could not be loaded</span>'">
                </div>` : "";
            const descHtml = imgDesc
                ? `<div class="source-image-desc">${escapeHtml(imgDesc.substring(0, 200))}${imgDesc.length > 200 ? "…" : ""}</div>` : "";
            return `
                <div class="source-item source-item-image" data-source-index="${index}">
                    <div class="source-header">
                        <span class="source-type-badge source-type-image">IMAGE</span>
                        <div class="source-details">
                            <strong>${escapeHtml(filename)}</strong>
                            <span>Page ${escapeHtml(String(page))}</span>
                        </div>
                        ${traceButton}
                    </div>
                    ${imgHtml}${descHtml}
                </div>`;
        }

        // Text / OCR
        const typeLabel = ct === "ocr" ? "OCR" : "TEXT";
        const typeCls   = ct === "ocr" ? "source-type-ocr" : "source-type-text";
        const chunkInfo = source.chunk ? ` · Chunk ${escapeHtml(String(source.chunk))}` : "";
        const snippet   = source.snippet ? `<div class="source-image-desc">${escapeHtml(source.snippet.substring(0, 200))}${source.snippet.length > 200 ? "…" : ""}</div>` : "";
        return `
            <div class="source-item" data-source-index="${index}">
                <div class="source-header">
                    <span class="source-type-badge ${typeCls}">${typeLabel}</span>
                    <div class="source-details">
                        <strong>${escapeHtml(filename)}</strong>
                        <span>Page ${escapeHtml(String(page))}${chunkInfo}</span>
                    </div>
                    ${traceButton}
                </div>
                ${snippet}
            </div>`;
    }).join("");
}

function clearEvidence() {
    currentSources = [];
    renderSources([]);
    kgNodes = [];
    kgEdges = [];
    kgSelectedNode = null;
    dom.kgDetailPanel()?.classList.add("hidden");
    const section = dom.kgSection();
    if (section) section.style.display = "none";
    const showButton = dom.showKgBtn();
    if (showButton) showButton.style.display = "none";
    drawKnowledgeGraph();
    dom.chatArea()?.querySelectorAll(".citation-summary").forEach(s => s.remove());
}

/* =========================================================
   WELCOME MESSAGE
========================================================= */

function createWelcomeMessage() {
    return `
        <div class="welcome-message">
            <div class="welcome-icon" aria-hidden="true">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6">
                    <path d="m12 3 1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8L12 3Z"/>
                    <path d="m19 16 .8 2.2L22 19l-2.2.8L19 22l-.8-2.2L16 19l2.2-.8L19 16Z"/>
                </svg>
            </div>
            <h3>Your answer will appear here</h3>
            <p>Ask a question above to see a grounded, source-cited response.</p>
        </div>`;
}

/* =========================================================
   DOCUMENTS
========================================================= */

async function loadDocuments() {
    try {
        const r = await fetch(`${API_BASE}/api/documents`);
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || data.message || "Unable to load documents.");
        const docs = data.documents || [];
        indexedDocuments = docs;
        updateStatistics(docs);
        renderDocumentsList(docs);
        filterDocuments(document.getElementById("documentSearch")?.value || "");
    } catch (err) {
        const dl = dom.documentsList();
        if (dl) dl.innerHTML = `<div class="empty-state-small">Unable to load documents</div>`;
        const container = dom.documentsContainer();
        if (container) {
            container.innerHTML = `<div class="error-state">Could not load the document library. <span>${escapeHtml(err.message || "Backend request failed.")}</span></div>`;
        }
    }
}

function updateStatistics(documents) {
    const dc = dom.documentCount();
    if (dc) dc.textContent = `${documents.length} doc${documents.length === 1 ? "" : "s"}`;
    const emptyUploadButton = document.getElementById("welcomeUploadButton");
    if (emptyUploadButton) emptyUploadButton.hidden = documents.length > 0;
}

function renderDocumentsList(documents) {
    const dl = dom.documentsList();
    if (!dl) return;
    if (!documents.length) {
        dl.innerHTML = '<div class="sidebar-empty">No documents yet</div>';
        return;
    }
    dl.innerHTML = documents.map(doc => {
        const ext = (doc.filename || "").split(".").pop()?.toUpperCase() || "DOC";
        return `
            <div class="document-item">
                <div class="doc-icon">${ext.substring(0, 3)}</div>
                <div class="doc-info">
                    <div class="doc-name" title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</div>
                    <div class="doc-meta">${documentLocationLabel(doc)}</div>
                </div>
                <button class="doc-delete" data-filename="${escapeHtml(doc.filename)}" title="Delete ${escapeHtml(doc.filename)}" aria-label="Delete ${escapeHtml(doc.filename)}">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                        <path d="M3 6h18M8 6V4h8v2m3 0-1 14H6L5 6"/>
                    </svg>
                </button>
            </div>`;
    }).join("");

    dl.querySelectorAll(".doc-delete").forEach(btn => {
        btn.addEventListener("click", e => {
            e.stopPropagation();
            deleteDocument(btn.dataset.filename);
        });
    });
}

function renderDocuments(documents = indexedDocuments) {
    const dc = dom.documentsContainer();
    if (!dc) return;

    if (!documents.length) {
        const query = document.getElementById("documentSearch")?.value.trim();
        dc.innerHTML = query
            ? `<div class="empty-state">No documents match “${escapeHtml(query)}”.</div>`
            : `<div class="empty-state">No documents have been indexed yet. Upload a PDF or supported image from Workspace to begin.</div>`;
        return;
    }

    dc.innerHTML = `
        <div class="doc-row doc-row-header">
            <div class="doc-name"><strong>Document</strong></div>
            <div class="doc-stat"><strong>Pages</strong></div>
            <div class="doc-stat"><strong>Chunks</strong></div>
            <div class="doc-stat"><strong>Actions</strong></div>
        </div>
        ${documents.map(doc => {
            const ext = (doc.filename || "").split(".").pop()?.toUpperCase() || "DOC";
            return `
                <div class="doc-row">
                    <div class="doc-name">
                        <div class="doc-icon">${ext.substring(0, 3)}</div>
                        <div class="doc-name-text">
                            <strong title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</strong>
                            <span>Indexed document${doc.tables > 0 ? ` · ${doc.tables} table${doc.tables !== 1 ? "s" : ""}` : ""}${doc.images > 0 ? ` · ${doc.images} image${doc.images !== 1 ? "s" : ""}` : ""}</span>
                        </div>
                    </div>
                    <div class="doc-stat">${doc.pages != null ? `<strong>${doc.pages}</strong> pages` : `<strong>${ext}</strong> pagination unavailable`}</div>
                    <div class="doc-stat"><strong>${doc.chunks ?? 0}</strong> chunks</div>
                    <div class="doc-actions">
                        <button class="btn-view-chunks" data-filename="${escapeHtml(doc.filename)}" aria-label="View chunks of ${escapeHtml(doc.filename)}">View Chunks</button>
                        <button class="delete-btn" data-filename="${escapeHtml(doc.filename)}" aria-label="Delete ${escapeHtml(doc.filename)}">Delete</button>
                    </div>
                </div>`;
        }).join("")}
    `;

    dc.querySelectorAll(".delete-btn").forEach(btn => {
        btn.addEventListener("click", () => deleteDocument(btn.dataset.filename));
    });
    dc.querySelectorAll(".btn-view-chunks").forEach(btn => {
        btn.addEventListener("click", () => openChunkViewer(btn.dataset.filename));
    });
}

function filterDocuments(query) {
    const normalizedQuery = query.trim().toLocaleLowerCase();
    const filtered = indexedDocuments.filter(doc =>
        (doc.filename || "").toLocaleLowerCase().includes(normalizedQuery)
    );
    renderDocuments(filtered);
}

async function deleteDocument(filename) {
    if (!window.confirm(`Delete "${filename}" from the knowledge base?`)) return;
    try {
        const r = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" });
        const data = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(data.detail || data.message || "Unable to delete.");
        showToast("Document Deleted", `${filename} was removed from the knowledge base.`);
        await loadDocuments();
    } catch (err) {
        showToast("Delete Failed", err.message || "Unable to delete document.");
    }
}

/* =========================================================
   CHUNK VIEWER
========================================================= */

async function openChunkViewer(filename) {
    const modal = dom.chunkModal();
    const title = dom.chunkModalTitle();
    const sub   = dom.chunkModalSubtitle();
    const body  = dom.chunkModalBody();
    if (!modal) return;

    if (title) title.textContent = "Document Chunks";
    if (sub)   sub.textContent   = filename;
    if (body)  body.innerHTML    = '<div class="loading-state">Loading chunks…</div>';
    modal.style.display = "flex";

    try {
        const r = await fetch(`${API_BASE}/api/documents/${encodeURIComponent(filename)}/chunks`);
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || "Unable to load chunks.");
        renderChunkViewer(data);
    } catch (err) {
        if (body) body.innerHTML = `<div class="loading-state">Error: ${escapeHtml(err.message)}</div>`;
    }
}

function closeChunkViewer() {
    const modal = dom.chunkModal();
    if (modal) modal.style.display = "none";
}

function renderChunkViewer(data) {
    const body = dom.chunkModalBody();
    if (!body) return;
    const { filename, chunk_size, chunk_overlap, total_chunks, chunks } = data;

    const pageGroups = {};
    (chunks || []).forEach(chunk => {
        const page = chunk.page;
        if (!pageGroups[page]) pageGroups[page] = [];
        pageGroups[page].push(chunk);
    });
    const hasPageNumbers = (chunks || []).some(chunk => chunk.page != null);

    let html = `
        <div class="chunk-stats">
            <div class="chunk-stat"><div class="chunk-stat-value">${hasPageNumbers ? Object.keys(pageGroups).length : "—"}</div><div class="chunk-stat-label">${hasPageNumbers ? "Pages" : "Pagination unavailable"}</div></div>
            <div class="chunk-stat"><div class="chunk-stat-value">${total_chunks}</div><div class="chunk-stat-label">Chunks</div></div>
            <div class="chunk-stat"><div class="chunk-stat-value">${chunk_size}</div><div class="chunk-stat-label">Chunk Size</div></div>
            <div class="chunk-stat"><div class="chunk-stat-value">${chunk_overlap}</div><div class="chunk-stat-label">Overlap</div></div>
        </div>`;

    Object.keys(pageGroups).sort((a, b) => {
        if (a === "null") return 1;
        if (b === "null") return -1;
        return a - b;
    }).forEach(page => {
        const pageChunks = pageGroups[page];
        const pageLabel = page === "null" ? "Pagination unavailable" : `Page ${page}`;
        html += `
            <div class="chunk-page-group">
                <div class="chunk-page-header">
                    <span class="chunk-page-badge">${pageLabel}</span>
                    <span>${pageChunks.length} chunk${pageChunks.length !== 1 ? "s" : ""}</span>
                </div>`;

        pageChunks.forEach((chunk, index) => {
            const isLastChunk = index === pageChunks.length - 1 && chunk.content.length < chunk_size;
            const imagePreview = chunk.content_type === "image" && chunk.image_filename
                ? `<div class="source-image-preview">
                       <img src="${escapeHtml(`${API_BASE}/api/images/${encodeURIComponent(chunk.image_filename)}`)}"
                            alt="Extracted image" class="source-image" loading="lazy"
                            onerror="this.parentElement.innerHTML='<span class=source-image-error>Image could not be loaded</span>'">
                   </div>${chunk.image_description ? `<div class="source-image-desc">${escapeHtml(chunk.image_description)}</div>` : ""}`
                : "";

            html += `
                <div class="chunk-card">
                    <div class="chunk-card-header">
                        <span class="chunk-card-id">Chunk ${chunk.chunk_number}</span>
                        <span class="chunk-card-size">${chunk.content.length} chars${isLastChunk ? " (partial)" : ""}</span>
                    </div>
                    ${imagePreview}
                    <div class="chunk-card-content">${escapeHtml(chunk.content)}</div>
                </div>`;
        });

        html += `</div>`;
    });

    body.innerHTML = html;
}

/* =========================================================
   HISTORY VIEW
========================================================= */

function renderHistoryView() {
    const hc = dom.historyContainer();
    if (!hc) return;

    if (!conversationHistory.length) {
        hc.innerHTML = `<div class="empty-state">No history yet. Ask a question to get started.</div>`;
        return;
    }

    hc.innerHTML = conversationHistory.reduce((html, entry, i) => {
        if (entry.role !== "user") return html;

        const nextEntry   = conversationHistory[i + 1];
        const answerText  = nextEntry ? nextEntry.content : "(no response)";
        const latency     = nextEntry?.latency_ms
            ? `<span class="history-entry-latency">${nextEntry.latency_ms < 1000 ? nextEntry.latency_ms + "ms" : (nextEntry.latency_ms / 1000).toFixed(1) + "s"}</span>`
            : "";
        const time        = entry.timestamp ? new Date(entry.timestamp).toLocaleString() : "";
        const srcCount    = nextEntry?.sources?.length ?? 0;
        const srcNote     = srcCount > 0
                ? `<span class="history-entry-latency">${srcCount} source${srcCount === 1 ? "" : "s"}</span>` : "";

        return html + `
            <div class="history-entry" data-history-index="${i}">
                <div class="history-entry-head">
                    <div class="history-entry-time">
                        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true">
                            <circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/>
                        </svg>
                        ${time}
                    </div>
                    <div style="display:flex;gap:6px;">${latency}${srcNote}</div>
                </div>
                <div class="history-q">
                    <div class="history-q-icon" aria-hidden="true">Q</div>
                    <div class="history-q-text">${escapeHtml(entry.content)}</div>
                </div>
                <div class="history-a">${formatAnswer(answerText, nextEntry?.sources || [])}</div>
            </div>`;
    }, "");
}

function saveHistory() {
    try { localStorage.setItem("llmrag-history", JSON.stringify(conversationHistory)); } catch (_) {}
}

function loadHistoryFromStorage() {
    try {
        const data = localStorage.getItem("llmrag-history");
        if (data) conversationHistory = JSON.parse(data);
    } catch (_) { conversationHistory = []; }
}

function exportHistoryAsMarkdown() {
    if (!conversationHistory.length) {
        showToast("No History", "There is no conversation history to export.");
        return;
    }
    let md = "# LLM-RAG Conversation History\n\n";
    md += `Exported: ${new Date().toLocaleString()}\n\n---\n\n`;
    for (let i = 0; i < conversationHistory.length; i++) {
        const entry = conversationHistory[i];
        if (entry.role === "user") {
            const answerEntry = conversationHistory[i + 1];
            md += `## Q: ${entry.content}\n\n`;
            if (answerEntry) {
                md += `**Answer:**\n\n${answerEntry.content}\n\n`;
                if (answerEntry.sources?.length) {
                    md += `**Sources:** ${answerEntry.sources.map(s => s.filename || "unknown").join(", ")}\n\n`;
                }
                if (answerEntry.latency_ms) md += `*Latency: ${answerEntry.latency_ms}ms*\n\n`;
            }
            md += "---\n\n";
        }
    }
    const blob = new Blob([md], { type: "text/markdown" });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url;
    a.download = `llmrag-history-${new Date().toISOString().slice(0, 10)}.md`;
    a.click();
    URL.revokeObjectURL(url);
}

/* =========================================================
   FEEDBACK
========================================================= */

async function sendFeedback(rating, question, answer, bubble) {
    try {
        const r = await fetch(`${API_BASE}/api/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question, answer, rating }),
        });
        if (!r.ok) throw new Error("Feedback failed");
        const btns = bubble?.querySelectorAll(".feedback-btn");
        btns?.forEach(b => b.classList.remove("active"));
        bubble?.querySelector(`.feedback-btn[data-rating="${rating}"]`)?.classList.add("active");
        showToast("Feedback Recorded", "Thank you for your feedback!");
    } catch (_) {
        showToast("Error", "Could not send feedback.");
    }
}

/* =========================================================
   ANALYTICS
========================================================= */

async function loadAnalytics() {
    const ac = dom.analyticsContainer();
    if (!ac) return;
    ac.innerHTML = `<div class="loading-state">Loading analytics…</div>`;
    try {
        const r = await fetch(`${API_BASE}/api/analytics`);
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || "Unable to load analytics.");
        renderAnalytics(data);
    } catch (err) {
        ac.innerHTML = `
            <div class="error-state" role="alert">
                <strong>Evaluation data unavailable</strong>
                <span>${escapeHtml(err.message || "The analytics endpoint could not be reached.")}</span>
                <button class="btn btn-secondary" id="analyticsRetry" type="button">Try again</button>
            </div>`;
    }
}

async function resetAnalytics() {
    if (!window.confirm("Reset query counts, feedback, and latency metrics? Documents and conversation history will be kept.")) return;
    const btn = document.getElementById("resetAnalytics");
    if (btn) btn.disabled = true;
    try {
        const r = await fetch(`${API_BASE}/api/analytics/reset`, { method: "POST" });
        const data = await r.json();
        if (!r.ok) throw new Error(data.detail || "Unable to reset analytics.");
        await loadAnalytics();
        showToast("Analytics Reset", data.message || "Analytics metrics were reset.");
    } catch (err) {
        showToast("Reset Failed", err.message || "Unable to reset analytics.");
    } finally {
        if (btn) btn.disabled = false;
    }
}

function renderAnalytics(data) {
    const ac = dom.analyticsContainer();
    if (!ac) return;

    const totalQueries  = data.total_queries  ?? 0;
    const thumbsUp      = data.thumbs_up      ?? 0;
    const thumbsDown    = data.thumbs_down    ?? 0;
    const avgLatency    = data.avg_latency_ms ?? 0;
    const popularDocs   = data.popular_documents ?? [];
    const recentQueries = data.recent_queries    ?? [];
    const feedbackTotal = thumbsUp + thumbsDown;
    const positiveRate  = feedbackTotal ? Math.round((thumbsUp / feedbackTotal) * 100) : null;
    const latencyLabel  = avgLatency > 0
        ? (avgLatency < 1000 ? `${Math.round(avgLatency)} ms` : `${(avgLatency / 1000).toFixed(1)} s`)
        : "—";
    const maxChunks = Math.max(1, ...popularDocs.map(doc => Number(doc.chunks) || 0));
    const formatTime = value => {
        const n = Number(value);
        if (!Number.isFinite(n) || n <= 0) return "—";
        const d = new Date(n < 1e12 ? n * 1000 : n);
        return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(d);
    };

    ac.innerHTML = `
        <p class="analytics-data-note">Metrics below come from the live analytics endpoint. Retrieval relevance, groundedness, citation accuracy, and baseline comparisons are not provided by this endpoint.</p>
        <div class="analytics-grid">
            <article class="analytics-card">
                <span class="analytics-card-label">Successful queries</span>
                <strong class="analytics-card-value">${totalQueries.toLocaleString()}</strong>
                <span class="analytics-card-sub">Across all sessions</span>
            </article>
            <article class="analytics-card">
                <span class="analytics-card-label">Mean response latency</span>
                <strong class="analytics-card-value">${latencyLabel}</strong>
                <span class="analytics-card-sub">Server-recorded completed answers</span>
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
                            </div>`).join("")
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
                                const name   = doc.filename || doc.name || "Unknown document";
                                const chunks = Number(doc.chunks) || 0;
                                const width  = Math.round((chunks / maxChunks) * 100);
                                return `
                                    <div class="analytics-doc-row">
                                        <div class="analytics-doc-meta">
                                            <strong title="${escapeHtml(name)}">${escapeHtml(name)}</strong>
                                            <span>${chunks.toLocaleString()} chunks</span>
                                        </div>
                                        <div class="analytics-bar-track" aria-hidden="true">
                                            <div class="analytics-bar-fill" style="width:${width}%;"></div>
                                        </div>
                                    </div>`;
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
                        <span><i class="analytics-legend-dot analytics-positive-dot" aria-hidden="true"></i>Positive <strong>${thumbsUp}</strong></span>
                        <span><i class="analytics-legend-dot analytics-negative-dot" aria-hidden="true"></i>Negative <strong>${thumbsDown}</strong></span>
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
    const qi = dom.questionInput();
    const cc = dom.characterCount();
    if (cc && qi) cc.textContent = qi.value.length;
}

/* =========================================================
   NOTIFICATIONS & TOAST
========================================================= */

function showNotification(message, type = "success") {
    const n = dom.uploadNotification();
    if (!n) return;
    n.textContent = message;
    n.className = `notification show ${type}`;
    setTimeout(() => n.classList.remove("show"), 5000);
}

function showToast(title, message, type) {
    const t  = dom.toast();
    const tt = dom.toastTitle();
    const tm = dom.toastMessage();
    if (!t) return;
    if (tt) tt.textContent = title;
    if (tm) tm.textContent = message;

    t.className = "toast";
    t.classList.add("show");

    const isError = !type ? (
        title.toLowerCase().includes("fail") ||
        title.toLowerCase().includes("error") ||
        title.toLowerCase().includes("offline") ||
        title.toLowerCase().includes("failed")
    ) : type === "error";

    t.classList.add(isError ? "toast-error" : "toast-success");

    // Update toast icon SVG
    const icon = t.querySelector(".toast-icon");
    if (icon) {
        icon.innerHTML = isError
            ? `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`
            : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" aria-hidden="true"><polyline points="20,6 9,17 4,12"/></svg>`;
    }

    setTimeout(() => t.classList.remove("show"), 4500);
}

/* =========================================================
   HTML ESCAPING
========================================================= */

function escapeHtml(value) {
    const div = document.createElement("div");
    div.textContent = String(value ?? "");
    return div.innerHTML;
}

/* =========================================================
   EVENT LISTENERS SETUP
========================================================= */

function setupEventListeners() {
    // File input
    const fi = dom.fileInput();
    if (fi) fi.addEventListener("change", () => { if (fi.files.length) handleFile(fi.files[0]); });

    // Drop zone
    const dz = dom.dropZone();
    if (dz) {
        dz.addEventListener("click", e => { if (e.target !== fi) fi?.click(); });
        dz.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") fi?.click(); });
        dz.addEventListener("dragover", e => { e.preventDefault(); dz.classList.add("drag-over"); });
        dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
        dz.addEventListener("drop", e => {
            e.preventDefault();
            dz.classList.remove("drag-over");
            const file = e.dataTransfer?.files?.[0];
            if (file) handleFile(file);
        });
    }

    // Upload button
    dom.uploadButton()?.addEventListener("click", uploadPdf);
    document.getElementById("welcomeUploadButton")?.addEventListener("click", () => fi?.click());

    // Ask button
    dom.askButton()?.addEventListener("click", askQuestion);
    document.getElementById("documentSearch")?.addEventListener("input", event => {
        filterDocuments(event.currentTarget.value);
    });
    dom.sourcesList()?.addEventListener("click", event => {
        const traceButton = event.target.closest(".source-focus");
        if (traceButton) focusSource(Number(traceButton.dataset.sourceIndex));
    });
    dom.chatArea()?.addEventListener("click", event => {
        const citation = event.target.closest(".answer-citation");
        if (citation) {
            activateCitation(citation);
            return;
        }
        const feedbackButton = event.target.closest(".feedback-btn");
        const bubble = feedbackButton?.closest(".chat-bubble");
        if (feedbackButton && bubble?.answerText) {
            sendFeedback(
                feedbackButton.dataset.rating,
                bubble.dataset.question || "",
                bubble.answerText.substring(0, 50),
                bubble,
            );
        }
    });
    dom.historyContainer()?.addEventListener("click", event => {
        const citation = event.target.closest(".answer-citation");
        if (citation) activateCitation(citation);
    });
    dom.kgDetailSourceButton()?.addEventListener("click", () => {
        if (Number.isInteger(kgSelectedNode?.sourceIndex)) focusSource(kgSelectedNode.sourceIndex);
    });
    dom.kgDetailDocumentButton()?.addEventListener("click", () => {
        const filename = kgSelectedNode?.metadata?.filename;
        if (filename) openChunkViewer(filename);
    });

    // Question input
    const qi = dom.questionInput();
    if (qi) {
        qi.addEventListener("input", updateCharacterCount);
        qi.addEventListener("keydown", e => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askQuestion(); }
        });
        qi.addEventListener("input", () => {
            qi.style.height = "auto";
            qi.style.height = `${Math.min(qi.scrollHeight, 180)}px`;
        });
    }

    // Suggestion chips
    document.querySelectorAll(".suggestion-chip").forEach(btn => {
        btn.addEventListener("click", () => {
            const qi = dom.questionInput();
            if (qi) { qi.value = btn.textContent.trim(); updateCharacterCount(); qi.focus(); }
        });
    });

    // Navigation
    document.getElementById("navDashboard")?.addEventListener("click", () => navigateTo("dashboard"));
    document.getElementById("navDocuments")?.addEventListener("click", () => navigateTo("documents"));
    document.getElementById("navHistory")?.addEventListener("click", () => navigateTo("history"));
    document.getElementById("navAnalytics")?.addEventListener("click", () => navigateTo("analytics"));

    // Documents page
    document.getElementById("refreshDocuments")?.addEventListener("click", loadDocuments);
    document.getElementById("refreshAnalytics")?.addEventListener("click", loadAnalytics);
    document.getElementById("resetAnalytics")?.addEventListener("click", resetAnalytics);
    dom.analyticsContainer()?.addEventListener("click", event => {
        if (event.target.closest("#analyticsRetry")) loadAnalytics();
    });

    // History
    document.getElementById("clearHistory")?.addEventListener("click", () => {
        conversationHistory = [];
        saveHistory();
        renderHistoryView();
        showToast("History Cleared", "Conversation history was cleared.");
    });
    document.getElementById("exportHistory")?.addEventListener("click", exportHistoryAsMarkdown);

    // Clear chat
    dom.clearChatBtn()?.addEventListener("click", () => {
        if (isStreaming) return;
        const ca = dom.chatArea();
        if (ca) ca.innerHTML = createWelcomeMessage();
        conversationHistory = [];
        saveHistory();
        clearEvidence();
        const state = dom.answerState();
        if (state) {
            state.textContent = "Ready";
            state.classList.remove("is-loading");
        }
    });

    // Clear evidence
    dom.clearEvidence()?.addEventListener("click", clearEvidence);

    // Chunk modal
    dom.closeChunkModal()?.addEventListener("click", closeChunkViewer);
    dom.chunkModal()?.addEventListener("click", e => {
        if (e.target === dom.chunkModal()) closeChunkViewer();
    });

    // Knowledge graph toggle
    dom.hideKgBtn()?.addEventListener("click", () => {
        const kgSec = dom.kgSection();
        if (kgSec) kgSec.style.display = "none";
        dom.showKgBtn()?.style && (dom.showKgBtn().style.display = "flex");
    });
    dom.showKgBtn()?.addEventListener("click", () => {
        const kgSec = dom.kgSection();
        if (kgSec) kgSec.style.display = "block";
        dom.showKgBtn().style.display = "none";
        window.requestAnimationFrame(drawKnowledgeGraph);
    });

    // Mobile nav
    dom.mobileNavToggle()?.addEventListener("click", () => {
        const sidebar = dom.appSidebar();
        if (sidebar?.classList.contains("mobile-open")) closeMobileSidebar();
        else openMobileSidebar();
    });
    dom.sidebarOverlay()?.addEventListener("click", closeMobileSidebar);

    // Window resize: redraw canvases
    window.addEventListener("resize", () => {
        if (kgNodes.length) drawKnowledgeGraph();
    });

    // Keyboard shortcuts
    document.addEventListener("keydown", e => {
        if (e.key === "Escape") {
            closeChunkViewer();
            closeMobileSidebar();
        }
    });
}

/* =========================================================
   STARTUP
========================================================= */

// Load history from localStorage
loadHistoryFromStorage();

// Setup DOM event listeners immediately
setupEventListeners();

// Start intro animation immediately
runIntroAnimation();

// Init KG canvas (empty state)
initKnowledgeGraph();

// Async startup
(async () => {
    API_BASE = await resolveApiBase();
    await Promise.all([checkBackendStatus(), loadDocuments()]);
})();
