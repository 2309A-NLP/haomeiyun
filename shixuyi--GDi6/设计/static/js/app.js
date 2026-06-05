const API_BASE = "";
const PDF_DB_NAME = "prospectus-rag";
const PDF_STORE_NAME = "saved-files";
const CURRENT_PDF_KEY = "current-upload";
const SESSION_STORAGE_KEY = "prospectus-session-id";
const SELECTED_DOCUMENT_KEY = "selected-document-id";

let currentTheme = localStorage.getItem("theme") || "dark";
let currentAttachment = null;
let sessionId = getOrCreateSessionId();
let availableDocuments = [];

document.addEventListener("DOMContentLoaded", async () => {
    initTheme();
    bindEvents();
    await restoreSavedPdf();
    await refreshDocumentOptions(getStoredSelectedDocumentId());
    checkHealth();
    autoResizeTextarea();
});

function bindEvents() {
    document.getElementById("themeToggle").addEventListener("click", toggleTheme);
    document.getElementById("benchmarkBtn").addEventListener("click", handleBenchmark);
    document.getElementById("sendBtn").addEventListener("click", handleSend);
    document.getElementById("questionInput").addEventListener("input", handleInputChange);
    document.getElementById("questionInput").addEventListener("keydown", handleKeyDown);
    document.getElementById("focusComposerBtn").addEventListener("click", focusComposer);
    document.getElementById("pdfFileInput").addEventListener("change", handlePdfSelected);
    document.getElementById("restorePdfBtn").addEventListener("click", handleRestoreSavedPdf);
    document.getElementById("clearSavedPdfBtn").addEventListener("click", handleClearSavedPdf);
    document.getElementById("documentSelect").addEventListener("change", handleDocumentSelectionChange);
    document.getElementById("modalClose").addEventListener("click", closeModal);
    document.getElementById("modal").addEventListener("click", (event) => {
        if (event.target.id === "modal") {
            closeModal();
        }
    });

    document.querySelectorAll(".quick-chip").forEach((button) => {
        button.addEventListener("click", () => {
            const input = document.getElementById("questionInput");
            input.value = button.dataset.question || "";
            handleInputChange();
            focusComposer();
        });
    });
}

function initTheme() {
    document.documentElement.setAttribute("data-theme", currentTheme === "light" ? "light" : "dark");
}

function toggleTheme() {
    currentTheme = currentTheme === "dark" ? "light" : "dark";
    localStorage.setItem("theme", currentTheme);
    initTheme();
}

function focusComposer() {
    const input = document.getElementById("questionInput");
    input.focus();
    document.getElementById("composerBox").scrollIntoView({ behavior: "smooth", block: "center" });
}

function autoResizeTextarea() {
    const textarea = document.getElementById("questionInput");
    textarea.addEventListener("input", function onInput() {
        this.style.height = "auto";
        this.style.height = `${Math.min(this.scrollHeight, 180)}px`;
    });
}

function handleInputChange() {
    const hasValue = document.getElementById("questionInput").value.trim().length > 0;
    document.getElementById("sendBtn").disabled = !hasValue;
}

function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        handleSend();
    }
}

async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        const data = await response.json();
        setStatusText(
            document.getElementById("healthStatus"),
            data.status === "healthy" ? "在线" : "异常",
            data.status === "healthy" ? "status-ok" : "status-bad"
        );
        setStatusText(
            document.getElementById("indexStatus"),
            data.index_ready ? "已就绪" : "待构建",
            data.index_ready ? "status-ok" : "status-warn"
        );
    } catch (error) {
        setStatusText(document.getElementById("healthStatus"), "无法连接", "status-bad");
        setStatusText(document.getElementById("indexStatus"), "未知", "status-bad");
        showToast("无法连接到后端服务", "error");
    }
}

function setStatusText(element, text, className) {
    element.textContent = text;
    element.className = `metric-value ${className}`;
}

async function handlePdfSelected(event) {
    const file = event.target.files?.[0];
    if (!file) {
        return;
    }
    if (!file.name.toLowerCase().endsWith(".pdf")) {
        showToast("请选择 PDF 文件", "error");
        event.target.value = "";
        return;
    }

    try {
        try {
            await savePdfToBrowser(file);
        } catch (error) {
            console.warn("savePdfToBrowser failed", error);
            showToast("已上传到后端，但浏览器本地保存失败", "warning");
        }
        await uploadPdfFile(file);
    } catch (error) {
        showToast(`PDF 上传失败：${error.message}`, "error");
    } finally {
        event.target.value = "";
    }
}

async function uploadPdfFile(file, options = {}) {
    const { silent = false, fromRestore = false } = options;
    if (!silent) {
        showToast(fromRestore ? "正在恢复并重新上传 PDF..." : "正在上传并构建索引...", "warning");
    }

    const response = await fetch(`${API_BASE}/upload-pdf`, {
        method: "POST",
        headers: {
            "Content-Type": file.type || "application/pdf",
            "X-Filename": encodeURIComponent(file.name),
        },
        body: file,
    });
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();
    currentAttachment = {
        name: data.filename,
        documentId: data.document_id,
        documentLabel: data.document_label,
        companyName: data.company_name,
        sizeBytes: data.size_bytes,
        savedLocally: true,
    };
    renderSavedPdfStatus(currentAttachment, true);
    renderSystemUploadCard(currentAttachment, data.ingest, data.message);
    await refreshDocumentOptions(data.document_id);
    checkHealth();
    if (!silent && data.message) {
        showToast(data.message, "success");
        return data;
    }

    if (!silent) {
        showToast(fromRestore ? "已恢复本地 PDF，并重新建索引" : "PDF 上传完成，已自动建索引", "success");
    }
    return data;
}

async function handleRestoreSavedPdf() {
    try {
        const saved = await getSavedPdfRecord();
        if (!saved?.blob) {
            showToast("当前没有可恢复的本地 PDF", "warning");
            syncSavedPdfControls(false);
            return;
        }

        const file = new File([saved.blob], saved.name, {
            type: saved.type || "application/pdf",
            lastModified: saved.lastModified || Date.now(),
        });
        await uploadPdfFile(file, { fromRestore: true });
    } catch (error) {
        showToast(`恢复 PDF 失败：${error.message}`, "error");
    }
}

async function handleClearSavedPdf() {
    try {
        await deleteSavedPdf();
        currentAttachment = null;
        renderSavedPdfStatus(null, false);
        showToast("已清除前端保存的 PDF", "success");
    } catch (error) {
        showToast(`清除本地 PDF 失败：${error.message}`, "error");
    }
}

function renderSystemUploadCard(fileInfo, ingest) {
    const chatStage = document.getElementById("chatStage");
    removeChatEmpty();

    const row = document.createElement("div");
    row.className = "chat-row assistant";
    row.innerHTML = `
        <div class="chat-avatar">🤖</div>
        <div class="chat-card">
            <div class="bubble">
                已接收 PDF：${escapeHtml(fileInfo.name)}，并完成索引构建。
                标签：${escapeHtml(fileInfo.documentLabel || "未设置")}；
                公司：${escapeHtml(fileInfo.companyName || "未识别")}；
                当前切块数：${ingest.chunks}，后续提问会优先基于对应文档作答。
            </div>
        </div>
    `;
    chatStage.appendChild(row);
    chatStage.scrollTop = chatStage.scrollHeight;
}

async function handleBenchmark() {
    showToast("正在跑 10 问 benchmark...", "warning");
    try {
        const response = await fetch(`${API_BASE}/benchmark`);
        const data = await response.json();

        const items = (data.results || []).map((item) => `
            <div class="benchmark-item ${item.matched ? "ok" : "bad"}">
                <strong>${item.matched ? "命中" : "未命中"} · 问题 ${item.id}</strong>
                <div style="margin-top: 6px;">${escapeHtml(item.question)}</div>
                <div style="margin-top: 6px; font-size: 12px; color: var(--muted);">
                    ${escapeHtml(item.source)}
                </div>
            </div>
        `).join("");

        showModal(
            "10 问 Benchmark 结果",
            `
                <div class="modal-stat"><span>总问题数</span><strong>${data.total}</strong></div>
                <div class="modal-stat"><span>答对数</span><strong>${data.correct}</strong></div>
                <div class="modal-stat"><span>平均延迟</span><strong>${data.avg_latency_ms.toFixed(0)} ms</strong></div>
                <div class="benchmark-list">${items}</div>
            `
        );
        showToast("Benchmark 已完成", "success");
    } catch (error) {
        showToast(`Benchmark 运行失败：${error.message}`, "error");
    }
}

async function handleSend() {
    const input = document.getElementById("questionInput");
    const question = input.value.trim();
    if (!question) {
        return;
    }

    document.getElementById("sendBtn").disabled = true;
    document.getElementById("latencyBadge").textContent = "检索中";

    renderUserMessage(question, currentAttachment);

    try {
        const response = await fetch(`${API_BASE}/ask`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question,
                session_id: sessionId,
                document_id: getSelectedDocumentId(),
                compare_plain_llm: document.getElementById("compareLLM").checked,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();
        renderAssistantMessage(data);
        renderResult(data);
        input.value = "";
        input.style.height = "auto";
        handleInputChange();
    } catch (error) {
        showToast(`提问失败：${error.message}`, "error");
        document.getElementById("latencyBadge").textContent = "请求失败";
    } finally {
        handleInputChange();
    }
}

function renderUserMessage(question, attachment) {
    removeChatEmpty();
    const chatStage = document.getElementById("chatStage");
    const row = document.createElement("div");
    row.className = "chat-row user";

    let attachmentHtml = "";
    if (attachment) {
        attachmentHtml = `
            <div class="attachment-card">
                <div class="attachment-icon">PDF</div>
                <div class="attachment-meta">
                    <strong>${escapeHtml(attachment.name)}</strong>
                    <span>PDF, ${formatFileSize(attachment.sizeBytes)}</span>
                </div>
            </div>
        `;
    }

    row.innerHTML = `
        <div class="chat-avatar">你</div>
        <div class="chat-card">
            <div class="bubble">${escapeHtml(question)}</div>
            ${attachmentHtml}
        </div>
    `;
    chatStage.appendChild(row);
    chatStage.scrollTop = chatStage.scrollHeight;
}

function renderAssistantMessage(data) {
    const chatStage = document.getElementById("chatStage");

    const row = document.createElement("div");
    row.className = "chat-row assistant";
    row.innerHTML = `
        <div class="chat-avatar">🤖</div>
        <div class="chat-card">
            <div class="bubble">
                <div><strong>【答案】</strong>${escapeHtml(data.answer || "未返回答案")}</div>
                <div style="margin-top: 10px;"><strong>【来源】</strong>${escapeHtml(data.source || "未知来源")}</div>
            </div>
            <div class="assistant-actions">
                <button type="button" data-copy="${escapeAttribute(data.answer || "")}">复制</button>
                <button type="button" data-copy="${escapeAttribute(data.pdf_answer || "")}">复制格式化结果</button>
                <button type="button" data-copy="${escapeAttribute(data.plain_llm_answer || "")}">复制纯 LLM 对比</button>
            </div>
        </div>
    `;

    row.querySelectorAll("[data-copy]").forEach((button) => {
        button.addEventListener("click", async () => {
            const text = button.getAttribute("data-copy") || "";
            if (!text) {
                showToast("当前没有可复制内容", "warning");
                return;
            }
            try {
                await navigator.clipboard.writeText(text);
                showToast("已复制到剪贴板", "success");
            } catch {
                showToast("复制失败", "error");
            }
        });
    });

    chatStage.appendChild(row);
    chatStage.scrollTop = chatStage.scrollHeight;
}

function renderResult(data) {
    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("resultStack").classList.remove("hidden");

    document.getElementById("answerText").textContent = data.answer || "未返回答案";
    document.getElementById("sourceText").textContent = data.source || "未知来源";
    document.getElementById("latencyBadge").textContent = `${data.latency_ms} ms`;

    renderAnalysis(data.analysis || {});

    const plainCard = document.getElementById("plainLlmCard");
    const plainText = document.getElementById("plainLlmText");
    if (data.plain_llm_answer) {
        plainCard.classList.remove("hidden");
        plainText.textContent = data.plain_llm_answer;
    } else {
        plainCard.classList.add("hidden");
        plainText.textContent = "";
    }
}

function renderAnalysis(analysis) {
    const intentRack = document.getElementById("intentRack");
    const keywordRack = document.getElementById("keywordRack");
    const subQueryList = document.getElementById("subQueryList");
    const entityList = document.getElementById("entityList");

    intentRack.innerHTML = `
        <span class="soft-tag">${escapeHtml(analysis.intent || "unknown")}</span>
        <span class="soft-tag">${escapeHtml(analysis.normalized_query || "")}</span>
    `;

    const keywords = analysis.keywords || [];
    keywordRack.innerHTML = keywords.length
        ? keywords.map((item) => `<span class="soft-tag">${escapeHtml(item)}</span>`).join("")
        : `<span class="soft-tag">未生成关键词</span>`;

    const subQueries = analysis.sub_queries || [];
    subQueryList.innerHTML = subQueries.length
        ? subQueries.map((item) => `<div class="stack-item">${escapeHtml(item)}</div>`).join("")
        : `<div class="stack-item muted-card">未生成子查询</div>`;

    const entities = [];
    if (analysis.entities?.company) {
        entities.push(`公司：${analysis.entities.company}`);
    }
    if (analysis.entities?.years?.length) {
        entities.push(`时间：${analysis.entities.years.join("、")}`);
    }
    if (analysis.entities?.indicators?.length) {
        entities.push(`指标：${analysis.entities.indicators.join("、")}`);
    }
    if (analysis.entities?.domains?.length) {
        entities.push(`领域：${analysis.entities.domains.join("、")}`);
    }
    entityList.innerHTML = entities.length
        ? entities.map((item) => `<div class="stack-item">${escapeHtml(item)}</div>`).join("")
        : `<div class="stack-item muted-card">未抽取到明显实体</div>`;
}

async function restoreSavedPdf() {
    try {
        const saved = await getSavedPdfRecord();
        if (!saved) {
            renderSavedPdfStatus(null, false);
            return;
        }

        currentAttachment = {
            name: saved.name,
            sizeBytes: saved.sizeBytes,
            savedLocally: true,
        };
        renderSavedPdfStatus(currentAttachment, true);
    } catch (error) {
        console.error("restoreSavedPdf failed", error);
        renderSavedPdfStatus(null, false);
    }
}

function renderSavedPdfStatus(fileInfo, hasSavedPdf) {
    const uploadStatus = document.getElementById("uploadStatus");
    if (!fileInfo) {
        uploadStatus.textContent = "未上传";
        uploadStatus.className = "metric-value";
        syncSavedPdfControls(false);
        return;
    }

    uploadStatus.textContent = hasSavedPdf ? `${fileInfo.name}（前端已保存）` : fileInfo.name;
    uploadStatus.className = "metric-value status-ok";
    syncSavedPdfControls(hasSavedPdf);
}

async function refreshDocumentOptions(preferredDocumentId = "") {
    try {
        const response = await fetch(`${API_BASE}/documents`);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        availableDocuments = await response.json();
        availableDocuments.sort((left, right) => {
            const leftScore = left.uploaded_at || 0;
            const rightScore = right.uploaded_at || 0;
            return rightScore - leftScore;
        });
        renderDocumentOptions(preferredDocumentId);
    } catch (error) {
        console.error("refreshDocumentOptions failed", error);
    }
}

function renderDocumentOptions(preferredDocumentId = "") {
    const select = document.getElementById("documentSelect");
    const currentValue = preferredDocumentId || select.value || getStoredSelectedDocumentId();
    const options = ['<option value="">自动判断</option>'];
    for (const item of availableDocuments) {
        const primary = item.company_name || item.document_label || item.filename;
        const secondary = item.filename && item.filename !== primary ? ` · ${item.filename}` : "";
        options.push(
            `<option value="${escapeAttribute(item.document_id)}">${escapeHtml(primary)}${escapeHtml(secondary)}</option>`
        );
    }
    select.innerHTML = options.join("");
    const resolvedValue = availableDocuments.some((item) => item.document_id === currentValue)
        ? currentValue
        : "";
    select.value = resolvedValue;
    persistSelectedDocumentId(resolvedValue);
}

function getSelectedDocumentId() {
    return document.getElementById("documentSelect").value || null;
}

function handleDocumentSelectionChange(event) {
    persistSelectedDocumentId(event.target.value || "");
}

function getStoredSelectedDocumentId() {
    return localStorage.getItem(SELECTED_DOCUMENT_KEY) || "";
}

function persistSelectedDocumentId(documentId) {
    if (!documentId) {
        localStorage.removeItem(SELECTED_DOCUMENT_KEY);
        return;
    }
    localStorage.setItem(SELECTED_DOCUMENT_KEY, documentId);
}

function syncSavedPdfControls(hasSavedPdf) {
    document.getElementById("restorePdfBtn").hidden = !hasSavedPdf;
    document.getElementById("clearSavedPdfBtn").hidden = !hasSavedPdf;
}

async function savePdfToBrowser(file) {
    const record = {
        id: CURRENT_PDF_KEY,
        name: file.name,
        type: file.type || "application/pdf",
        sizeBytes: file.size,
        lastModified: file.lastModified || Date.now(),
        savedAt: Date.now(),
        blob: file,
    };
    await withPdfStore("readwrite", (store) => store.put(record));
    currentAttachment = {
        name: record.name,
        sizeBytes: record.sizeBytes,
        savedLocally: true,
    };
    renderSavedPdfStatus(currentAttachment, true);
}

async function getSavedPdfRecord() {
    return withPdfStore("readonly", (store) => store.get(CURRENT_PDF_KEY));
}

async function deleteSavedPdf() {
    return withPdfStore("readwrite", (store) => store.delete(CURRENT_PDF_KEY));
}

function withPdfStore(mode, operation) {
    return new Promise((resolve, reject) => {
        if (!window.indexedDB) {
            reject(new Error("当前浏览器不支持 IndexedDB"));
            return;
        }

        const request = indexedDB.open(PDF_DB_NAME, 1);

        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(PDF_STORE_NAME)) {
                db.createObjectStore(PDF_STORE_NAME, { keyPath: "id" });
            }
        };

        request.onerror = () => {
            reject(request.error || new Error("IndexedDB 打开失败"));
        };

        request.onsuccess = () => {
            const db = request.result;
            const transaction = db.transaction(PDF_STORE_NAME, mode);
            const store = transaction.objectStore(PDF_STORE_NAME);
            const action = operation(store);

            action.onerror = () => {
                db.close();
                reject(action.error || new Error("本地 PDF 保存失败"));
            };
            action.onsuccess = () => {
                const result = action.result;
                transaction.oncomplete = () => {
                    db.close();
                    resolve(result);
                };
            };
            transaction.onerror = () => {
                db.close();
                reject(transaction.error || new Error("IndexedDB 事务失败"));
            };
        };
    });
}

function removeChatEmpty() {
    document.getElementById("chatEmpty")?.remove();
}

function showModal(title, body) {
    document.getElementById("modalTitle").textContent = title;
    document.getElementById("modalBody").innerHTML = body;
    document.getElementById("modal").classList.add("show");
}

function closeModal() {
    document.getElementById("modal").classList.remove("show");
}

function showToast(message, type = "success") {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    window.clearTimeout(showToast._timer);
    showToast._timer = window.setTimeout(() => {
        toast.classList.remove("show");
    }, 2600);
}

function getOrCreateSessionId() {
    const existing = localStorage.getItem(SESSION_STORAGE_KEY);
    if (existing) {
        return existing;
    }
    const generated = window.crypto?.randomUUID?.() || `session-${Date.now()}`;
    localStorage.setItem(SESSION_STORAGE_KEY, generated);
    return generated;
}

function formatFileSize(sizeBytes) {
    if (!sizeBytes) {
        return "0 B";
    }
    const units = ["B", "KB", "MB", "GB"];
    let size = sizeBytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }
    return `${size.toFixed(unitIndex === 0 ? 0 : 2)} ${units[unitIndex]}`;
}

function escapeHtml(text) {
    const div = document.createElement("div");
    div.textContent = text ?? "";
    return div.innerHTML;
}

function escapeAttribute(text) {
    return escapeHtml(text).replace(/"/g, "&quot;");
}
