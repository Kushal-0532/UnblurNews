/**
 * sidebar.js — UnBlur Sidebar Logic
 * ======================================
 *
 * Receives article data from content.js via postMessage,
 * calls the backend API, and renders the full UI.
 */

"use strict";

// ── Config (read from chrome.storage.sync, fallback to default) ─
let BACKEND_URL = "http://localhost:8000";

// Load persisted backend URL from extension storage
if (typeof chrome !== "undefined" && chrome.storage) {
  chrome.storage.sync.get(["backendUrl"], (result) => {
    if (result.backendUrl) BACKEND_URL = result.backendUrl;
  });
}

// ── Chart instance ──────────────────────────────────────────────
let biasChart = null;

// ── DOM refs ────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const loading   = $("loading-overlay");
const errorPanel = $("error-panel");
const mainContent = $("main-content");

// ═══════════════════════════════════════════════════════════════
//  UI state helpers
// ═══════════════════════════════════════════════════════════════

function showLoading()  { loading.classList.remove("hidden"); mainContent.classList.add("hidden"); errorPanel.classList.add("hidden"); }
function showError(msg) { $("error-message").textContent = msg; errorPanel.classList.remove("hidden"); loading.classList.add("hidden"); mainContent.classList.add("hidden"); }
function showMain()     { mainContent.classList.remove("hidden"); loading.classList.add("hidden"); errorPanel.classList.add("hidden"); }

// ═══════════════════════════════════════════════════════════════
//  API calls
// ═══════════════════════════════════════════════════════════════

async function callAnalyze(article) {
  const resp = await fetch(`${BACKEND_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: article.title,
      body:  article.body,
      url:   article.url,
    }),
  });
  if (!resp.ok) throw new Error(`/analyze returned ${resp.status}`);
  return resp.json();
}

async function callRelated(topic, politicalScore, sentimentScore) {
  const params = new URLSearchParams({
    topic,
    political_score: politicalScore,
    sentiment_score: sentimentScore,
  });
  const resp = await fetch(`${BACKEND_URL}/related?${params}`);
  if (!resp.ok) throw new Error(`/related returned ${resp.status}`);
  return resp.json();
}

// ═══════════════════════════════════════════════════════════════
//  Render: clickbait bar
// ═══════════════════════════════════════════════════════════════

function renderClickbait(pct) {
  const bar     = $("clickbait-bar");
  const label   = $("clickbait-pct-label");
  const verdict = $("clickbait-verdict");

  label.textContent = `${Math.round(pct)}%`;

  // Use a clip-path gradient trick: always render full gradient,
  // clip the bar to the right width
  bar.style.width           = `${pct}%`;
  bar.style.backgroundSize  = `${(100 / pct) * 100}% 100%`;

  if (pct < 40)      verdict.textContent = "Low clickbait";
  else if (pct < 70) verdict.textContent = "Moderate";
  else               verdict.textContent = "High clickbait";
}

// ═══════════════════════════════════════════════════════════════
//  Render: bias scatter chart
// ═══════════════════════════════════════════════════════════════

const LEANING_COLORS = {
  left:   "#3b82f6",
  center: "#94a3b8",
  right:  "#ef4444",
};

function leaningFromScore(score) {
  if (score < -0.3) return "left";
  if (score >  0.3) return "right";
  return "center";
}

function buildChartDatasets(currentArticle, relatedArticles) {
  // Related articles: colored circles
  const relatedData = relatedArticles.map((a) => ({
    x:     a.political_score,
    y:     a.sentiment_score,
    label: a.title,
    source: a.source,
    leaning: leaningFromScore(a.political_score),
  }));

  // Group by leaning for color coding
  const groups = { left: [], center: [], right: [] };
  relatedData.forEach((d) => groups[d.leaning].push(d));

  const datasets = Object.entries(groups)
    .filter(([, pts]) => pts.length > 0)
    .map(([leaning, pts]) => ({
      label: leaning.charAt(0).toUpperCase() + leaning.slice(1),
      data: pts.map((p) => ({ x: p.x, y: p.y, title: p.label, source: p.source })),
      backgroundColor: LEANING_COLORS[leaning] + "cc",
      borderColor:     LEANING_COLORS[leaning],
      pointRadius: 6,
      pointHoverRadius: 8,
    }));

  // Current article: star marker
  datasets.push({
    label: "This article",
    data: [{
      x: currentArticle.political_score,
      y: currentArticle.sentiment_score,
      title: "This article",
      source: "",
    }],
    backgroundColor: "#FF6B6B",
    borderColor: "#FF6B6B",
    pointRadius: 10,
    pointHoverRadius: 12,
    pointStyle: "star",
  });

  return datasets;
}

function renderChart(currentArticle, relatedArticles) {
  const canvas = $("bias-chart");
  if (!canvas) return;

  if (biasChart) {
    biasChart.destroy();
    biasChart = null;
  }

  const datasets = buildChartDatasets(currentArticle, relatedArticles);

  biasChart = new Chart(canvas, {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: {
          min: -1.1, max: 1.1,
          grid: { color: "rgba(255,255,255,0.07)" },
          ticks: { color: "#8892a4", font: { size: 9 }, maxTicksLimit: 5 },
          border: { color: "rgba(255,255,255,0.15)" },
          title: {
            display: true,
            text: "← Left  |  Right →",
            color: "#8892a4",
            font: { size: 10 },
            padding: { top: 4 },
          },
        },
        y: {
          min: -1.1, max: 1.1,
          grid: { color: "rgba(255,255,255,0.07)" },
          ticks: { color: "#8892a4", font: { size: 9 }, maxTicksLimit: 5 },
          border: { color: "rgba(255,255,255,0.15)" },
          title: {
            display: true,
            text: "↓ Negative  |  Positive ↑",
            color: "#8892a4",
            font: { size: 10 },
            padding: { bottom: 4 },
          },
        },
      },
      plugins: {
        legend: {
          labels: { color: "#8892a4", font: { size: 10 }, boxWidth: 8 },
        },
        tooltip: {
          callbacks: {
            label(ctx) {
              const pt = ctx.raw;
              const src = pt.source ? ` [${pt.source}]` : "";
              const t = pt.title ? pt.title.slice(0, 55) : "";
              return `${t}${src}`;
            },
          },
          backgroundColor: "#1a1d26",
          titleColor: "#e2e8f0",
          bodyColor: "#8892a4",
          borderColor: "#2a2d3a",
          borderWidth: 1,
        },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════════
//  Render: case diagnosis
// ═══════════════════════════════════════════════════════════════

const CASE_CONFIG = {
  echo_chamber: {
    icon: `<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="10" stroke="#3b82f6" stroke-width="2"/>
      <circle cx="14" cy="14" r="6"  fill="#3b82f6" opacity=".3"/>
      <circle cx="10" cy="14" r="2"  fill="#3b82f6"/>
      <circle cx="14" cy="14" r="2"  fill="#3b82f6"/>
      <circle cx="18" cy="14" r="2"  fill="#3b82f6"/>
    </svg>`,
    title: "Echo Chamber",
    desc:  "Most coverage shares the same political leaning and emotional tone as this article.",
  },
  contradiction: {
    icon: `<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path d="M5 14 L12 7 L12 11 L23 11" stroke="#ef4444" stroke-width="2" stroke-linecap="round" fill="none"/>
      <path d="M23 14 L16 21 L16 17 L5 17"  stroke="#3b82f6" stroke-width="2" stroke-linecap="round" fill="none"/>
    </svg>`,
    title: "Contradiction",
    desc:  "Coverage is highly polarized — strong opposing perspectives exist on both sides.",
  },
  internal_split: {
    icon: `<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path d="M14 6 L14 14" stroke="#eab308" stroke-width="2" stroke-linecap="round"/>
      <path d="M14 14 L8 22"  stroke="#eab308" stroke-width="2" stroke-linecap="round"/>
      <path d="M14 14 L20 22" stroke="#eab308" stroke-width="2" stroke-linecap="round"/>
    </svg>`,
    title: "Internal Split",
    desc:  "The same political side has significantly different emotional takes on this story.",
  },
  balanced: {
    icon: `<svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <path d="M14 5 L14 8" stroke="#22c55e" stroke-width="2" stroke-linecap="round"/>
      <path d="M8 14 L20 14" stroke="#22c55e" stroke-width="2" stroke-linecap="round"/>
      <path d="M8 14 L5 20 M20 14 L23 20" stroke="#22c55e" stroke-width="2" stroke-linecap="round"/>
      <path d="M4 20 L12 20 M16 20 L24 20" stroke="#22c55e" stroke-width="2.5" stroke-linecap="round"/>
    </svg>`,
    title: "Balanced",
    desc:  "Coverage appears relatively even across different political perspectives.",
  },
};

function renderCase(caseLabel) {
  const cfg = CASE_CONFIG[caseLabel] || CASE_CONFIG["balanced"];
  $("case-icon").innerHTML = cfg.icon;
  $("case-title").textContent = cfg.title;
  $("case-desc").textContent  = cfg.desc;
}

// ═══════════════════════════════════════════════════════════════
//  Render: summary + opposing links
// ═══════════════════════════════════════════════════════════════

function renderSummary(relatedData, analysisData) {
  const lean = relatedData.dominant_leaning || "center";
  const pct  = relatedData.dominant_pct || 0;
  const leanLabel = lean.toUpperCase();
  $("summary-stat").textContent =
    `${pct}% of coverage leans ${leanLabel}.`;
  $("summary-body").textContent =
    relatedData.summary || "No summary available.";

  // Opposing articles = most different (already sorted by distance)
  const opposing = relatedData.articles
    .filter((a) => leaningFromScore(a.political_score) !== leaningFromScore(analysisData.political_score))
    .slice(0, 3);

  const container = $("opposing-links");
  container.innerHTML = "";

  if (opposing.length === 0) {
    container.innerHTML = '<p style="font-size:11px;color:#8892a4">No opposing articles found.</p>';
    return;
  }

  opposing.forEach((art) => {
    const btn = document.createElement("button");
    btn.className = "opposing-link-btn";
    const leaning = leaningFromScore(art.political_score);
    const dotClass = `dot-${leaning}`;
    btn.innerHTML = `
      <span class="link-source ${dotClass}">${art.source || leaning}</span>
      <span class="link-title">${art.title}</span>
      <span class="link-arrow">↗</span>
    `;
    btn.addEventListener("click", () => {
      // Ask background to open URL in new tab
      if (typeof chrome !== "undefined" && chrome.runtime) {
        chrome.runtime.sendMessage({ type: "OPEN_URL", url: art.url });
      } else {
        window.open(art.url, "_blank");
      }
    });
    container.appendChild(btn);
  });
}

// ═══════════════════════════════════════════════════════════════
//  Main flow
// ═══════════════════════════════════════════════════════════════

let currentArticle = null;
let lastAnalysis   = null;

async function runAnalysis(article) {
  showLoading();
  currentArticle = article;

  try {
    // Step 1: Analyze current article
    const analysis = await callAnalyze(article);
    lastAnalysis = analysis;

    // Step 2: Render clickbait immediately (fast feedback)
    renderClickbait(analysis.clickbait_pct);
    renderCase(analysis.case);

    // Step 3: Fetch related articles
    const keywords = article.title
      .replace(/[^a-zA-Z ]/g, " ")
      .split(" ")
      .filter((w) => w.length > 3)
      .slice(0, 4)
      .join(" ");

    const related = await callRelated(
      keywords || article.title,
      analysis.political_score,
      analysis.sentiment_score
    );

    // Step 4: Render chart with real data
    renderChart(analysis, related.articles || []);

    // Step 5: Update case with richer related-context case
    // (backend /related returns the case from case_logic.determine_case)
    renderCase(related.case || analysis.case);

    // Step 6: Render summary
    renderSummary(related, analysis);

    showMain();
  } catch (err) {
    console.error("[UnBlur]", err);
    showError(`Failed to analyze: ${err.message}. Is the UnBlur backend running at ${BACKEND_URL}?`);
  }
}

// ── Close button ───────────────────────────────────────────────
$("btn-close").addEventListener("click", () => {
  window.parent.postMessage({ type: "CLOSE_SIDEBAR" }, "*");
});

// ── Retry button ───────────────────────────────────────────────
$("btn-retry").addEventListener("click", () => {
  if (currentArticle) runAnalysis(currentArticle);
});

// ── Receive article data from content.js ──────────────────────
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "ARTICLE_DATA") {
    runAnalysis(event.data.payload);
  }
});

// ── If opened as standalone page, show loading and wait ───────
showLoading();
