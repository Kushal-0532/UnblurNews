/**
 * content.js — Article Extractor + Sidebar Injector
 * ===================================================
 *
 * Runs on every page at document_idle.
 * Responsibilities:
 *   1. Extract title, body, url, source from the DOM
 *   2. Listen for TOGGLE_SIDEBAR message from background.js
 *   3. Inject / remove a sidebar iframe
 *   4. Pass extracted article data to the sidebar via postMessage
 */

"use strict";

// ── State ──────────────────────────────────────────────────────
let sidebarFrame = null;
let sidebarVisible = false;

// ═══════════════════════════════════════════════════════════════
//  1. ARTICLE EXTRACTION
// ═══════════════════════════════════════════════════════════════

/**
 * Extract the article title from the page.
 * Tries: og:title meta → <h1> → document.title
 */
function extractTitle() {
  // og:title meta tag
  const og = document.querySelector('meta[property="og:title"]');
  if (og && og.content && og.content.trim()) return og.content.trim();

  // First <h1>
  const h1 = document.querySelector("h1");
  if (h1 && h1.innerText.trim()) return h1.innerText.trim();

  return document.title.trim();
}

/**
 * Site-specific body extractors.
 * Returns the main article text or null if not matched.
 */
const SITE_EXTRACTORS = {
  "nytimes.com": () => {
    const els = document.querySelectorAll("section[name='articleBody'] p");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "cnn.com": () => {
    const els = document.querySelectorAll(".article__content p, .zn-body__paragraph");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "foxnews.com": () => {
    const els = document.querySelectorAll(".article-body p");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "bbc.com": () => {
    const els = document.querySelectorAll('[data-component="text-block"] p, article p');
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "bbc.co.uk": () => SITE_EXTRACTORS["bbc.com"](),
  "theguardian.com": () => {
    const els = document.querySelectorAll(".article-body-commercial-selector p, [data-gu-name='body'] p");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "reuters.com": () => {
    const els = document.querySelectorAll(".article-body__content__17Yit p, [data-testid='paragraph'] p");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
  "apnews.com": () => {
    const els = document.querySelectorAll(".RichTextStoryBody p");
    return Array.from(els).map(p => p.innerText).join(" ");
  },
};

/**
 * Generic fallback: grab all <p> tags with > 100 chars, first 5.
 */
function extractBodyGeneric() {
  const paragraphs = Array.from(document.querySelectorAll("p"))
    .map(p => p.innerText.trim())
    .filter(text => text.length > 100);
  return paragraphs.slice(0, 5).join(" ");
}

/**
 * Extract the article body text (capped at 1500 chars).
 */
function extractBody() {
  const hostname = window.location.hostname.replace(/^www\./, "");

  for (const [domain, extractor] of Object.entries(SITE_EXTRACTORS)) {
    if (hostname.endsWith(domain)) {
      const text = extractor();
      if (text && text.trim().length > 50) {
        return text.trim().slice(0, 1500);
      }
    }
  }

  return extractBodyGeneric().slice(0, 1500);
}

/**
 * Build the full article data object.
 */
function extractArticleData() {
  return {
    title:  extractTitle(),
    body:   extractBody(),
    url:    window.location.href,
    source: window.location.hostname.replace(/^www\./, ""),
  };
}

// ═══════════════════════════════════════════════════════════════
//  2. SIDEBAR MANAGEMENT
// ═══════════════════════════════════════════════════════════════

function getSidebarUrl() {
  return chrome.runtime.getURL("sidebar/sidebar.html");
}

function injectSidebar() {
  if (sidebarFrame) return; // already injected

  // Push page content left to make room for sidebar
  document.body.style.transition = "margin-right 0.3s ease";
  document.body.style.marginRight = "420px";

  // Create iframe
  sidebarFrame = document.createElement("iframe");
  sidebarFrame.id = "unblur-sidebar";
  sidebarFrame.src = getSidebarUrl();
  sidebarFrame.style.cssText = `
    position: fixed;
    top: 0;
    right: 0;
    width: 420px;
    height: 100vh;
    border: none;
    z-index: 2147483647;
    box-shadow: -4px 0 20px rgba(0,0,0,0.2);
    background: #fff;
  `;

  document.body.appendChild(sidebarFrame);

  // Once iframe loads, send it the article data
  sidebarFrame.addEventListener("load", () => {
    const articleData = extractArticleData();
    sidebarFrame.contentWindow.postMessage(
      { type: "ARTICLE_DATA", payload: articleData },
      "*"
    );
  });

  sidebarVisible = true;
}

function removeSidebar() {
  if (!sidebarFrame) return;
  document.body.style.marginRight = "";
  sidebarFrame.remove();
  sidebarFrame = null;
  sidebarVisible = false;
}

function toggleSidebar() {
  if (sidebarVisible) {
    removeSidebar();
  } else {
    injectSidebar();
  }
}

// ═══════════════════════════════════════════════════════════════
//  3. MESSAGE LISTENER
// ═══════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "TOGGLE_SIDEBAR") {
    toggleSidebar();
    sendResponse({ success: true, visible: sidebarVisible });
  }
  // Allow sidebar to request fresh article data
  if (message.type === "GET_ARTICLE_DATA") {
    sendResponse({ success: true, data: extractArticleData() });
  }
});

// Also listen for messages from the sidebar iframe (e.g., close button)
window.addEventListener("message", (event) => {
  if (event.data && event.data.type === "CLOSE_SIDEBAR") {
    removeSidebar();
  }
});
