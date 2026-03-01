/**
 * background.js — Service Worker / Message Hub
 * ==============================================
 *
 * Handles the extension icon click:
 *   1. Receives the browserAction click event
 *   2. Sends TOGGLE_SIDEBAR to the active tab's content.js
 *
 * Also handles OPEN_URL messages from the sidebar (open article in new tab).
 */

"use strict";

// ── Icon click → toggle sidebar ────────────────────────────────
chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;

  // Make sure content.js is injected (handles cases where the page loaded
  // before the extension was installed / enabled).
  try {
    await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      files:  ["content.js"],
    });
  } catch (_) {
    // Already injected or restricted page — ignore
  }

  chrome.tabs.sendMessage(tab.id, { type: "TOGGLE_SIDEBAR" }, (response) => {
    if (chrome.runtime.lastError) {
      console.warn("[UnBlur] Could not reach content script:", chrome.runtime.lastError.message);
    }
  });
});

// ── Messages from sidebar ───────────────────────────────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "OPEN_URL" && message.url) {
    chrome.tabs.create({ url: message.url });
    sendResponse({ success: true });
  }
});
