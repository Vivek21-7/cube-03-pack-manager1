"""Pack Manager Web Application: Interactive Logistics Inspector Dashboard & REST API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import logging
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from PIL import Image

from pack_manager.agent import PackManagerAIAgent
from pack_manager.fixture_scenarios import generate_all_scenarios
from pack_manager.vlm_client import get_vlm_client

logger = logging.getLogger("pack_manager.server")

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = REPO_ROOT / "fixtures" / "scenarios"


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Pack Manager AI — Autonomous Logistics Inspector</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card-bg: rgba(18, 24, 38, 0.75);
      --card-border: rgba(255, 255, 255, 0.08);
      --card-hover: rgba(255, 255, 255, 0.12);
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
      --accent-glow: rgba(59, 130, 246, 0.25);
      --seal-green: #10b981;
      --seal-glow: rgba(16, 185, 129, 0.25);
      --stop-red: #ef4444;
      --stop-glow: rgba(239, 68, 68, 0.25);
      --warn-amber: #f59e0b;
      --warn-glow: rgba(245, 158, 11, 0.25);
      --radius: 14px;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: radial-gradient(circle at 15% 15%, #131b2e 0%, var(--bg) 60%);
      color: var(--text);
      min-height: 100vh;
      line-height: 1.5;
      padding-bottom: 60px;
    }
    header {
      padding: 18px 36px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--card-border);
      background: rgba(9, 13, 22, 0.85);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 38px;
      height: 38px;
      background: linear-gradient(135deg, #2563eb, #7c3aed);
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      color: white;
      font-size: 19px;
      box-shadow: 0 4px 14px var(--accent-glow);
    }
    .brand-title {
      font-weight: 800;
      font-size: 1.15rem;
      letter-spacing: -0.02em;
    }
    .brand-badge {
      background: rgba(59, 130, 246, 0.15);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 3px 8px;
      border-radius: 6px;
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .btn-settings {
      background: rgba(255, 255, 255, 0.06);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.2s;
    }
    .btn-settings:hover {
      background: rgba(255, 255, 255, 0.12);
    }
    main {
      max-width: 1440px;
      margin: 28px auto;
      padding: 0 28px;
    }
    .scenarios-bar {
      margin-bottom: 24px;
    }
    .scenarios-label {
      font-size: 0.78rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.06em;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .scenarios-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .scenario-pill {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 8px 14px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .scenario-pill:hover {
      background: var(--card-hover);
      transform: translateY(-1px);
    }
    .scenario-pill.active {
      background: #2563eb;
      border-color: #3b82f6;
      box-shadow: 0 2px 10px var(--accent-glow);
    }
    .grid-layout {
      display: grid;
      grid-template-columns: 1fr 1.15fr;
      gap: 24px;
    }
    @media (max-width: 1024px) {
      .grid-layout { grid-template-columns: 1fr; }
    }
    .panel {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      backdrop-filter: blur(16px);
      padding: 24px;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    }
    .panel-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      padding-bottom: 12px;
      border-bottom: 1px solid var(--card-border);
    }
    .panel-title {
      font-size: 1rem;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .img-preview-container {
      position: relative;
      width: 100%;
      height: 320px;
      background: #060911;
      border: 2px dashed rgba(255, 255, 255, 0.15);
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 16px;
    }
    .img-preview-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .scan-line {
      position: absolute;
      top: 0;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, #3b82f6, #60a5fa, transparent);
      box-shadow: 0 0 12px #3b82f6;
      display: none;
    }
    .scanning .scan-line {
      display: block;
      animation: scanAnim 1.8s infinite ease-in-out;
    }
    @keyframes scanAnim {
      0% { top: 0%; opacity: 0.8; }
      50% { top: 96%; opacity: 1; }
      100% { top: 0%; opacity: 0.8; }
    }
    .photo-actions {
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }
    .file-input-label {
      flex: 1;
      text-align: center;
      padding: 9px;
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.82rem;
      font-weight: 600;
      transition: background 0.2s;
    }
    .file-input-label:hover { background: rgba(255, 255, 255, 0.1); }
    .order-editor-container {
      margin-bottom: 20px;
    }
    .order-editor-label {
      font-size: 0.78rem;
      font-weight: 700;
      color: var(--text-muted);
      margin-bottom: 6px;
      display: flex;
      justify-content: space-between;
    }
    textarea.order-editor {
      width: 100%;
      height: 150px;
      background: #060911;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.82rem;
      padding: 12px;
      resize: vertical;
      outline: none;
    }
    textarea.order-editor:focus { border-color: var(--accent); }
    .btn-verify {
      width: 100%;
      padding: 14px;
      background: linear-gradient(135deg, #2563eb, #1d4ed8);
      border: none;
      border-radius: 10px;
      color: white;
      font-size: 0.95rem;
      font-weight: 700;
      cursor: pointer;
      box-shadow: 0 4px 18px var(--accent-glow);
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      transition: all 0.2s ease;
    }
    .btn-verify:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 24px var(--accent-glow);
      background: linear-gradient(135deg, #3b82f6, #2563eb);
    }
    .btn-verify:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }
    /* Verdict Card */
    .verdict-banner {
      padding: 20px;
      border-radius: 12px;
      margin-bottom: 20px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid var(--card-border);
      transition: all 0.3s;
    }
    .verdict-banner.SEAL {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(6, 78, 59, 0.25));
      border: 1px solid rgba(16, 185, 129, 0.4);
      box-shadow: 0 4px 20px var(--seal-glow);
    }
    .verdict-banner.STOP_FIX {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(127, 29, 29, 0.25));
      border: 1px solid rgba(239, 68, 68, 0.4);
      box-shadow: 0 4px 20px var(--stop-glow);
    }
    .verdict-banner.UNCERTAIN {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.15), rgba(120, 53, 15, 0.25));
      border: 1px solid rgba(245, 158, 11, 0.4);
      box-shadow: 0 4px 20px var(--warn-glow);
    }
    .verdict-text-group h3 {
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.02em;
    }
    .verdict-banner.SEAL h3 { color: var(--seal-green); }
    .verdict-banner.STOP_FIX h3 { color: var(--stop-red); }
    .verdict-banner.UNCERTAIN h3 { color: var(--warn-amber); }
    .verdict-reason {
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .confidence-meter {
      text-align: right;
    }
    .confidence-tag {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .conf-high { background: rgba(16, 185, 129, 0.2); color: #34d399; }
    .conf-medium { background: rgba(245, 158, 11, 0.2); color: #fbbf24; }
    .conf-low { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    /* Tables and Cards */
    .section-title {
      font-size: 0.82rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      margin: 16px 0 8px 0;
      letter-spacing: 0.05em;
    }
    .match-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.84rem;
      margin-bottom: 14px;
    }
    .match-table th, .match-table td {
      padding: 10px 12px;
      text-align: left;
      border-bottom: 1px solid var(--card-border);
    }
    .match-table th {
      color: var(--text-muted);
      font-weight: 600;
      font-size: 0.76rem;
      text-transform: uppercase;
    }
    .badge-pass {
      background: rgba(16, 185, 129, 0.15);
      color: #34d399;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 0.72rem;
    }
    .badge-fail {
      background: rgba(239, 68, 68, 0.15);
      color: #f87171;
      padding: 3px 8px;
      border-radius: 4px;
      font-weight: 700;
      font-size: 0.72rem;
    }
    .alert-card {
      padding: 12px 16px;
      border-radius: 8px;
      font-size: 0.84rem;
      margin-bottom: 10px;
      display: flex;
      gap: 10px;
      align-items: flex-start;
    }
    .alert-missing {
      background: rgba(239, 68, 68, 0.1);
      border-left: 4px solid var(--stop-red);
      color: #fca5a5;
    }
    .alert-extra {
      background: rgba(245, 158, 11, 0.1);
      border-left: 4px solid var(--warn-amber);
      color: #fcd34d;
    }
    .alert-damage {
      background: rgba(220, 38, 38, 0.15);
      border-left: 4px solid #b91c1c;
      color: #fca5a5;
    }
    .evidence-list {
      list-style: none;
      font-size: 0.84rem;
      color: #cbd5e1;
    }
    .evidence-list li {
      padding: 6px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .evidence-list li::before {
      content: "•";
      color: #60a5fa;
      font-weight: bold;
    }
    .json-box {
      background: #060911;
      padding: 14px;
      border-radius: 8px;
      border: 1px solid var(--card-border);
      max-height: 220px;
      overflow-y: auto;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.76rem;
      color: #94a3b8;
      white-space: pre-wrap;
    }
    /* Settings Modal */
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.7);
      backdrop-filter: blur(8px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }
    .modal.open { display: flex; }
    .modal-content {
      background: #111827;
      border: 1px solid var(--card-border);
      border-radius: 16px;
      width: 90%;
      max-width: 480px;
      padding: 24px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.5);
    }
    .modal-title { font-size: 1.15rem; font-weight: 700; margin-bottom: 16px; }
    .form-group { margin-bottom: 16px; }
    .form-label { display: block; font-size: 0.8rem; font-weight: 600; margin-bottom: 6px; color: var(--text-muted); }
    .form-input, .form-select {
      width: 100%;
      padding: 10px;
      background: #090d16;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      color: white;
      font-size: 0.85rem;
    }
    .modal-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 20px; }
    .btn-secondary {
      padding: 8px 14px;
      background: transparent;
      border: 1px solid var(--card-border);
      color: var(--text);
      border-radius: 8px;
      cursor: pointer;
    }
    .btn-primary {
      padding: 8px 16px;
      background: #2563eb;
      border: none;
      color: white;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-icon">P</div>
      <div>
        <div style="display: flex; align-items: center; gap: 8px;">
          <span class="brand-title">Pack Manager AI</span>
          <span class="brand-badge">Logistics Inspector</span>
        </div>
        <div style="font-size: 0.75rem; color: var(--text-muted);">CUBE Outbound Verification Station</div>
      </div>
    </div>
    <div class="header-actions">
      <div id="active-provider-pill" class="scenario-pill" style="font-size: 0.75rem; padding: 5px 12px;">
        Backend: <strong id="provider-label">Offline Simulation</strong>
      </div>
      <button class="btn-settings" onclick="openSettingsModal()">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        VLM Config
      </button>
    </div>
  </header>

  <main>
    <div class="scenarios-bar">
      <div class="scenarios-label">Preset Evaluation Scenarios (From AI Prompt Spec)</div>
      <div class="scenarios-grid" id="scenarios-container">
        <!-- Rendered via JS -->
      </div>
    </div>

    <div class="grid-layout">
      <!-- Left Panel: Input Station -->
      <section class="panel">
        <div class="panel-header">
          <div class="panel-title">1. Open Package Intake</div>
          <span id="scenario-id-tag" style="font-size: 0.75rem; color: var(--text-muted); font-family: monospace;">example_1_correct_order</span>
        </div>

        <div class="img-preview-container" id="img-preview-box">
          <div class="scan-line"></div>
          <img id="box-image" src="" alt="Open Box Photo">
        </div>

        <div class="photo-actions">
          <label class="file-input-label">
            <input type="file" id="file-uploader" accept="image/*" style="display:none;" onchange="handleFileUpload(event)">
            📁 Upload Custom Photo
          </label>
        </div>

        <div class="order-editor-container">
          <div class="order-editor-label">
            <span>Expected Order Specification (JSON)</span>
            <span style="font-weight: normal; font-size: 0.72rem;">Editable</span>
          </div>
          <textarea id="order-json-input" class="order-editor" spellcheck="false"></textarea>
        </div>

        <button id="btn-run-verify" class="btn-verify" onclick="triggerInspection()">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          Run AI Pack Verification
        </button>
      </section>

      <!-- Right Panel: Inspection Verdict -->
      <section class="panel">
        <div class="panel-header">
          <div class="panel-title">2. AI Verification Verdict</div>
          <span id="order-id-label" style="font-size: 0.75rem; font-family: monospace; color: #60a5fa;">ORD-2024-001</span>
        </div>

        <div id="verdict-banner" class="verdict-banner SEAL">
          <div class="verdict-text-group">
            <h3 id="verdict-title">SEAL</h3>
            <div id="verdict-reason" class="verdict-reason">All items present, correct quantities, correct colors</div>
          </div>
          <div class="confidence-meter">
            <div style="font-size: 0.7rem; text-transform: uppercase; color: var(--text-muted); margin-bottom: 2px;">Confidence</div>
            <span id="verdict-confidence" class="confidence-tag conf-high">HIGH</span>
          </div>
        </div>

        <div id="damage-alert" style="display: none;" class="alert-card alert-damage">
          <strong>Physical Damage Warning:</strong>
          <span id="damage-notes">Product packaging shows visible compromise.</span>
        </div>

        <div id="missing-alert-box" style="display: none;" class="alert-card alert-missing">
          <div>
            <strong>Missing Items Detected:</strong>
            <div id="missing-items-list" style="margin-top: 4px;"></div>
          </div>
        </div>

        <div id="extra-alert-box" style="display: none;" class="alert-card alert-extra">
          <div>
            <strong>Unauthorized Extra Items Detected:</strong>
            <div id="extra-items-list" style="margin-top: 4px;"></div>
          </div>
        </div>

        <div class="section-title">Order Matching Matrix</div>
        <table class="match-table">
          <thead>
            <tr>
              <th>Item Name</th>
              <th>Expected</th>
              <th>Detected</th>
              <th>Variant Match</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody id="matches-table-body">
            <!-- Rendered via JS -->
          </tbody>
        </table>

        <div class="section-title">Visual Evidence & Findings</div>
        <ul id="evidence-list" class="evidence-list">
          <!-- Rendered via JS -->
        </ul>

        <div class="section-title" style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
          <span>Raw Agent Response (Canonical JSON)</span>
          <button onclick="copyRawJson()" style="background: none; border: none; color: #60a5fa; cursor: pointer; font-size: 0.75rem;">Copy JSON</button>
        </div>
        <pre id="raw-json-viewer" class="json-box"></pre>
      </section>
    </div>
  </main>

  <!-- Settings Modal -->
  <div id="settings-modal" class="modal">
    <div class="modal-content">
      <div class="modal-title">VLM Provider Configuration</div>
      <div class="form-group">
        <label class="form-label">Active Provider</label>
        <select id="cfg-provider" class="form-select" onchange="updateProviderOptions()">
          <option value="simulation">Offline Simulation (Local Heuristics)</option>
          <option value="gemini">Google Gemini (Gemini 2.0 Flash)</option>
          <option value="openai">OpenAI (GPT-4o / GPT-4o-mini)</option>
          <option value="anthropic">Anthropic (Claude 3.5 Sonnet)</option>
        </select>
      </div>
      <div class="form-group" id="api-key-group" style="display: none;">
        <label class="form-label">API Key</label>
        <input type="password" id="cfg-api-key" class="form-input" placeholder="Paste API Key here (stored only in browser)">
      </div>
      <div class="modal-actions">
        <button class="btn-secondary" onclick="closeSettingsModal()">Cancel</button>
        <button class="btn-primary" onclick="saveSettings()">Save & Apply</button>
      </div>
    </div>
  </div>

  <script>
    var currentScenario = "example_1_correct_order";
    var currentPhotoBase64 = "";
    var providerConfig = {
      provider: localStorage.getItem("pm_provider") || "simulation",
      apiKey: localStorage.getItem("pm_api_key") || ""
    };
    var scenarios = [
      { id: "example_1_correct_order", label: "1. Correct Order (SEAL)" },
      { id: "example_2_wrong_item",    label: "2. Wrong Item / Variant (STOP and FIX)" },
      { id: "example_3_missing_item",  label: "3. Missing Item (STOP and FIX)" },
      { id: "example_4_extra_item",    label: "4. Extra Item (STOP and FIX)" },
      { id: "example_5_unclear_photo", label: "5. Unclear Photo (UNCERTAIN)" },
      { id: "example_6_damaged_goods", label: "6. Product Damage (STOP and FIX)" }
    ];

    function init() {
      renderScenarioPills();
      updateProviderLabel();
      loadScenario(currentScenario);
    }

    function renderScenarioPills() {
      var container = document.getElementById("scenarios-container");
      var html = "";
      for (var i = 0; i < scenarios.length; i++) {
        var s = scenarios[i];
        var active = (s.id === currentScenario) ? " active" : "";
        html += '<button class="scenario-pill' + active + '" data-scenario="' + s.id + '">' + s.label + '</button>';
      }
      container.innerHTML = html;
    }

    function selectScenario(scenarioId) {
      currentScenario = scenarioId;
      renderScenarioPills();
      loadScenario(scenarioId);
    }

    async function loadScenario(scenarioId) {
      var imgEl = document.getElementById("box-image");
      document.getElementById("scenario-id-tag").innerText = scenarioId;
      imgEl.style.opacity = "0.4";
      try {
        var resp = await fetch("/api/scenario/" + scenarioId);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        var data = await resp.json();
        imgEl.src = data.photo_data_url;
        imgEl.style.opacity = "1";
        currentPhotoBase64 = data.photo_base64;
        document.getElementById("order-json-input").value = JSON.stringify(data.order, null, 2);
        triggerInspection();
      } catch (err) {
        imgEl.style.opacity = "1";
        console.error("Failed to load scenario", err);
        showError("Failed to load scenario: " + err.message);
      }
    }

    function handleFileUpload(event) {
      var file = event.target.files[0];
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function(ev) {
        document.getElementById("box-image").src = ev.target.result;
        currentPhotoBase64 = ev.target.result.split(",")[1];
        document.getElementById("scenario-id-tag").innerText = "custom: " + file.name;
      };
      reader.readAsDataURL(file);
    }

    async function triggerInspection() {
      var btn = document.getElementById("btn-run-verify");
      var imgBox = document.getElementById("img-preview-box");
      if (btn && btn.disabled) return;
      if (btn) {
        btn.disabled = true;
        btn.textContent = "Inspecting Package...";
      }
      if (imgBox) imgBox.classList.add("scanning");

      var orderPayload = {};
      try {
        orderPayload = JSON.parse(document.getElementById("order-json-input").value);
      } catch (e) {
        showError("Invalid JSON in order specification");
        resetBtn(btn, imgBox);
        return;
      }
      if (!currentPhotoBase64) {
        showError("No photo loaded - select a scenario first");
        resetBtn(btn, imgBox);
        return;
      }
      try {
        var resp = await fetch("/api/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            order: orderPayload,
            photo_base64: currentPhotoBase64,
            provider: providerConfig.provider,
            api_key: providerConfig.apiKey
          })
        });
        if (!resp.ok) {
          var errData = await resp.json().catch(function() { return { error: "Server error " + resp.status }; });
          throw new Error(errData.error || "Server returned " + resp.status);
        }
        renderResult(await resp.json());
      } catch (err) {
        console.error("Inspection error", err);
        showError("Inspection failed: " + err.message);
      } finally {
        resetBtn(btn, imgBox);
      }
    }

    function resetBtn(btn, imgBox) {
      btn.disabled = false;
      btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run AI Pack Verification';
      imgBox.classList.remove("scanning");
    }

    function showError(msg) {
      document.getElementById("verdict-banner").className = "verdict-banner STOP_FIX";
      document.getElementById("verdict-title").textContent = "ERROR";
      document.getElementById("verdict-reason").textContent = msg;
    }

    function renderResult(result) {
      document.getElementById("order-id-label").innerText = result.order_id || "ORD-INSPECTED";
      document.getElementById("verdict-banner").className = "verdict-banner " + result.decision;
      var titles = {
        "SEAL": "SEAL FOR SHIPPING",
        "STOP_FIX": "STOP and FIX - DO NOT SHIP",
        "UNCERTAIN": "UNCERTAIN - RETAKE PHOTO"
      };
      document.getElementById("verdict-title").innerText = titles[result.decision] || result.decision;
      document.getElementById("verdict-reason").innerText = result.decision_reason;

      var confTag = document.getElementById("verdict-confidence");
      confTag.innerText = (result.confidence || "high").toUpperCase();
      confTag.className = "confidence-tag conf-" + (result.confidence || "high");

      var damageAlert = document.getElementById("damage-alert");
      if (result.product_condition && result.product_condition.visible_damage) {
        damageAlert.style.display = "flex";
        document.getElementById("damage-notes").innerText = result.product_condition.notes;
      } else { damageAlert.style.display = "none"; }

      var missingBox = document.getElementById("missing-alert-box");
      if (result.missing_items && result.missing_items.length > 0) {
        missingBox.style.display = "flex";
        document.getElementById("missing-items-list").innerHTML = result.missing_items.map(function(m) {
          return "<strong>" + m.expected_qty + "x " + m.item_name + "</strong>: " + m.reason;
        }).join("<br>");
      } else { missingBox.style.display = "none"; }

      var extraBox = document.getElementById("extra-alert-box");
      if (result.extra_items && result.extra_items.length > 0) {
        extraBox.style.display = "flex";
        document.getElementById("extra-items-list").innerHTML = result.extra_items.map(function(e) {
          return "<strong>" + e.qty + "x " + e.item_name + "</strong>: " + e.notes;
        }).join("<br>");
      } else { extraBox.style.display = "none"; }

      var tbody = document.getElementById("matches-table-body");
      if (result.matches && result.matches.length > 0) {
        tbody.innerHTML = result.matches.map(function(m) {
          return "<tr><td><strong>" + m.item_name + "</strong></td><td>" + m.expected_qty +
            "</td><td>" + m.detected_qty + "</td><td>" +
            (m.variant_match ? "Yes" : "<span style='color:#f87171'>No</span>") +
            "</td><td><span class='" + (m.status === "PASS" ? "badge-pass" : "badge-fail") + "'>" + m.status + "</span></td></tr>";
        }).join("");
      } else {
        tbody.innerHTML = "<tr><td colspan='5' style='text-align:center;color:var(--text-muted)'>No match details</td></tr>";
      }

      var evList = document.getElementById("evidence-list");
      evList.innerHTML = (result.evidence && result.evidence.length > 0)
        ? result.evidence.map(function(e) { return "<li>" + e + "</li>"; }).join("")
        : "<li>No specific visual evidence listed</li>";

      document.getElementById("raw-json-viewer").innerText = JSON.stringify(result, null, 2);
    }

    function copyRawJson() {
      var text = document.getElementById("raw-json-viewer").innerText;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function() {
          alert("JSON copied to clipboard!");
        }).catch(function() {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    }

    function fallbackCopy(text) {
      var ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      try {
        document.execCommand("copy");
        alert("JSON copied to clipboard!");
      } catch (e) {
        alert("Could not copy automatically. Please copy the JSON manually.");
      }
      document.body.removeChild(ta);
    }

    function openSettingsModal() {
      document.getElementById("cfg-provider").value = providerConfig.provider;
      document.getElementById("cfg-api-key").value = providerConfig.apiKey;
      updateProviderOptions();
      document.getElementById("settings-modal").classList.add("open");
    }

    function closeSettingsModal() {
      document.getElementById("settings-modal").classList.remove("open");
    }

    function updateProviderOptions() {
      var p = document.getElementById("cfg-provider").value;
      document.getElementById("api-key-group").style.display = (p === "simulation") ? "none" : "block";
    }

    function saveSettings() {
      providerConfig.provider = document.getElementById("cfg-provider").value;
      providerConfig.apiKey = document.getElementById("cfg-api-key").value.trim();
      localStorage.setItem("pm_provider", providerConfig.provider);
      localStorage.setItem("pm_api_key", providerConfig.apiKey);
      updateProviderLabel();
      closeSettingsModal();
      triggerInspection();
    }

    function updateProviderLabel() {
      var labels = {
        "simulation": "Offline Simulation",
        "gemini": "Google Gemini 2.0 Flash",
        "openai": "OpenAI GPT-4o",
        "anthropic": "Anthropic Claude 3.5"
      };
      document.getElementById("provider-label").innerText = labels[providerConfig.provider] || providerConfig.provider;
    }

    function initApp() {
      var sc = document.getElementById("scenarios-container");
      if (sc && !sc._hasListener) {
        sc.addEventListener("click", function(e) {
          var pill = e.target.closest("[data-scenario]");
          if (pill) { selectScenario(pill.getAttribute("data-scenario")); }
        });
        sc._hasListener = true;
      }
      init();
    }

    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", initApp);
    } else {
      initApp();
    }
  </script>
</body>
</html>
"""


class PackManagerRequestHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler serving UI and Inspection API."""

    def _send_json(self, status: int, data: Any) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            self._send_html(HTML_TEMPLATE)
            return

        if path == "/api/scenarios":
            scenarios = [
                {"id": "example_1_correct_order", "title": "Example 1: Correct Order (SEAL)"},
                {"id": "example_2_wrong_item", "title": "Example 2: Wrong Item / Variant (STOP & FIX)"},
                {"id": "example_3_missing_item", "title": "Example 3: Missing Item (STOP & FIX)"},
                {"id": "example_4_extra_item", "title": "Example 4: Extra Item (STOP & FIX)"},
                {"id": "example_5_unclear_photo", "title": "Example 5: Unclear Photo (UNCERTAIN)"},
                {"id": "example_6_damaged_goods", "title": "Example 6: Product Damage (STOP & FIX)"},
            ]
            self._send_json(200, {"scenarios": scenarios})
            return

        if path.startswith("/api/scenario/"):
            scenario_id = path.split("/api/scenario/")[1].strip("/")
            scen_dir = FIXTURES_DIR / scenario_id
            if not scen_dir.exists():
                # Generate if not exists
                generate_all_scenarios(FIXTURES_DIR)

            order_file = scen_dir / "order.json"
            photo_file = scen_dir / "box_photo.png"
            if not (order_file.exists() and photo_file.exists()):
                self._send_json(404, {"error": f"Scenario {scenario_id} not found"})
                return

            order_data = json.loads(order_file.read_text(encoding="utf-8"))
            photo_bytes = photo_file.read_bytes()
            b64_photo = base64.b64encode(photo_bytes).decode("ascii")

            self._send_json(
                200,
                {
                    "scenario_id": scenario_id,
                    "order": order_data,
                    "photo_base64": b64_photo,
                    "photo_data_url": f"data:image/png;base64,{b64_photo}",
                },
            )
            return

        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path == "/api/verify":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body)
            except Exception as e:
                self._send_json(400, {"error": f"Invalid JSON payload: {e}"})
                return

            order_payload = req.get("order", {})
            photo_b64 = req.get("photo_base64", "")
            provider = req.get("provider")
            api_key = req.get("api_key")

            if not photo_b64:
                self._send_json(400, {"error": "Missing photo_base64 in request"})
                return

            try:
                photo_bytes = base64.b64decode(photo_b64)
            except Exception as e:
                self._send_json(400, {"error": f"Invalid base64 image data: {e}"})
                return

            # Instantiate VLM and Agent
            vlm = get_vlm_client(provider=provider, api_key=api_key)
            agent = PackManagerAIAgent(vlm_client=vlm)

            try:
                result = agent.verify(order=order_payload, photo=photo_bytes)
                self._send_json(200, result.to_summary_dict())
            except Exception as e:
                logger.exception("Verification execution failed")
                self._send_json(500, {"error": f"Inspection failed: {e}"})
            return

        self._send_json(404, {"error": "Not found"})


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    # Ensure all scenarios exist
    generate_all_scenarios(FIXTURES_DIR)
    server = ThreadingHTTPServer((host, port), PackManagerRequestHandler)
    print(f"Pack Manager AI Logistics Dashboard running at http://{host}:{port}/")
    print("Press Ctrl+C to terminate.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pack Manager AI Interactive Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default 8000)")
    args = parser.parse_args(argv)
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
