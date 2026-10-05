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
  <title>Pack Manager AI — Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --sidebar-bg: #0f172a;
      --sidebar-text: #94a3b8;
      --sidebar-hover: rgba(255, 255, 255, 0.06);
      --sidebar-active: #2563eb;
      --body-bg: #f8fafc;
      --card-bg: #ffffff;
      --card-border: #e2e8f0;
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-light: #94a3b8;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --radius: 14px;
      --card-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--body-bg);
      color: var(--text-main);
      display: flex;
      height: 100vh;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    /* ==========================================================================
       Sidebar
       ========================================================================== */
    .sidebar {
      width: 250px;
      background: var(--sidebar-bg);
      color: var(--sidebar-text);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 24px 16px 20px;
      flex-shrink: 0;
      border-right: 1px solid rgba(255, 255, 255, 0.05);
      z-index: 50;
    }
    .brand-header {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 0 8px 24px;
      cursor: pointer;
    }
    .brand-cube-icon {
      width: 34px;
      height: 34px;
      border-radius: 9px;
      background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
      display: flex;
      align-items: center;
      justify-content: center;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
      flex-shrink: 0;
    }
    .brand-title {
      font-size: 1.18rem;
      font-weight: 700;
      color: #ffffff;
      letter-spacing: -0.01em;
    }
    .nav-list {
      display: flex;
      flex-direction: column;
      gap: 4px;
      list-style: none;
    }
    .nav-item {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 11px 14px;
      border-radius: 9px;
      font-size: 0.88rem;
      font-weight: 500;
      color: var(--sidebar-text);
      cursor: pointer;
      transition: all 0.15s ease;
      user-select: none;
    }
    .nav-item:hover {
      background: var(--sidebar-hover);
      color: #ffffff;
    }
    .nav-item.active {
      background: var(--sidebar-active);
      color: #ffffff;
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .nav-item svg {
      width: 19px;
      height: 19px;
      stroke-width: 2;
      flex-shrink: 0;
    }

    /* Sidebar Footer User Profile */
    .sidebar-user {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 10px;
      border-radius: 10px;
      cursor: pointer;
      transition: background 0.15s ease;
      background: rgba(255, 255, 255, 0.03);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-user:hover {
      background: rgba(255, 255, 255, 0.08);
    }
    .user-avatar-circle {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
      color: #ffffff;
      font-weight: 700;
      font-size: 0.95rem;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
    }
    .user-info {
      flex: 1;
      min-width: 0;
    }
    .user-name {
      font-size: 0.88rem;
      font-weight: 600;
      color: #ffffff;
      line-height: 1.2;
    }
    .user-role {
      font-size: 0.73rem;
      color: var(--sidebar-text);
      line-height: 1.2;
    }
    .user-chevron {
      color: #64748b;
      font-size: 0.8rem;
    }

    /* ==========================================================================
       Main Wrapper & Topbar
       ========================================================================== */
    .main-wrapper {
      flex: 1;
      display: flex;
      flex-direction: column;
      height: 100vh;
      overflow-y: auto;
      background: var(--body-bg);
    }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 24px 36px 16px;
      position: sticky;
      top: 0;
      background: var(--body-bg);
      z-index: 20;
    }
    .page-title {
      font-size: 1.65rem;
      font-weight: 700;
      color: var(--text-main);
      letter-spacing: -0.02em;
      line-height: 1.2;
    }
    .page-subtitle {
      font-size: 0.88rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .topbar-right {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .search-box {
      position: relative;
      display: flex;
      align-items: center;
    }
    .search-box svg {
      position: absolute;
      left: 14px;
      width: 17px;
      height: 17px;
      color: var(--text-light);
      pointer-events: none;
    }
    .search-input {
      padding: 9px 16px 9px 40px;
      border-radius: 20px;
      border: 1px solid var(--card-border);
      background: #ffffff;
      font-size: 0.86rem;
      color: var(--text-main);
      width: 250px;
      outline: none;
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .search-input:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.12);
      width: 280px;
    }
    .icon-btn {
      width: 38px;
      height: 38px;
      border-radius: 50%;
      background: #ffffff;
      border: 1px solid var(--card-border);
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-muted);
      cursor: pointer;
      position: relative;
      transition: all 0.15s ease;
    }
    .icon-btn:hover {
      background: #f1f5f9;
      color: var(--text-main);
    }
    .badge-dot {
      position: absolute;
      top: 8px;
      right: 9px;
      width: 7px;
      height: 7px;
      background: #ef4444;
      border-radius: 50%;
      border: 2px solid #ffffff;
    }

    /* Content Area */
    .content-area {
      padding: 8px 36px 40px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* ==========================================================================
       Row 1: 4 KPI Cards
       ========================================================================== */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 20px;
    }
    .kpi-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 20px 22px;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 14px;
      transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .kpi-top {
      display: flex;
      align-items: center;
      gap: 14px;
    }
    .kpi-icon-box {
      width: 48px;
      height: 48px;
      border-radius: 12px;
      display: flex;
      align-items: center;
      justify-content: center;
      flex-shrink: 0;
    }
    .kpi-icon-box.blue { background: #eff6ff; color: #3b82f6; }
    .kpi-icon-box.green { background: #ecfdf5; color: #10b981; }
    .kpi-icon-box.orange { background: #fff7ed; color: #f59e0b; }
    .kpi-icon-box.red { background: #fef2f2; color: #ef4444; }
    .kpi-icon-box svg { width: 24px; height: 24px; stroke-width: 2.2; }

    .kpi-meta {
      display: flex;
      flex-direction: column;
    }
    .kpi-title {
      font-size: 0.84rem;
      font-weight: 500;
      color: var(--text-muted);
    }
    .kpi-value {
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1.1;
      margin-top: 2px;
    }
    .kpi-bottom {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-top: 2px;
    }
    .kpi-trend {
      font-size: 0.82rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 4px;
    }
    .kpi-trend.up { color: #10b981; }
    .kpi-trend.down { color: #ef4444; }
    .sparkline-svg {
      width: 80px;
      height: 26px;
      overflow: visible;
    }

    /* ==========================================================================
       Row 2: Charts Grid
       ========================================================================== */
    .charts-grid {
      display: grid;
      grid-template-columns: 1.55fr 1fr;
      gap: 20px;
    }
    .chart-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 22px 24px;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
    }
    .chart-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .chart-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
    }
    .select-pill {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
      background: #ffffff;
      border: 1px solid var(--card-border);
      padding: 5px 12px;
      border-radius: 8px;
      cursor: pointer;
      outline: none;
      font-family: inherit;
    }
    .select-pill:hover { border-color: #cbd5e1; }

    /* Stacked Bar Chart */
    .barchart-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      justify-content: flex-end;
      height: 210px;
      position: relative;
    }
    .barchart-gridlines {
      position: absolute;
      top: 0;
      left: 30px;
      right: 0;
      bottom: 28px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      pointer-events: none;
    }
    .gridline-row {
      display: flex;
      align-items: center;
      gap: 10px;
      height: 1px;
      border-top: 1px dashed #e2e8f0;
      position: relative;
    }
    .gridline-label {
      position: absolute;
      left: -32px;
      top: -8px;
      font-size: 0.72rem;
      color: var(--text-light);
      font-weight: 600;
      width: 24px;
      text-align: right;
    }
    .barchart-bars {
      display: flex;
      align-items: flex-end;
      justify-content: space-around;
      margin-left: 30px;
      height: 180px;
      z-index: 2;
    }
    .bar-column {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      height: 100%;
      justify-content: flex-end;
      width: 38px;
    }
    .stacked-bar-pillar {
      width: 20px;
      border-radius: 4px;
      overflow: hidden;
      display: flex;
      flex-direction: column-reverse;
      transition: transform 0.2s ease, filter 0.2s ease;
      cursor: pointer;
    }
    .stacked-bar-pillar:hover {
      transform: scaleY(1.04);
      filter: brightness(1.1);
    }
    .bar-seg-blue { background: #3b82f6; width: 100%; }
    .bar-seg-green { background: #10b981; width: 100%; }
    .bar-seg-orange { background: #f59e0b; width: 100%; }
    .bar-day-label {
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .chart-legend {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 22px;
      margin-top: 16px;
      padding-top: 14px;
      border-top: 1px solid #f1f5f9;
    }
    .legend-item {
      display: flex;
      align-items: center;
      gap: 7px;
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text-muted);
    }
    .legend-dot {
      width: 9px;
      height: 9px;
      border-radius: 50%;
    }
    .legend-dot.blue { background: #3b82f6; }
    .legend-dot.green { background: #10b981; }
    .legend-dot.orange { background: #f59e0b; }

    /* Donut Chart */
    .donut-content {
      display: flex;
      align-items: center;
      justify-content: space-around;
      flex: 1;
      padding: 10px 0;
    }
    .donut-visual-box {
      position: relative;
      width: 170px;
      height: 170px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .donut-center-text {
      position: absolute;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      pointer-events: none;
    }
    .donut-total-num {
      font-size: 1.85rem;
      font-weight: 800;
      color: var(--text-main);
      line-height: 1;
    }
    .donut-total-sub {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-top: 3px;
    }
    .donut-breakdown-list {
      display: flex;
      flex-direction: column;
      gap: 16px;
      min-width: 150px;
    }
    .breakdown-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      font-size: 0.86rem;
    }
    .breakdown-left {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text-muted);
      font-weight: 500;
    }
    .breakdown-stat {
      font-weight: 700;
      color: var(--text-main);
    }

    /* ==========================================================================
       Row 3: Recent Packages Table
       ========================================================================== */
    .table-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--card-shadow);
    }
    .table-card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 18px;
    }
    .table-view-all {
      font-size: 0.85rem;
      font-weight: 600;
      color: var(--primary);
      text-decoration: none;
      cursor: pointer;
    }
    .table-view-all:hover { text-decoration: underline; }
    .recent-table {
      width: 100%;
      border-collapse: collapse;
      text-align: left;
    }
    .recent-table th {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--text-light);
      padding: 12px 14px;
      border-bottom: 1px solid #f1f5f9;
    }
    .recent-table td {
      padding: 14px 14px;
      font-size: 0.86rem;
      color: var(--text-main);
      border-bottom: 1px solid #f8fafc;
      vertical-align: middle;
    }
    .recent-table tbody tr {
      transition: background 0.15s ease;
    }
    .recent-table tbody tr:hover {
      background: #f8fafc;
    }
    .tracking-code {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 600;
      color: var(--text-main);
    }
    .pkg-name-text {
      font-weight: 600;
      color: var(--text-main);
    }

    /* Pills */
    .pill-category {
      font-size: 0.74rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 12px;
      display: inline-block;
    }
    .pill-category.electronics { background: #dbeafe; color: #1d4ed8; }
    .pill-category.books { background: #f3e8ff; color: #7e22ce; }
    .pill-category.fashion { background: #ffe4e6; color: #be123c; }
    .pill-category.home { background: #ccfbf1; color: #0f766e; }
    .pill-category.grocery { background: #ffedd5; color: #c2410c; }

    .pill-status {
      font-size: 0.74rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 12px;
      display: inline-block;
    }
    .pill-status.delivered { background: #dcfce7; color: #15803d; }
    .pill-status.intransit { background: #fef3c7; color: #b45309; }
    .pill-status.pending { background: #fee2e2; color: #b91c1c; }

    .btn-action-more {
      background: none;
      border: none;
      color: var(--text-light);
      cursor: pointer;
      font-size: 1.15rem;
      line-height: 1;
      padding: 4px 8px;
      border-radius: 6px;
      transition: all 0.15s ease;
    }
    .btn-action-more:hover {
      background: #f1f5f9;
      color: var(--text-main);
    }

    @media (max-width: 1200px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
      .charts-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <!-- ======================================================================
       Left Sidebar Navigation
       ====================================================================== -->
  <aside class="sidebar">
    <div>
      <!-- Brand Logo -->
      <div class="brand-header">
        <div class="brand-cube-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
            <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
            <line x1="12" y1="22.08" x2="12" y2="12"></line>
          </svg>
        </div>
        <span class="brand-title">Pack Manager</span>
      </div>

      <!-- Nav Items -->
      <ul class="nav-list">
        <li class="nav-item active">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          <span>Dashboard</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
          <span>Packages</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          <span>Add Package</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
          <span>Categories</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
          <span>Inventory</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
          <span>Reports</span>
        </li>
        <li class="nav-item">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          <span>Settings</span>
        </li>
      </ul>
    </div>

    <!-- Sidebar Bottom: Vivek Profile -->
    <div class="sidebar-user">
      <div class="user-avatar-circle">V</div>
      <div class="user-info">
        <div class="user-name">Vivek</div>
        <div class="user-role">Admin</div>
      </div>
      <div class="user-chevron">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
      </div>
    </div>
  </aside>

  <!-- ======================================================================
       Main Content (Only the Dashboard)
       ====================================================================== -->
  <div class="main-wrapper">
    <!-- Topbar -->
    <header class="topbar">
      <div>
        <h1 class="page-title">Dashboard</h1>
        <p class="page-subtitle">Welcome back, Vivek! Here's an overview of your packages.</p>
      </div>
      <div class="topbar-right">
        <!-- Search bar -->
        <div class="search-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="package-search-input" class="search-input" placeholder="Search packages..." oninput="handleTableSearch(event)">
        </div>
        <!-- Notification button -->
        <div class="icon-btn">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
          <span class="badge-dot"></span>
        </div>
        <!-- Vivek Avatar Topbar -->
        <div class="user-avatar-circle" style="width: 36px; height: 36px; font-size: 0.9rem;">V</div>
      </div>
    </header>

    <!-- Content Area: The Exact Dashboard -->
    <div class="content-area">
      <!-- 4 KPI Stat Cards -->
      <div class="kpi-grid">
        <!-- Card 1: Total Packages -->
        <div class="kpi-card">
          <div class="kpi-top">
            <div class="kpi-icon-box blue">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
            </div>
            <div class="kpi-meta">
              <span class="kpi-title">Total Packages</span>
              <span class="kpi-value">120</span>
            </div>
          </div>
          <div class="kpi-bottom">
            <span class="kpi-trend up">↑ 12%</span>
            <svg class="sparkline-svg" viewBox="0 0 80 26" fill="none">
              <path d="M2 20 Q 20 8, 40 16 T 78 4" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <!-- Card 2: In Transit -->
        <div class="kpi-card">
          <div class="kpi-top">
            <div class="kpi-icon-box green">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="1" y="3" width="15" height="13"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>
            </div>
            <div class="kpi-meta">
              <span class="kpi-title">In Transit</span>
              <span class="kpi-value">28</span>
            </div>
          </div>
          <div class="kpi-bottom">
            <span class="kpi-trend up">↑ 8%</span>
            <svg class="sparkline-svg" viewBox="0 0 80 26" fill="none">
              <path d="M2 22 Q 22 18, 42 12 T 78 6" stroke="#10b981" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <!-- Card 3: Delivered -->
        <div class="kpi-card">
          <div class="kpi-top">
            <div class="kpi-icon-box orange">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
            </div>
            <div class="kpi-meta">
              <span class="kpi-title">Delivered</span>
              <span class="kpi-value">84</span>
            </div>
          </div>
          <div class="kpi-bottom">
            <span class="kpi-trend up">↑ 15%</span>
            <svg class="sparkline-svg" viewBox="0 0 80 26" fill="none">
              <path d="M2 18 Q 24 14, 44 8 T 78 4" stroke="#f59e0b" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
          </div>
        </div>

        <!-- Card 4: Pending -->
        <div class="kpi-card">
          <div class="kpi-top">
            <div class="kpi-icon-box red">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            </div>
            <div class="kpi-meta">
              <span class="kpi-title">Pending</span>
              <span class="kpi-value">8</span>
            </div>
          </div>
          <div class="kpi-bottom">
            <span class="kpi-trend down">↓ 5%</span>
            <svg class="sparkline-svg" viewBox="0 0 80 26" fill="none">
              <path d="M2 8 Q 24 14, 44 18 T 78 22" stroke="#ef4444" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
          </div>
        </div>
      </div>

      <!-- Charts Grid -->
      <div class="charts-grid">
        <!-- Left: Packages Overview Bar Chart -->
        <div class="chart-card">
          <div class="chart-card-header">
            <span class="chart-title">Packages Overview</span>
            <select class="select-pill">
              <option>Last 7 Days ⌵</option>
              <option>Last 30 Days</option>
              <option>This Month</option>
            </select>
          </div>

          <div class="barchart-container">
            <!-- Grid lines with Y-axis markers -->
            <div class="barchart-gridlines">
              <div class="gridline-row"><span class="gridline-label">50</span></div>
              <div class="gridline-row"><span class="gridline-label">40</span></div>
              <div class="gridline-row"><span class="gridline-label">30</span></div>
              <div class="gridline-row"><span class="gridline-label">20</span></div>
              <div class="gridline-row"><span class="gridline-label">10</span></div>
              <div class="gridline-row"><span class="gridline-label">0</span></div>
            </div>

            <!-- Stacked Bars (Mon - Sun) -->
            <div class="barchart-bars">
              <!-- Mon (Total 16: Delivered 8, In Transit 6, Pending 2) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 58px;" title="Mon: 16 Packages">
                  <div class="bar-seg-blue" style="height: 50%;"></div>
                  <div class="bar-seg-green" style="height: 38%;"></div>
                  <div class="bar-seg-orange" style="height: 12%;"></div>
                </div>
                <span class="bar-day-label">Mon</span>
              </div>
              <!-- Tue (Total 24) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 86px;" title="Tue: 24 Packages">
                  <div class="bar-seg-blue" style="height: 45%;"></div>
                  <div class="bar-seg-green" style="height: 35%;"></div>
                  <div class="bar-seg-orange" style="height: 20%;"></div>
                </div>
                <span class="bar-day-label">Tue</span>
              </div>
              <!-- Wed (Total 23) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 82px;" title="Wed: 23 Packages">
                  <div class="bar-seg-blue" style="height: 48%;"></div>
                  <div class="bar-seg-green" style="height: 32%;"></div>
                  <div class="bar-seg-orange" style="height: 20%;"></div>
                </div>
                <span class="bar-day-label">Wed</span>
              </div>
              <!-- Thu (Total 27) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 98px;" title="Thu: 27 Packages">
                  <div class="bar-seg-blue" style="height: 45%;"></div>
                  <div class="bar-seg-green" style="height: 37%;"></div>
                  <div class="bar-seg-orange" style="height: 18%;"></div>
                </div>
                <span class="bar-day-label">Thu</span>
              </div>
              <!-- Fri (Peak Total 42) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 152px;" title="Fri: 42 Packages (Peak Day)">
                  <div class="bar-seg-blue" style="height: 57%;"></div>
                  <div class="bar-seg-green" style="height: 29%;"></div>
                  <div class="bar-seg-orange" style="height: 14%;"></div>
                </div>
                <span class="bar-day-label">Fri</span>
              </div>
              <!-- Sat (Total 23) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 82px;" title="Sat: 23 Packages">
                  <div class="bar-seg-blue" style="height: 45%;"></div>
                  <div class="bar-seg-green" style="height: 35%;"></div>
                  <div class="bar-seg-orange" style="height: 20%;"></div>
                </div>
                <span class="bar-day-label">Sat</span>
              </div>
              <!-- Sun (Total 31) -->
              <div class="bar-column">
                <div class="stacked-bar-pillar" style="height: 112px;" title="Sun: 31 Packages">
                  <div class="bar-seg-blue" style="height: 52%;"></div>
                  <div class="bar-seg-green" style="height: 32%;"></div>
                  <div class="bar-seg-orange" style="height: 16%;"></div>
                </div>
                <span class="bar-day-label">Sun</span>
              </div>
            </div>
          </div>

          <!-- Legend -->
          <div class="chart-legend">
            <div class="legend-item"><span class="legend-dot blue"></span> Delivered</div>
            <div class="legend-item"><span class="legend-dot green"></span> In Transit</div>
            <div class="legend-item"><span class="legend-dot orange"></span> Pending</div>
          </div>
        </div>

        <!-- Right: Package Status Donut Chart -->
        <div class="chart-card">
          <div class="chart-card-header">
            <span class="chart-title">Package Status</span>
          </div>

          <div class="donut-content">
            <div class="donut-visual-box">
              <!-- SVG Donut Chart with stroke-dasharray breakdown -->
              <svg width="160" height="160" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                <!-- Background base ring -->
                <circle cx="50" cy="50" r="38" fill="none" stroke="#f1f5f9" stroke-width="14"/>
                <!-- Blue Segment: 70% of 238.76 = 167.13 -->
                <circle cx="50" cy="50" r="38" fill="none" stroke="#3b82f6" stroke-width="14"
                        stroke-dasharray="167.13 238.76" stroke-dashoffset="0"/>
                <!-- Green Segment: 23% of 238.76 = 54.91 -->
                <circle cx="50" cy="50" r="38" fill="none" stroke="#10b981" stroke-width="14"
                        stroke-dasharray="54.91 238.76" stroke-dashoffset="-167.13"/>
                <!-- Orange Segment: 7% of 238.76 = 16.71 -->
                <circle cx="50" cy="50" r="38" fill="none" stroke="#f59e0b" stroke-width="14"
                        stroke-dasharray="16.71 238.76" stroke-dashoffset="-222.04"/>
              </svg>

              <div class="donut-center-text">
                <span class="donut-total-num">120</span>
                <span class="donut-total-sub">Total</span>
              </div>
            </div>

            <!-- Breakdown list -->
            <div class="donut-breakdown-list">
              <div class="breakdown-row">
                <div class="breakdown-left">
                  <span class="legend-dot blue"></span> Delivered
                </div>
                <span class="breakdown-stat">84 (70%)</span>
              </div>
              <div class="breakdown-row">
                <div class="breakdown-left">
                  <span class="legend-dot green"></span> In Transit
                </div>
                <span class="breakdown-stat">28 (23%)</span>
              </div>
              <div class="breakdown-row">
                <div class="breakdown-left">
                  <span class="legend-dot orange"></span> Pending
                </div>
                <span class="breakdown-stat">8 (7%)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Recent Packages Table -->
      <div class="table-card">
        <div class="table-card-header">
          <span class="chart-title">Recent Packages</span>
          <span class="table-view-all">View All</span>
        </div>

        <table class="recent-table" id="recent-packages-table">
          <thead>
            <tr>
              <th>Tracking ID</th>
              <th>Package Name</th>
              <th>Category</th>
              <th>Status</th>
              <th>Location</th>
              <th>Date</th>
              <th style="text-align: right;">Actions</th>
            </tr>
          </thead>
          <tbody id="packages-table-body">
            <tr>
              <td class="tracking-code">PKG001</td>
              <td class="pkg-name-text">Electronics Items</td>
              <td><span class="pill-category electronics">Electronics</span></td>
              <td><span class="pill-status delivered">Delivered</span></td>
              <td>Hyderabad</td>
              <td style="color: var(--text-muted);">Oct 5, 2026</td>
              <td style="text-align: right;">
                <button class="btn-action-more">⋮</button>
              </td>
            </tr>
            <tr>
              <td class="tracking-code">PKG002</td>
              <td class="pkg-name-text">Books Parcel</td>
              <td><span class="pill-category books">Books</span></td>
              <td><span class="pill-status intransit">In Transit</span></td>
              <td>Bengaluru</td>
              <td style="color: var(--text-muted);">Oct 5, 2026</td>
              <td style="text-align: right;">
                <button class="btn-action-more">⋮</button>
              </td>
            </tr>
            <tr>
              <td class="tracking-code">PKG003</td>
              <td class="pkg-name-text">Clothes Package</td>
              <td><span class="pill-category fashion">Fashion</span></td>
              <td><span class="pill-status pending">Pending</span></td>
              <td>Chennai</td>
              <td style="color: var(--text-muted);">Oct 4, 2026</td>
              <td style="text-align: right;">
                <button class="btn-action-more">⋮</button>
              </td>
            </tr>
            <tr>
              <td class="tracking-code">PKG004</td>
              <td class="pkg-name-text">Home Essentials</td>
              <td><span class="pill-category home">Home</span></td>
              <td><span class="pill-status delivered">Delivered</span></td>
              <td>Mumbai</td>
              <td style="color: var(--text-muted);">Oct 4, 2026</td>
              <td style="text-align: right;">
                <button class="btn-action-more">⋮</button>
              </td>
            </tr>
            <tr>
              <td class="tracking-code">PKG005</td>
              <td class="pkg-name-text">Grocery Items</td>
              <td><span class="pill-category grocery">Grocery</span></td>
              <td><span class="pill-status intransit">In Transit</span></td>
              <td>Delhi</td>
              <td style="color: var(--text-muted);">Oct 3, 2026</td>
              <td style="text-align: right;">
                <button class="btn-action-more">⋮</button>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- JavaScript -->
  <script>
    function handleTableSearch(e) {
      var query = e.target.value.toLowerCase();
      var rows = document.querySelectorAll("#packages-table-body tr");
      rows.forEach(function(row) {
        var text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
      });
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
