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
  <title>Pack Manager AI — Logistics & Inspection Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --sidebar-bg: #0f172a;
      --sidebar-hover: rgba(255, 255, 255, 0.07);
      --sidebar-active: #2563eb;
      --sidebar-text: #94a3b8;
      --sidebar-text-active: #ffffff;
      --body-bg: #f8fafc;
      --card-bg: #ffffff;
      --card-border: #e2e8f0;
      --card-shadow: 0 1px 3px rgba(15, 23, 42, 0.05), 0 1px 2px rgba(15, 23, 42, 0.03);
      --text-main: #0f172a;
      --text-muted: #64748b;
      --text-light: #94a3b8;
      --primary: #2563eb;
      --primary-hover: #1d4ed8;
      --primary-subtle: #eff6ff;
      --success: #10b981;
      --success-subtle: #ecfdf5;
      --warning: #f59e0b;
      --warning-subtle: #fff7ed;
      --danger: #ef4444;
      --danger-subtle: #fef2f2;
      --radius: 14px;
      --radius-sm: 8px;
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
      font-size: 1.15rem;
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
      border: 1px solid transparent;
    }
    .nav-item:hover {
      background: var(--sidebar-hover);
      color: #ffffff;
    }
    .nav-item.active {
      background: var(--sidebar-active);
      color: var(--sidebar-text-active);
      font-weight: 600;
      box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    .nav-item svg {
      width: 19px;
      height: 19px;
      stroke-width: 2;
      flex-shrink: 0;
    }
    .nav-pill-badge {
      margin-left: auto;
      font-size: 0.68rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 12px;
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      text-transform: uppercase;
      letter-spacing: 0.04em;
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
       Main Content Layout
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
      width: 290px;
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
      background: var(--danger);
      border-radius: 50%;
      border: 2px solid #ffffff;
    }
    .btn-verify-station {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 8px 16px;
      background: #2563eb;
      color: #ffffff;
      border: none;
      border-radius: 20px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
      transition: all 0.15s ease;
      font-family: inherit;
    }
    .btn-verify-station:hover {
      background: #1d4ed8;
      transform: translateY(-1px);
    }

    /* Content Area */
    .content-area {
      padding: 8px 36px 40px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }

    /* ==========================================================================
       Dashboard View: KPI Stat Cards
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
       Dashboard View: Charts Grid
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

    /* Bar Chart custom CSS */
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

    /* Donut Chart custom CSS */
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
       Dashboard View: Recent Packages Table
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
      cursor: pointer;
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

    /* ==========================================================================
       AI Inspector View: Vertical 1 -> 2 Layout
       ========================================================================== */
    .inspector-view-container {
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .scenario-selection-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 18px 24px;
      box-shadow: var(--card-shadow);
    }
    .scenarios-label {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      color: var(--text-muted);
      letter-spacing: 0.05em;
      margin-bottom: 12px;
    }
    .scenarios-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .scenario-pill {
      background: #f8fafc;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 8px 14px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .scenario-pill:hover {
      background: #e2e8f0;
      transform: translateY(-1px);
    }
    .scenario-pill.active {
      background: #2563eb;
      color: #ffffff;
      border-color: #2563eb;
      box-shadow: 0 3px 10px rgba(37, 99, 235, 0.25);
    }

    .panel-station {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 24px;
      box-shadow: var(--card-shadow);
    }
    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding-bottom: 14px;
      margin-bottom: 18px;
      border-bottom: 1px solid #f1f5f9;
    }
    .panel-title {
      font-size: 1.05rem;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .step-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 26px;
      height: 26px;
      border-radius: 8px;
      background: linear-gradient(135deg, #2563eb, #3b82f6);
      color: #ffffff;
      font-size: 0.82rem;
      font-weight: 800;
      box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25);
    }
    .intake-grid {
      display: grid;
      grid-template-columns: 1fr 1.05fr;
      gap: 24px;
      align-items: stretch;
    }
    @media (max-width: 900px) {
      .intake-grid { grid-template-columns: 1fr; }
    }
    .img-preview-container {
      position: relative;
      width: 100%;
      height: 330px;
      background: #090d16;
      border: 2px dashed #cbd5e1;
      border-radius: 12px;
      overflow: hidden;
      display: flex;
      align-items: center;
      justify-content: center;
      margin-bottom: 14px;
    }
    .img-preview-container img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
    .scan-line {
      position: absolute;
      top: 0; left: 0; right: 0;
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
    .photo-actions { display: flex; gap: 10px; }
    .file-input-label {
      background: #f8fafc;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      padding: 9px 16px;
      border-radius: 8px;
      font-size: 0.84rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .file-input-label:hover { background: #f1f5f9; }

    .order-editor-container {
      display: flex;
      flex-direction: column;
      flex: 1;
    }
    .order-editor-label {
      font-size: 0.82rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
    }
    .order-editor {
      width: 100%;
      flex: 1;
      min-height: 250px;
      background: #090d16;
      border: 1px solid #334155;
      color: #93c5fd;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.84rem;
      padding: 14px;
      border-radius: 10px;
      resize: vertical;
      line-height: 1.5;
    }
    .order-editor:focus { outline: none; border-color: #3b82f6; }
    .btn-verify {
      width: 100%;
      padding: 13px 20px;
      background: #2563eb;
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 0.94rem;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3);
      transition: all 0.2s ease;
      font-family: inherit;
    }
    .btn-verify:hover {
      background: #1d4ed8;
      transform: translateY(-1px);
    }
    .btn-verify:disabled { opacity: 0.6; cursor: not-allowed; }

    /* Verdict Banner & Inspection Details */
    .verdict-banner {
      padding: 18px 24px;
      border-radius: 12px;
      margin-bottom: 20px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      border: 1px solid transparent;
      transition: all 0.3s ease;
    }
    .verdict-banner.SEAL {
      background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.05));
      border-color: rgba(16, 185, 129, 0.3);
      color: #065f46;
    }
    .verdict-banner.STOP_FIX {
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.12), rgba(239, 68, 68, 0.05));
      border-color: rgba(239, 68, 68, 0.3);
      color: #991b1b;
    }
    .verdict-banner.UNCERTAIN {
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(245, 158, 11, 0.05));
      border-color: rgba(245, 158, 11, 0.3);
      color: #92400e;
    }
    .verdict-text-group h3 {
      font-size: 1.5rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      margin-bottom: 4px;
    }
    .verdict-reason { font-size: 0.88rem; font-weight: 500; opacity: 0.9; }
    .confidence-meter { text-align: right; }
    .confidence-tag {
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
      display: inline-block;
    }
    .conf-high { background: rgba(16, 185, 129, 0.2); color: #065f46; }
    .conf-medium { background: rgba(245, 158, 11, 0.2); color: #92400e; }
    .conf-low { background: rgba(239, 68, 68, 0.2); color: #991b1b; }

    .alert-card {
      padding: 12px 16px;
      border-radius: 10px;
      margin-bottom: 14px;
      font-size: 0.85rem;
      display: flex;
      gap: 10px;
      align-items: flex-start;
      border: 1px solid transparent;
    }
    .alert-damage { background: #fef2f2; border-color: #fecaca; color: #991b1b; }
    .alert-missing { background: #fff7ed; border-color: #fed7aa; color: #9a3412; }
    .alert-extra { background: #f5f3ff; border-color: #ddd6fe; color: #5b21b6; }

    .section-title {
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-main);
      margin: 20px 0 10px;
      letter-spacing: -0.01em;
    }
    .match-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.86rem;
      margin-bottom: 18px;
    }
    .match-table th {
      padding: 10px 14px;
      text-align: left;
      font-size: 0.75rem;
      font-weight: 700;
      color: var(--text-muted);
      border-bottom: 1px solid #e2e8f0;
      background: #f8fafc;
    }
    .match-table td {
      padding: 12px 14px;
      border-bottom: 1px solid #f1f5f9;
      color: var(--text-main);
    }
    .badge-pass {
      padding: 3px 8px;
      background: #dcfce7;
      color: #15803d;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 700;
    }
    .badge-fail {
      padding: 3px 8px;
      background: #fee2e2;
      color: #b91c1c;
      border-radius: 12px;
      font-size: 0.72rem;
      font-weight: 700;
    }
    .evidence-list {
      list-style: none;
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 16px;
    }
    .evidence-list li {
      position: relative;
      padding-left: 20px;
      font-size: 0.85rem;
      color: var(--text-muted);
      line-height: 1.45;
    }
    .evidence-list li::before {
      content: "•";
      position: absolute;
      left: 6px;
      color: #3b82f6;
      font-weight: bold;
      font-size: 1.1rem;
      line-height: 1;
    }
    .json-box {
      background: #090d16;
      border: 1px solid #334155;
      color: #a5f3fc;
      font-family: 'JetBrains Mono', monospace;
      font-size: 0.78rem;
      padding: 14px;
      border-radius: 10px;
      max-height: 220px;
      overflow-y: auto;
      white-space: pre-wrap;
    }

    /* ==========================================================================
       Modal
       ========================================================================== */
    .modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.6);
      backdrop-filter: blur(4px);
      z-index: 100;
      align-items: center;
      justify-content: center;
    }
    .modal.open { display: flex; }
    .modal-content {
      background: #ffffff;
      border-radius: 16px;
      padding: 28px;
      width: 90%;
      max-width: 480px;
      box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
    }
    .modal-title {
      font-size: 1.25rem;
      font-weight: 700;
      color: var(--text-main);
      margin-bottom: 18px;
    }
    .form-group { margin-bottom: 16px; }
    .form-label {
      display: block;
      font-size: 0.84rem;
      font-weight: 600;
      color: var(--text-muted);
      margin-bottom: 6px;
    }
    .form-select, .form-input {
      width: 100%;
      padding: 10px 14px;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      background: #ffffff;
      color: var(--text-main);
      font-family: inherit;
      font-size: 0.88rem;
    }
    .form-select:focus, .form-input:focus {
      outline: none;
      border-color: #2563eb;
    }
    .modal-actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 24px;
    }
    .btn-secondary {
      padding: 9px 16px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      color: var(--text-main);
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-primary {
      padding: 9px 18px;
      background: #2563eb;
      border: none;
      color: #ffffff;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
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
      <div class="brand-header" onclick="switchView('dashboard')">
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
        <li class="nav-item active" id="nav-dashboard" onclick="switchView('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          <span>Dashboard</span>
        </li>
        <li class="nav-item" id="nav-packages" onclick="switchView('inspector')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
          <span>Packages</span>
          <span class="nav-pill-badge">AI Live</span>
        </li>
        <li class="nav-item" onclick="openAddPackageModal()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          <span>Add Package</span>
        </li>
        <li class="nav-item" onclick="switchView('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
          <span>Categories</span>
        </li>
        <li class="nav-item" onclick="switchView('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
          <span>Inventory</span>
        </li>
        <li class="nav-item" onclick="switchView('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
          <span>Reports</span>
        </li>
        <li class="nav-item" onclick="openSettingsModal()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          <span>Settings</span>
        </li>
      </ul>
    </div>

    <!-- Sidebar Bottom: Vivek Profile -->
    <div class="sidebar-user" onclick="openSettingsModal()">
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
       Main Wrapper
       ====================================================================== -->
  <div class="main-wrapper">
    <!-- Topbar -->
    <header class="topbar">
      <div>
        <h1 class="page-title" id="page-title-heading">Dashboard</h1>
        <p class="page-subtitle" id="page-subtitle-heading">Welcome back, Vivek! Here's an overview of your packages.</p>
      </div>
      <div class="topbar-right">
        <!-- Search bar -->
        <div class="search-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="package-search-input" class="search-input" placeholder="Search packages..." oninput="handleTableSearch(event)">
        </div>
        <!-- Notification button -->
        <div class="icon-btn" onclick="alert('System Notifications: All 6 inspection stations operational.')">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
          <span class="badge-dot"></span>
        </div>
        <!-- Vivek Avatar Topbar -->
        <div class="user-avatar-circle" style="width: 36px; height: 36px; font-size: 0.9rem; cursor: pointer;" onclick="openSettingsModal()">V</div>
        <!-- AI Station quick toggle button -->
        <button class="btn-verify-station" id="btn-quick-switch" onclick="toggleInspectorView()">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
          <span id="btn-quick-switch-text">AI Verification Station</span>
        </button>
      </div>
    </header>

    <!-- Content Area -->
    <div class="content-area">
      <!-- ==================================================================
           VIEW 1: EXECUTIVE DASHBOARD (Matches User Screenshot 1:1)
           ================================================================== -->
      <div id="view-dashboard">
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
        <div class="charts-grid" style="margin-top: 24px;">
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
        <div class="table-card" style="margin-top: 24px;">
          <div class="table-card-header">
            <span class="chart-title">Recent Packages</span>
            <span class="table-view-all" onclick="switchView('inspector')">View All</span>
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
              <tr onclick="loadScenarioFromRow('example_1_correct_order')" title="Click to inspect this package in AI Station">
                <td class="tracking-code">PKG001</td>
                <td class="pkg-name-text">Electronics Items</td>
                <td><span class="pill-category electronics">Electronics</span></td>
                <td><span class="pill-status delivered">Delivered</span></td>
                <td>Hyderabad</td>
                <td style="color: var(--text-muted);">Oct 5, 2026</td>
                <td style="text-align: right;">
                  <button class="btn-action-more" title="Inspect">⋮</button>
                </td>
              </tr>
              <tr onclick="loadScenarioFromRow('example_2_wrong_item')" title="Click to inspect this package in AI Station">
                <td class="tracking-code">PKG002</td>
                <td class="pkg-name-text">Books Parcel</td>
                <td><span class="pill-category books">Books</span></td>
                <td><span class="pill-status intransit">In Transit</span></td>
                <td>Bengaluru</td>
                <td style="color: var(--text-muted);">Oct 5, 2026</td>
                <td style="text-align: right;">
                  <button class="btn-action-more" title="Inspect">⋮</button>
                </td>
              </tr>
              <tr onclick="loadScenarioFromRow('example_3_missing_item')" title="Click to inspect this package in AI Station">
                <td class="tracking-code">PKG003</td>
                <td class="pkg-name-text">Clothes Package</td>
                <td><span class="pill-category fashion">Fashion</span></td>
                <td><span class="pill-status pending">Pending</span></td>
                <td>Chennai</td>
                <td style="color: var(--text-muted);">Oct 4, 2026</td>
                <td style="text-align: right;">
                  <button class="btn-action-more" title="Inspect">⋮</button>
                </td>
              </tr>
              <tr onclick="loadScenarioFromRow('example_1_correct_order')" title="Click to inspect this package in AI Station">
                <td class="tracking-code">PKG004</td>
                <td class="pkg-name-text">Home Essentials</td>
                <td><span class="pill-category home">Home</span></td>
                <td><span class="pill-status delivered">Delivered</span></td>
                <td>Mumbai</td>
                <td style="color: var(--text-muted);">Oct 4, 2026</td>
                <td style="text-align: right;">
                  <button class="btn-action-more" title="Inspect">⋮</button>
                </td>
              </tr>
              <tr onclick="loadScenarioFromRow('example_4_extra_item')" title="Click to inspect this package in AI Station">
                <td class="tracking-code">PKG005</td>
                <td class="pkg-name-text">Grocery Items</td>
                <td><span class="pill-category grocery">Grocery</span></td>
                <td><span class="pill-status intransit">In Transit</span></td>
                <td>Delhi</td>
                <td style="color: var(--text-muted);">Oct 3, 2026</td>
                <td style="text-align: right;">
                  <button class="btn-action-more" title="Inspect">⋮</button>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ==================================================================
           VIEW 2: AI PACK INSPECTOR STATION (Vertical 1 -> 2 Flow)
           ================================================================== -->
      <div id="view-inspector" style="display: none;" class="inspector-view-container">
        <!-- Preset Evaluation Scenarios Bar -->
        <div class="scenario-selection-card">
          <div class="scenarios-label">Preset Evaluation Scenarios (From AI Prompt Spec)</div>
          <div class="scenarios-grid" id="scenarios-container">
            <!-- Rendered via JS -->
          </div>
        </div>

        <!-- 1. Open Package Intake & Order Spec -->
        <section class="panel-station">
          <div class="panel-header">
            <div class="panel-title">
              <span class="step-badge">1</span>
              <span>Open Package Intake &amp; Order Spec</span>
            </div>
            <span id="scenario-id-tag" style="font-size: 0.78rem; color: var(--text-muted); font-family: 'JetBrains Mono', monospace; background: #f1f5f9; padding: 4px 10px; border-radius: 6px;">example_1_correct_order</span>
          </div>

          <div class="intake-grid">
            <div>
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
            </div>

            <div style="display: flex; flex-direction: column;">
              <div class="order-editor-container">
                <div class="order-editor-label">
                  <span>Expected Order Specification (JSON)</span>
                  <span style="font-weight: normal; font-size: 0.72rem; color: #3b82f6;">Editable</span>
                </div>
                <textarea id="order-json-input" class="order-editor" spellcheck="false"></textarea>
              </div>

              <button id="btn-run-verify" class="btn-verify" onclick="triggerInspection()" style="margin-top: 14px;">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Run AI Pack Verification
              </button>
            </div>
          </div>
        </section>

        <!-- 2. AI Verification Verdict & Dashboard -->
        <section class="panel-station">
          <div class="panel-header">
            <div class="panel-title">
              <span class="step-badge">2</span>
              <span>AI Verification Verdict &amp; Dashboard</span>
            </div>
            <span id="order-id-label" style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; color: #2563eb; background: #eff6ff; padding: 4px 10px; border-radius: 6px;">ORD-2024-001</span>
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

          <div class="section-title">Visual Evidence &amp; Findings</div>
          <ul id="evidence-list" class="evidence-list">
            <!-- Rendered via JS -->
          </ul>

          <div class="section-title" style="display: flex; justify-content: space-between; align-items: center; margin-top: 20px;">
            <span>Raw Agent Response (Canonical JSON)</span>
            <button onclick="copyRawJson()" style="background: none; border: none; color: #2563eb; cursor: pointer; font-size: 0.75rem; font-weight: 600;">Copy JSON</button>
          </div>
          <pre id="raw-json-viewer" class="json-box"></pre>
        </section>
      </div>
    </div>
  </div>

  <!-- ======================================================================
       Modals: Settings and Add Package
       ====================================================================== -->
  <!-- VLM Provider Settings Modal -->
  <div id="settings-modal" class="modal">
    <div class="modal-content">
      <div class="modal-title">System &amp; VLM Configuration</div>
      <div class="form-group">
        <label class="form-label">Active Vision Model</label>
        <select id="cfg-provider" class="form-select" onchange="updateProviderOptions()">
          <option value="simulation">Offline Simulation (Local Logistics Heuristics)</option>
          <option value="gemini">Google Gemini (Gemini 2.0 Flash)</option>
          <option value="openai">OpenAI (GPT-4o / GPT-4o-mini)</option>
          <option value="anthropic">Anthropic (Claude 3.5 Sonnet)</option>
        </select>
      </div>
      <div class="form-group" id="api-key-group" style="display: none;">
        <label class="form-label">API Key</label>
        <input type="password" id="cfg-api-key" class="form-input" placeholder="Paste API Key here (stored only in browser localStorage)">
      </div>
      <div class="modal-actions">
        <button class="btn-secondary" onclick="closeSettingsModal()">Cancel</button>
        <button class="btn-primary" onclick="saveSettings()">Save &amp; Apply</button>
      </div>
    </div>
  </div>

  <!-- Add Package Modal -->
  <div id="add-package-modal" class="modal">
    <div class="modal-content">
      <div class="modal-title">Register New Package</div>
      <div class="form-group">
        <label class="form-label">Tracking ID</label>
        <input type="text" id="add-pkg-id" class="form-input" value="PKG006">
      </div>
      <div class="form-group">
        <label class="form-label">Package Name</label>
        <input type="text" id="add-pkg-name" class="form-input" placeholder="e.g. Wireless Headphones">
      </div>
      <div class="form-group">
        <label class="form-label">Category</label>
        <select id="add-pkg-category" class="form-select">
          <option value="electronics">Electronics</option>
          <option value="fashion">Fashion</option>
          <option value="books">Books</option>
          <option value="home">Home</option>
          <option value="grocery">Grocery</option>
        </select>
      </div>
      <div class="form-group">
        <label class="form-label">Destination Location</label>
        <input type="text" id="add-pkg-location" class="form-input" placeholder="e.g. Kolkata, IN">
      </div>
      <div class="modal-actions">
        <button class="btn-secondary" onclick="closeAddPackageModal()">Cancel</button>
        <button class="btn-primary" onclick="saveNewPackage()">Add &amp; Inspect</button>
      </div>
    </div>
  </div>

  <!-- ======================================================================
       Application Logic (JavaScript)
       ====================================================================== -->
  <script>
    var currentView = "dashboard";
    var currentScenario = "example_1_correct_order";
    var currentPhotoBase64 = "";
    var providerConfig = {
      provider: localStorage.getItem("pm_provider") || "simulation",
      apiKey: localStorage.getItem("pm_api_key") || ""
    };
    var scenarios = [
      { id: "example_1_correct_order", label: "1. Correct Order (SEAL)" },
      { id: "example_2_wrong_item",    label: "2. Wrong Item / Variant (STOP & FIX)" },
      { id: "example_3_missing_item",  label: "3. Missing Item (STOP & FIX)" },
      { id: "example_4_extra_item",    label: "4. Extra Item (STOP & FIX)" },
      { id: "example_5_unclear_photo", label: "5. Unclear Photo (UNCERTAIN)" },
      { id: "example_6_damaged_goods", label: "6. Product Damage (STOP & FIX)" }
    ];

    function init() {
      renderScenarioPills();
      loadScenario(currentScenario, false);
    }

    /* Switch between Executive Dashboard and AI Inspector */
    function switchView(viewName) {
      currentView = viewName;
      var dashView = document.getElementById("view-dashboard");
      var inspView = document.getElementById("view-inspector");
      var navDash = document.getElementById("nav-dashboard");
      var navPkg = document.getElementById("nav-packages");
      var pageTitle = document.getElementById("page-title-heading");
      var pageSubtitle = document.getElementById("page-subtitle-heading");
      var switchBtnText = document.getElementById("btn-quick-switch-text");

      if (viewName === "dashboard") {
        dashView.style.display = "block";
        inspView.style.display = "none";
        navDash.classList.add("active");
        navPkg.classList.remove("active");
        pageTitle.innerText = "Dashboard";
        pageSubtitle.innerText = "Welcome back, Vivek! Here's an overview of your packages.";
        switchBtnText.innerText = "AI Verification Station";
      } else {
        dashView.style.display = "none";
        inspView.style.display = "flex";
        navDash.classList.remove("active");
        navPkg.classList.add("active");
        pageTitle.innerText = "AI Outbound Inspection Station";
        pageSubtitle.innerText = "Autonomous package verification before seal & dispatch.";
        switchBtnText.innerText = "Back to Dashboard";
      }
    }

    function toggleInspectorView() {
      if (currentView === "dashboard") {
        switchView("inspector");
      } else {
        switchView("dashboard");
      }
    }

    function loadScenarioFromRow(scenarioId) {
      switchView("inspector");
      selectScenario(scenarioId);
    }

    function renderScenarioPills() {
      var container = document.getElementById("scenarios-container");
      if (!container) return;
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
      loadScenario(scenarioId, true);
    }

    async function loadScenario(scenarioId, autoVerify) {
      var imgEl = document.getElementById("box-image");
      var tagEl = document.getElementById("scenario-id-tag");
      if (tagEl) tagEl.innerText = scenarioId;
      if (imgEl) imgEl.style.opacity = "0.4";
      try {
        var resp = await fetch("/api/scenario/" + scenarioId);
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        var data = await resp.json();
        if (imgEl) {
          imgEl.src = data.photo_data_url;
          imgEl.style.opacity = "1";
        }
        currentPhotoBase64 = data.photo_base64;
        var editor = document.getElementById("order-json-input");
        if (editor) editor.value = JSON.stringify(data.order, null, 2);
        if (autoVerify) {
          triggerInspection();
        }
      } catch (err) {
        if (imgEl) imgEl.style.opacity = "1";
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
      if (btn) {
        btn.disabled = false;
        btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> Run AI Pack Verification';
      }
      if (imgBox) imgBox.classList.remove("scanning");
    }

    function showError(msg) {
      var b = document.getElementById("verdict-banner");
      if (b) b.className = "verdict-banner STOP_FIX";
      var t = document.getElementById("verdict-title");
      if (t) t.textContent = "ERROR";
      var r = document.getElementById("verdict-reason");
      if (r) r.textContent = msg;
    }

    function renderResult(result) {
      var ordLabel = document.getElementById("order-id-label");
      if (ordLabel) ordLabel.innerText = result.order_id || "ORD-INSPECTED";

      var banner = document.getElementById("verdict-banner");
      if (banner) banner.className = "verdict-banner " + result.decision;

      var titles = {
        "SEAL": "SEAL FOR SHIPPING",
        "STOP_FIX": "STOP & FIX — DO NOT SHIP",
        "UNCERTAIN": "UNCERTAIN — RETAKE PHOTO"
      };
      var t = document.getElementById("verdict-title");
      if (t) t.innerText = titles[result.decision] || result.decision;
      var r = document.getElementById("verdict-reason");
      if (r) r.innerText = result.decision_reason;

      var confTag = document.getElementById("verdict-confidence");
      if (confTag) {
        confTag.innerText = (result.confidence || "high").toUpperCase();
        confTag.className = "confidence-tag conf-" + (result.confidence || "high");
      }

      var damageAlert = document.getElementById("damage-alert");
      if (damageAlert) {
        if (result.product_condition && result.product_condition.visible_damage) {
          damageAlert.style.display = "flex";
          document.getElementById("damage-notes").innerText = result.product_condition.notes;
        } else {
          damageAlert.style.display = "none";
        }
      }

      var missingBox = document.getElementById("missing-alert-box");
      if (missingBox) {
        if (result.missing_items && result.missing_items.length > 0) {
          missingBox.style.display = "flex";
          document.getElementById("missing-items-list").innerHTML = result.missing_items.map(function(m) {
            return "<strong>" + m.expected_qty + "x " + m.item_name + "</strong>: " + m.reason;
          }).join("<br>");
        } else {
          missingBox.style.display = "none";
        }
      }

      var extraBox = document.getElementById("extra-alert-box");
      if (extraBox) {
        if (result.extra_items && result.extra_items.length > 0) {
          extraBox.style.display = "flex";
          document.getElementById("extra-items-list").innerHTML = result.extra_items.map(function(e) {
            return "<strong>" + e.qty + "x " + e.item_name + "</strong>: " + e.notes;
          }).join("<br>");
        } else {
          extraBox.style.display = "none";
        }
      }

      var tbody = document.getElementById("matches-table-body");
      if (tbody) {
        if (result.matches && result.matches.length > 0) {
          tbody.innerHTML = result.matches.map(function(m) {
            return "<tr><td><strong>" + m.item_name + "</strong></td><td>" + m.expected_qty +
              "</td><td>" + m.detected_qty + "</td><td>" +
              (m.variant_match ? "Yes" : "<span style='color:#ef4444; font-weight:600;'>No</span>") +
              "</td><td><span class='" + (m.status === "PASS" ? "badge-pass" : "badge-fail") + "'>" + m.status + "</span></td></tr>";
          }).join("");
        } else {
          tbody.innerHTML = "<tr><td colspan='5' style='text-align:center;color:var(--text-muted);'>No match details</td></tr>";
        }
      }

      var evList = document.getElementById("evidence-list");
      if (evList) {
        evList.innerHTML = (result.evidence && result.evidence.length > 0)
          ? result.evidence.map(function(e) { return "<li>" + e + "</li>"; }).join("")
          : "<li>No specific visual evidence listed</li>";
      }

      var rawViewer = document.getElementById("raw-json-viewer");
      if (rawViewer) rawViewer.innerText = JSON.stringify(result, null, 2);
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

    /* Modals & Providers */
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
      closeSettingsModal();
      if (currentView === "inspector") {
        triggerInspection();
      }
    }

    function openAddPackageModal() {
      document.getElementById("add-package-modal").classList.add("open");
    }
    function closeAddPackageModal() {
      document.getElementById("add-package-modal").classList.remove("open");
    }
    function saveNewPackage() {
      var id = document.getElementById("add-pkg-id").value.trim();
      var name = document.getElementById("add-pkg-name").value.trim() || "Package Goods";
      var cat = document.getElementById("add-pkg-category").value;
      var loc = document.getElementById("add-pkg-location").value.trim() || "Warehouse Outbound";
      
      var tbody = document.getElementById("packages-table-body");
      var tr = document.createElement("tr");
      tr.onclick = function() { loadScenarioFromRow("example_1_correct_order"); };
      tr.innerHTML = '<td class="tracking-code">' + id + '</td>' +
        '<td class="pkg-name-text">' + name + '</td>' +
        '<td><span class="pill-category ' + cat + '">' + cat.charAt(0).toUpperCase() + cat.slice(1) + '</span></td>' +
        '<td><span class="pill-status pending">Pending</span></td>' +
        '<td>' + loc + '</td>' +
        '<td style="color: var(--text-muted);">Just now</td>' +
        '<td style="text-align: right;"><button class="btn-action-more">⋮</button></td>';
      tbody.insertBefore(tr, tbody.firstChild);
      closeAddPackageModal();
      switchView("inspector");
      selectScenario("example_1_correct_order");
    }

    /* Live Search in Recent Packages Table */
    function handleTableSearch(e) {
      var query = e.target.value.toLowerCase();
      var rows = document.querySelectorAll("#packages-table-body tr");
      rows.forEach(function(row) {
        var text = row.innerText.toLowerCase();
        row.style.display = text.includes(query) ? "" : "none";
      });
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
