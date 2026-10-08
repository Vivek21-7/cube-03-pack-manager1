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
      position: relative;
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
      position: relative;
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
      position: relative;
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
       Dashboard View Elements
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
      cursor: pointer;
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

    /* Charts Grid */
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
      transition: all 0.15s ease;
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
      transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      cursor: pointer;
    }
    .stacked-bar-pillar:hover {
      transform: scaleY(1.05);
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
      cursor: pointer;
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
      padding: 4px 6px;
      border-radius: 6px;
      transition: background 0.15s ease;
      cursor: pointer;
    }
    .breakdown-row:hover {
      background: #f8fafc;
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

    /* Recent Packages Table */
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
      transition: color 0.15s ease;
    }
    .table-view-all:hover { text-decoration: underline; color: var(--primary-hover); }
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

    /* Category & Status Badges */
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
      background: #e2e8f0;
      color: var(--text-main);
    }

    /* Row Action Dropdown Menu */
    .action-menu-dropdown {
      position: absolute;
      right: 40px;
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12);
      width: 160px;
      z-index: 100;
      display: none;
      flex-direction: column;
      padding: 6px;
    }
    .action-menu-dropdown.show { display: flex; }
    .action-menu-item {
      padding: 8px 12px;
      font-size: 0.82rem;
      color: var(--text-main);
      border-radius: 6px;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
      transition: background 0.1s ease;
    }
    .action-menu-item:hover { background: #f1f5f9; }
    .action-menu-item.danger { color: #ef4444; }
    .action-menu-item.danger:hover { background: #fef2f2; }

    /* Other Views Styles */
    .view-container {
      display: none;
      flex-direction: column;
      gap: 24px;
    }
    .view-container.active { display: flex; }

    .filter-tabs {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 16px;
    }
    .filter-tab {
      padding: 7px 14px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
      background: #f1f5f9;
      color: var(--text-muted);
      cursor: pointer;
      border: 1px solid transparent;
      transition: all 0.15s ease;
    }
    .filter-tab:hover { background: #e2e8f0; color: var(--text-main); }
    .filter-tab.active { background: #2563eb; color: #ffffff; }

    .category-cards-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 18px;
    }
    .category-box {
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: var(--radius);
      padding: 22px;
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      gap: 8px;
      transition: transform 0.15s ease;
      cursor: pointer;
    }
    .category-box:hover {
      transform: translateY(-2px);
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .cat-icon-lg {
      width: 44px;
      height: 44px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.4rem;
      margin-bottom: 4px;
    }
    .cat-name { font-size: 1rem; font-weight: 700; color: var(--text-main); }
    .cat-count { font-size: 0.84rem; color: var(--text-muted); }

    /* Modals */
    .modal-overlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.5);
      backdrop-filter: blur(4px);
      z-index: 1000;
      align-items: center;
      justify-content: center;
    }
    .modal-overlay.open { display: flex; }
    #pkg-details-modal .modal-card {
      max-width: 980px;
      width: 95%;
      max-height: 92vh;
      display: flex;
      flex-direction: column;
      padding: 22px 26px 16px;
      box-shadow: 0 25px 60px -12px rgba(15, 23, 42, 0.35);
    }
    #pkg-details-modal #modal-pkg-body {
      overflow-y: auto;
      max-height: calc(92vh - 140px);
      padding-right: 6px;
    }
    .modal-two-col {
      display: grid;
      grid-template-columns: 1fr 1.08fr;
      gap: 20px;
      align-items: start;
    }
    @media (max-width: 860px) {
      .modal-two-col {
        grid-template-columns: 1fr;
      }
    }
    .photo-viewport-card {
      background: #090d16;
      border: 1px solid #1e293b;
      border-radius: 12px;
      overflow: hidden;
      position: relative;
    }
    .photo-viewport-header {
      padding: 8px 12px;
      background: #0f172a;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #1e293b;
      font-size: 0.74rem;
      font-weight: 700;
      color: #94a3b8;
    }
    .photo-viewport-img-wrap {
      position: relative;
      width: 100%;
      height: 250px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: radial-gradient(circle at center, #111827 0%, #030712 100%);
      overflow: hidden;
    }
    .photo-viewport-img-wrap img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
      transition: opacity 0.25s ease;
    }
    .photo-viewport-img-wrap video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      background: #000;
    }
    .scan-laser-line {
      display: none;
      position: absolute;
      left: 0;
      right: 0;
      height: 3px;
      background: linear-gradient(90deg, transparent, #38bdf8, #60a5fa, transparent);
      box-shadow: 0 0 14px #38bdf8;
      animation: scanLaserAnim 1.6s ease-in-out infinite;
      z-index: 5;
    }
    @keyframes scanLaserAnim {
      0% { top: 0%; opacity: 0.8; }
      50% { top: 96%; opacity: 1; }
      100% { top: 0%; opacity: 0.8; }
    }
    .btn-analyze-ai {
      width: 100%;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      padding: 12px 20px;
      border-radius: 12px;
      background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
      color: #ffffff;
      font-size: 0.94rem;
      font-weight: 700;
      border: none;
      cursor: pointer;
      box-shadow: 0 4px 14px rgba(37, 99, 235, 0.35);
      transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .btn-analyze-ai:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45);
      filter: brightness(1.08);
    }
    .btn-analyze-ai:disabled {
      opacity: 0.65;
      cursor: not-allowed;
      transform: none;
    }
    .action-pill-btn {
      padding: 5px 11px;
      border-radius: 14px;
      font-size: 0.74rem;
      font-weight: 600;
      border: 1px solid #cbd5e1;
      background: #ffffff;
      color: #334155;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
    }
    .action-pill-btn:hover {
      background: #f1f5f9;
      border-color: #94a3b8;
    }
    .action-pill-btn.active {
      background: #eff6ff;
      border-color: #3b82f6;
      color: #1d4ed8;
      font-weight: 700;
    }
    .dropzone-active {
      border: 2px dashed #3b82f6 !important;
      background: rgba(59, 130, 246, 0.15) !important;
    }
        .modal-card {
      background: #ffffff;
      border-radius: 16px;
      padding: 28px;
      width: 90%;
      max-width: 460px;
      box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
    }
    .modal-header-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 20px;
    }
    .modal-heading { font-size: 1.2rem; font-weight: 700; color: var(--text-main); }
    .btn-close-modal {
      background: none;
      border: none;
      font-size: 1.3rem;
      color: var(--text-muted);
      cursor: pointer;
    }
    .form-field { margin-bottom: 16px; }
    .field-label { display: block; font-size: 0.82rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; }
    .field-input, .field-select {
      width: 100%;
      padding: 9px 12px;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      font-size: 0.88rem;
      color: var(--text-main);
      font-family: inherit;
      outline: none;
    }
    .field-input:focus, .field-select:focus { border-color: var(--primary); }
    .modal-footer {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 24px;
    }
    .btn-cancel {
      padding: 8px 16px;
      background: #f1f5f9;
      border: 1px solid var(--card-border);
      color: var(--text-main);
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-save {
      padding: 8px 18px;
      background: var(--primary);
      border: none;
      color: #ffffff;
      border-radius: 8px;
      font-weight: 600;
      cursor: pointer;
    }
    .btn-save:hover { background: var(--primary-hover); }

    /* Dropdown Menus (Notifications & User) */
    .popover-menu {
      position: absolute;
      top: 52px;
      right: 0;
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(15, 23, 42, 0.12);
      width: 290px;
      z-index: 100;
      display: none;
      flex-direction: column;
      overflow: hidden;
    }
    .popover-menu.open { display: flex; }
    .popover-header {
      padding: 14px 16px;
      font-size: 0.88rem;
      font-weight: 700;
      border-bottom: 1px solid #f1f5f9;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .popover-list { display: flex; flex-direction: column; max-height: 250px; overflow-y: auto; }
    .popover-item {
      padding: 12px 16px;
      border-bottom: 1px solid #f8fafc;
      font-size: 0.82rem;
      display: flex;
      flex-direction: column;
      gap: 3px;
      cursor: pointer;
      transition: background 0.15s ease;
    }
    .popover-item:hover { background: #f8fafc; }
    .popover-time { font-size: 0.72rem; color: var(--text-light); }

    /* Toast Notification */
    .toast-container {
      position: fixed;
      bottom: 24px;
      right: 28px;
      background: #0f172a;
      color: #ffffff;
      padding: 12px 20px;
      border-radius: 10px;
      font-size: 0.86rem;
      font-weight: 500;
      box-shadow: 0 10px 30px rgba(0,0,0,0.25);
      z-index: 2000;
      display: none;
      align-items: center;
      gap: 10px;
      animation: slideInToast 0.25s ease forwards;
    }
    @keyframes slideInToast {
      from { transform: translateY(20px); opacity: 0; }
      to { transform: translateY(0); opacity: 1; }
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
      <div class="brand-header" onclick="navigateTo('dashboard')">
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
        <li class="nav-item active" id="nav-dashboard" onclick="navigateTo('dashboard')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
          <span>Dashboard</span>
        </li>
        <li class="nav-item" id="nav-packages" onclick="navigateTo('packages')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path></svg>
          <span>Packages</span>
        </li>
        <li class="nav-item" id="nav-add-pkg" onclick="openAddPackageModal()">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="16"></line><line x1="8" y1="12" x2="16" y2="12"></line></svg>
          <span>Add Package</span>
        </li>
        <li class="nav-item" id="nav-categories" onclick="navigateTo('categories')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>
          <span>Categories</span>
        </li>
        <li class="nav-item" id="nav-inventory" onclick="navigateTo('inventory')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
          <span>Inventory</span>
        </li>
        <li class="nav-item" id="nav-reports" onclick="navigateTo('reports')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
          <span>Reports</span>
        </li>
        <li class="nav-item" id="nav-settings" onclick="navigateTo('settings')">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          <span>Settings</span>
        </li>
      </ul>
    </div>

    <!-- Sidebar Bottom: Vivek Profile -->
    <div class="sidebar-user" onclick="toggleUserDropdown(event)">
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
       Main Content Wrapper
       ====================================================================== -->
  <div class="main-wrapper" onclick="closeAllPopovers(event)">
    <!-- Topbar -->
    <header class="topbar">
      <div>
        <h1 class="page-title" id="view-title">Dashboard</h1>
        <p class="page-subtitle" id="view-subtitle">Welcome back, Vivek! Here's an overview of your packages.</p>
      </div>
      <div class="topbar-right">
        <!-- Open Box Photograph Quick Access Button -->
        <button class="btn-box-photo" onclick="openPackageDetails(0)" title="View Open Box Photograph for PKG001" style="display: flex; align-items: center; gap: 7px; padding: 8px 15px; border-radius: 20px; border: 1px solid #93c5fd; background: #eff6ff; color: #1d4ed8; font-size: 0.84rem; font-weight: 700; cursor: pointer; transition: all 0.15s ease; box-shadow: 0 1px 3px rgba(37,99,235,0.12);">
          <span style="font-size: 1rem;">📸</span>
          <span>Open Box Photograph</span>
        </button>

        <!-- Search bar -->
        <div class="search-box">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
          <input type="text" id="package-search-input" class="search-input" placeholder="Search packages..." oninput="handleTableSearch(event)">
        </div>
        <!-- Notification button -->
        <div class="icon-btn" onclick="toggleNotifications(event)" title="Notifications">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9"></path><path d="M13.73 21a2 2 0 0 1-3.46 0"></path></svg>
          <span class="badge-dot" id="notif-badge"></span>
        </div>
        <!-- Notifications Popover -->
        <div class="popover-menu" id="notif-popover">
          <div class="popover-header">
            <span>Notifications</span>
            <span style="font-size: 0.72rem; color: #2563eb; cursor: pointer;" onclick="clearNotifs()">Mark all read</span>
          </div>
          <div class="popover-list" id="notif-list">
            <div class="popover-item">
              <span style="font-weight: 600;">Package PKG001 Delivered</span>
              <span style="color: var(--text-muted);">Successfully received at Hyderabad warehouse hub.</span>
              <span class="popover-time">10 minutes ago</span>
            </div>
            <div class="popover-item">
              <span style="font-weight: 600;">Package PKG002 In Transit</span>
              <span style="color: var(--text-muted);">Departed Bengaluru sorting facility on route.</span>
              <span class="popover-time">1 hour ago</span>
            </div>
            <div class="popover-item">
              <span style="font-weight: 600;">Package PKG003 Pending</span>
              <span style="color: var(--text-muted);">Awaiting packaging clearance at Chennai.</span>
              <span class="popover-time">3 hours ago</span>
            </div>
          </div>
        </div>

        <!-- Vivek Avatar Topbar -->
        <div class="user-avatar-circle" style="width: 36px; height: 36px; font-size: 0.9rem; cursor: pointer;" onclick="toggleUserDropdown(event)">V</div>
        <!-- User Popover -->
        <div class="popover-menu" id="user-popover" style="width: 220px;">
          <div class="popover-header">
            <span>Vivek (Admin)</span>
          </div>
          <div class="popover-list">
            <div class="popover-item" onclick="navigateTo('settings')">⚙️ Account Settings</div>
            <div class="popover-item" onclick="showToast('Profile refreshed')">🔄 Refresh Profile</div>
            <div class="popover-item" onclick="showToast('Session is active')">🔒 Security Logs</div>
          </div>
        </div>
      </div>
    </header>

    <!-- Content Area -->
    <div class="content-area">
      <!-- ==================================================================
           VIEW: DASHBOARD (Matches User Screenshot 1:1)
           ================================================================== -->
      <div id="view-dashboard" class="view-container active">
        <!-- 4 KPI Stat Cards -->
        <div class="kpi-grid">
          <!-- Card 1: Total Packages -->
          <div class="kpi-card" onclick="filterByStatus('all')">
            <div class="kpi-top">
              <div class="kpi-icon-box blue">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg>
              </div>
              <div class="kpi-meta">
                <span class="kpi-title">Total Packages</span>
                <span class="kpi-value" id="kpi-val-total">120</span>
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
          <div class="kpi-card" onclick="filterByStatus('In Transit')">
            <div class="kpi-top">
              <div class="kpi-icon-box green">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><rect x="1" y="3" width="15" height="13"></rect><polygon points="16 8 20 8 23 11 23 16 16 16 16 8"></polygon><circle cx="5.5" cy="18.5" r="2.5"></circle><circle cx="18.5" cy="18.5" r="2.5"></circle></svg>
              </div>
              <div class="kpi-meta">
                <span class="kpi-title">In Transit</span>
                <span class="kpi-value" id="kpi-val-transit">28</span>
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
          <div class="kpi-card" onclick="filterByStatus('Delivered')">
            <div class="kpi-top">
              <div class="kpi-icon-box orange">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>
              </div>
              <div class="kpi-meta">
                <span class="kpi-title">Delivered</span>
                <span class="kpi-value" id="kpi-val-delivered">84</span>
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
          <div class="kpi-card" onclick="filterByStatus('Pending')">
            <div class="kpi-top">
              <div class="kpi-icon-box red">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
              </div>
              <div class="kpi-meta">
                <span class="kpi-title">Pending</span>
                <span class="kpi-value" id="kpi-val-pending">8</span>
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
              <select class="select-pill" id="timeframe-select" onchange="updateChartTimeframe(event)">
                <option value="7">Last 7 Days ⌵</option>
                <option value="30">Last 30 Days</option>
                <option value="month">This Month</option>
              </select>
            </div>

            <div class="barchart-container">
              <div class="barchart-gridlines">
                <div class="gridline-row"><span class="gridline-label">50</span></div>
                <div class="gridline-row"><span class="gridline-label">40</span></div>
                <div class="gridline-row"><span class="gridline-label">30</span></div>
                <div class="gridline-row"><span class="gridline-label">20</span></div>
                <div class="gridline-row"><span class="gridline-label">10</span></div>
                <div class="gridline-row"><span class="gridline-label">0</span></div>
              </div>

              <!-- Stacked Bars -->
              <div class="barchart-bars" id="barchart-bars-container">
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 58px;" onclick="showToast('Mon: 16 packages (8 Delivered, 6 Transit, 2 Pending)')">
                    <div class="bar-seg-blue" style="height: 50%;"></div>
                    <div class="bar-seg-green" style="height: 38%;"></div>
                    <div class="bar-seg-orange" style="height: 12%;"></div>
                  </div>
                  <span class="bar-day-label">Mon</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 86px;" onclick="showToast('Tue: 24 packages (10 Delivered, 8 Transit, 6 Pending)')">
                    <div class="bar-seg-blue" style="height: 45%;"></div>
                    <div class="bar-seg-green" style="height: 35%;"></div>
                    <div class="bar-seg-orange" style="height: 20%;"></div>
                  </div>
                  <span class="bar-day-label">Tue</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 82px;" onclick="showToast('Wed: 23 packages (11 Delivered, 7 Transit, 5 Pending)')">
                    <div class="bar-seg-blue" style="height: 48%;"></div>
                    <div class="bar-seg-green" style="height: 32%;"></div>
                    <div class="bar-seg-orange" style="height: 20%;"></div>
                  </div>
                  <span class="bar-day-label">Wed</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 98px;" onclick="showToast('Thu: 27 packages (12 Delivered, 10 Transit, 5 Pending)')">
                    <div class="bar-seg-blue" style="height: 45%;"></div>
                    <div class="bar-seg-green" style="height: 37%;"></div>
                    <div class="bar-seg-orange" style="height: 18%;"></div>
                  </div>
                  <span class="bar-day-label">Thu</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 152px;" onclick="showToast('Fri: 42 packages (24 Delivered, 12 Transit, 6 Pending)')">
                    <div class="bar-seg-blue" style="height: 57%;"></div>
                    <div class="bar-seg-green" style="height: 29%;"></div>
                    <div class="bar-seg-orange" style="height: 14%;"></div>
                  </div>
                  <span class="bar-day-label">Fri</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 82px;" onclick="showToast('Sat: 23 packages (11 Delivered, 7 Transit, 5 Pending)')">
                    <div class="bar-seg-blue" style="height: 45%;"></div>
                    <div class="bar-seg-green" style="height: 35%;"></div>
                    <div class="bar-seg-orange" style="height: 20%;"></div>
                  </div>
                  <span class="bar-day-label">Sat</span>
                </div>
                <div class="bar-column">
                  <div class="stacked-bar-pillar" style="height: 112px;" onclick="showToast('Sun: 31 packages (16 Delivered, 10 Transit, 5 Pending)')">
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
              <div class="legend-item" onclick="filterByStatus('Delivered')"><span class="legend-dot blue"></span> Delivered</div>
              <div class="legend-item" onclick="filterByStatus('In Transit')"><span class="legend-dot green"></span> In Transit</div>
              <div class="legend-item" onclick="filterByStatus('Pending')"><span class="legend-dot orange"></span> Pending</div>
            </div>
          </div>

          <!-- Right: Package Status Donut Chart -->
          <div class="chart-card">
            <div class="chart-card-header">
              <span class="chart-title">Package Status</span>
            </div>

            <div class="donut-content">
              <div class="donut-visual-box">
                <svg width="160" height="160" viewBox="0 0 100 100" style="transform: rotate(-90deg);">
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#f1f5f9" stroke-width="14"/>
                  <!-- Blue 70% -->
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#3b82f6" stroke-width="14"
                          stroke-dasharray="167.13 238.76" stroke-dashoffset="0"/>
                  <!-- Green 23% -->
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#10b981" stroke-width="14"
                          stroke-dasharray="54.91 238.76" stroke-dashoffset="-167.13"/>
                  <!-- Orange 7% -->
                  <circle cx="50" cy="50" r="38" fill="none" stroke="#f59e0b" stroke-width="14"
                          stroke-dasharray="16.71 238.76" stroke-dashoffset="-222.04"/>
                </svg>

                <div class="donut-center-text">
                  <span class="donut-total-num" id="donut-total-val">120</span>
                  <span class="donut-total-sub">Total</span>
                </div>
              </div>

              <!-- Breakdown list -->
              <div class="donut-breakdown-list">
                <div class="breakdown-row" onclick="filterByStatus('Delivered')">
                  <div class="breakdown-left">
                    <span class="legend-dot blue"></span> Delivered
                  </div>
                  <span class="breakdown-stat" id="donut-deliv-stat">84 (70%)</span>
                </div>
                <div class="breakdown-row" onclick="filterByStatus('In Transit')">
                  <div class="breakdown-left">
                    <span class="legend-dot green"></span> In Transit
                  </div>
                  <span class="breakdown-stat" id="donut-transit-stat">28 (23%)</span>
                </div>
                <div class="breakdown-row" onclick="filterByStatus('Pending')">
                  <div class="breakdown-left">
                    <span class="legend-dot orange"></span> Pending
                  </div>
                  <span class="breakdown-stat" id="donut-pending-stat">8 (7%)</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Recent Packages Table -->
        <div class="table-card">
          <div class="table-card-header">
            <span class="chart-title">Recent Packages <span style="font-size: 0.74rem; font-weight: 600; color: #2563eb; background: #eff6ff; border: 1px solid #bfdbfe; padding: 2px 10px; border-radius: 12px; margin-left: 8px; vertical-align: middle; display: inline-flex; align-items: center; gap: 4px;">📸 Click any package to view Open Box Photo</span></span>
            <span class="table-view-all" onclick="navigateTo('packages')">View All</span>
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
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- ==================================================================
           VIEW: PACKAGES (Full Management View)
           ================================================================== -->
      <div id="view-packages" class="view-container">
        <div class="table-card">
          <div class="table-card-header">
            <div>
              <span class="chart-title">All Shipments &amp; Packages</span>
              <p style="font-size: 0.82rem; color: var(--text-muted); margin-top: 2px;">Manage outbound tracking, destinations, and delivery statuses.</p>
            </div>
            <button class="btn-save" onclick="openAddPackageModal()">+ Add New Package</button>
          </div>

          <div class="filter-tabs">
            <button class="filter-tab active" onclick="applyStatusFilter(this, 'all')">All (120)</button>
            <button class="filter-tab" onclick="applyStatusFilter(this, 'Delivered')">Delivered (84)</button>
            <button class="filter-tab" onclick="applyStatusFilter(this, 'In Transit')">In Transit (28)</button>
            <button class="filter-tab" onclick="applyStatusFilter(this, 'Pending')">Pending (8)</button>
          </div>

          <table class="recent-table">
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
            <tbody id="full-packages-table-body">
              <!-- Rendered via JS -->
            </tbody>
          </table>
        </div>
      </div>

      <!-- ==================================================================
           VIEW: CATEGORIES
           ================================================================== -->
      <div id="view-categories" class="view-container">
        <div class="category-cards-grid">
          <div class="category-box" onclick="filterByCategory('Electronics')">
            <div class="cat-icon-lg" style="background: #dbeafe; color: #1d4ed8;">💻</div>
            <div class="cat-name">Electronics</div>
            <div class="cat-count">48 Packages &bull; 40% Volume</div>
          </div>
          <div class="category-box" onclick="filterByCategory('Books')">
            <div class="cat-icon-lg" style="background: #f3e8ff; color: #7e22ce;">📚</div>
            <div class="cat-name">Books &amp; Media</div>
            <div class="cat-count">24 Packages &bull; 20% Volume</div>
          </div>
          <div class="category-box" onclick="filterByCategory('Fashion')">
            <div class="cat-icon-lg" style="background: #ffe4e6; color: #be123c;">👗</div>
            <div class="cat-name">Fashion &amp; Apparel</div>
            <div class="cat-count">20 Packages &bull; 17% Volume</div>
          </div>
          <div class="category-box" onclick="filterByCategory('Home')">
            <div class="cat-icon-lg" style="background: #ccfbf1; color: #0f766e;">🏠</div>
            <div class="cat-name">Home Essentials</div>
            <div class="cat-count">16 Packages &bull; 13% Volume</div>
          </div>
          <div class="category-box" onclick="filterByCategory('Grocery')">
            <div class="cat-icon-lg" style="background: #ffedd5; color: #c2410c;">🛒</div>
            <div class="cat-name">Grocery Items</div>
            <div class="cat-count">12 Packages &bull; 10% Volume</div>
          </div>
        </div>
      </div>

      <!-- ==================================================================
           VIEW: INVENTORY
           ================================================================== -->
      <div id="view-inventory" class="view-container">
        <div class="table-card">
          <div class="table-card-header">
            <span class="chart-title">Warehouse Stock Inventory</span>
            <button class="btn-save" onclick="showToast('Inventory synchronizing with warehouse floor...')">Sync Stock</button>
          </div>
          <table class="recent-table">
            <thead>
              <tr>
                <th>SKU</th>
                <th>Item Description</th>
                <th>Category</th>
                <th>In Stock</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr><td class="tracking-code">SKU-ELEC-01</td><td>Wireless Noise-Canceling Headphones</td><td>Electronics</td><td>342 units</td><td><span class="pill-status delivered">In Stock</span></td></tr>
              <tr><td class="tracking-code">SKU-BOOK-04</td><td>Logistics Automation Guide (Hardcover)</td><td>Books</td><td>189 units</td><td><span class="pill-status delivered">In Stock</span></td></tr>
              <tr><td class="tracking-code">SKU-FASH-09</td><td>Cotton Crewneck T-Shirt (Blue)</td><td>Fashion</td><td>28 units</td><td><span class="pill-status intransit">Low Stock</span></td></tr>
              <tr><td class="tracking-code">SKU-HOME-12</td><td>Aroma Diffuser with LED Lamp</td><td>Home</td><td>120 units</td><td><span class="pill-status delivered">In Stock</span></td></tr>
              <tr><td class="tracking-code">SKU-GROC-03</td><td>Organic Ground Arabica Coffee (500g)</td><td>Grocery</td><td>4 units</td><td><span class="pill-status pending">Critical</span></td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <!-- ==================================================================
           VIEW: REPORTS
           ================================================================== -->
      <div id="view-reports" class="view-container">
        <div class="kpi-grid">
          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-icon-box green">📈</div>
              <div class="kpi-meta"><span class="kpi-title">On-Time Delivery</span><span class="kpi-value">98.4%</span></div>
            </div>
            <div class="kpi-bottom"><span class="kpi-trend up">↑ 2.1% this week</span></div>
          </div>
          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-icon-box blue">⏱️</div>
              <div class="kpi-meta"><span class="kpi-title">Avg Packing Speed</span><span class="kpi-value">42 sec</span></div>
            </div>
            <div class="kpi-bottom"><span class="kpi-trend up">↓ 5s faster</span></div>
          </div>
          <div class="kpi-card">
            <div class="kpi-top">
              <div class="kpi-icon-box orange">🎯</div>
              <div class="kpi-meta"><span class="kpi-title">Accuracy Rate</span><span class="kpi-value">99.8%</span></div>
            </div>
            <div class="kpi-bottom"><span class="kpi-trend up">0 packaging errors</span></div>
          </div>
        </div>

        <div class="table-card">
          <div class="table-card-header">
            <span class="chart-title">Export Weekly Summaries</span>
            <button class="btn-save" onclick="exportDataCsv()">Export as CSV</button>
          </div>
          <p style="font-size: 0.86rem; color: var(--text-muted); line-height: 1.6;">
            All 120 packages tracked across 5 metropolitan distribution hubs (Hyderabad, Bengaluru, Chennai, Mumbai, Delhi).
            Detailed audit trails available for download.
          </p>
        </div>
      </div>

      <!-- ==================================================================
           VIEW: SETTINGS
           ================================================================== -->
      <div id="view-settings" class="view-container">
        <div class="table-card" style="max-width: 600px;">
          <div class="table-card-header">
            <span class="chart-title">Administrator Settings</span>
          </div>
          <div class="form-field">
            <label class="field-label">Administrator Name</label>
            <input type="text" class="field-input" value="Vivek" id="cfg-user-name">
          </div>
          <div class="form-field">
            <label class="field-label">Warehouse Terminal</label>
            <input type="text" class="field-input" value="CUBE Pack Station #03" id="cfg-station">
          </div>
          <div class="form-field">
            <label class="field-label">Default Tracking Prefix</label>
            <input type="text" class="field-input" value="PKG" id="cfg-prefix">
          </div>
          <div class="form-field">
            <label class="field-label">Notifications</label>
            <select class="field-select">
              <option>Push notifications enabled for all package status changes</option>
              <option>Only critical / pending alerts</option>
              <option>Mute alerts</option>
            </select>
          </div>
          <button class="btn-save" style="margin-top: 10px;" onclick="showToast('Settings saved successfully!')">Save Preferences</button>
        </div>
      </div>
    </div>
  </div>

  <!-- ======================================================================
       Action Menu Popover for Table Rows
       ====================================================================== -->
  <div class="action-menu-dropdown" id="row-action-menu">
    <div class="action-menu-item" onclick="openPackageDetails(activePkgIndex)">🔍 View Details</div>
    <div class="action-menu-item" onclick="changeStatus(activePkgIndex, 'Delivered')">✅ Mark Delivered</div>
    <div class="action-menu-item" onclick="changeStatus(activePkgIndex, 'In Transit')">🚚 Mark In Transit</div>
    <div class="action-menu-item" onclick="changeStatus(activePkgIndex, 'Pending')">⏳ Mark Pending</div>
    <div class="action-menu-item danger" onclick="deletePackage(activePkgIndex)">🗑️ Delete Package</div>
  </div>

  <!-- ======================================================================
       Package Details Modal
       ====================================================================== -->
  <div class="modal-overlay" id="pkg-details-modal">
    <div class="modal-card">
      <div class="modal-header-row" style="margin-bottom: 14px;">
        <div>
          <span class="modal-heading" id="modal-pkg-title">Outbound Packaging Verification Station</span>
          <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 2px;">Inspect product photos, capture live packing station camera feeds, and verify with Multimodal AI Agent.</p>
        </div>
        <button class="btn-close-modal" onclick="closeDetailsModal()">&times;</button>
      </div>

      <div id="modal-pkg-body" style="font-size: 0.88rem; color: var(--text-main);">
        <!-- Injected dynamically by openPackageDetails -->
      </div>

      <div class="modal-footer" style="margin-top: 14px; padding-top: 12px; border-top: 1px solid #e2e8f0; display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.76rem; color: var(--text-muted);">Station terminal connected to Outbound AI Agent &bull; 24-bit RGB inspection</span>
        <button class="btn-cancel" onclick="closeDetailsModal()">Close Terminal</button>
      </div>
    </div>
  </div>

  <!-- ======================================================================
       Add Package Modal
       ====================================================================== -->
  <div class="modal-overlay" id="add-pkg-modal">
    <div class="modal-card">
      <div class="modal-header-row">
        <span class="modal-heading">Add New Package</span>
        <button class="btn-close-modal" onclick="closeAddPackageModal()">&times;</button>
      </div>
      <div class="form-field">
        <label class="field-label">Tracking ID</label>
        <input type="text" class="field-input" id="new-pkg-id" value="PKG006">
      </div>
      <div class="form-field">
        <label class="field-label">Package Name</label>
        <input type="text" class="field-input" id="new-pkg-name" placeholder="e.g. Wireless Audio Gear">
      </div>
      <div class="form-field">
        <label class="field-label">Category</label>
        <select class="field-select" id="new-pkg-category">
          <option value="Electronics">Electronics</option>
          <option value="Books">Books</option>
          <option value="Fashion">Fashion</option>
          <option value="Home">Home</option>
          <option value="Grocery">Grocery</option>
        </select>
      </div>
      <div class="form-field">
        <label class="field-label">Status</label>
        <select class="field-select" id="new-pkg-status">
          <option value="Pending">Pending</option>
          <option value="In Transit">In Transit</option>
          <option value="Delivered">Delivered</option>
        </select>
      </div>
      <div class="form-field">
        <label class="field-label">Destination Location</label>
        <input type="text" class="field-input" id="new-pkg-location" placeholder="e.g. Kolkata, IN">
      </div>
      <div class="modal-footer">
        <button class="btn-cancel" onclick="closeAddPackageModal()">Cancel</button>
        <button class="btn-save" onclick="submitAddPackage()">Save Package</button>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div class="toast-container" id="toast-msg">
    <span id="toast-text">Action completed successfully</span>
  </div>

  <!-- ======================================================================
       JavaScript Logic: 100% Fully Working
       ====================================================================== -->
  <script>
    // Master Reactive State
    var packagesData = [
      { id: "PKG009", name: "Action Camera 4K", category: "Electronics", status: "Pending", location: "Bengaluru Hub", date: "Oct 5, 2026", scenarioId: "example_7_electronics_order", verdict: "SEAL", verdictText: "Electronic hardware device verified against manifest. Packaging intact." },
      { id: "PKG001", name: "Electronics Items", category: "Electronics", status: "Delivered", location: "Hyderabad", date: "Oct 5, 2026", scenarioId: "example_7_electronics_order", verdict: "SEAL", verdictText: "Electronic device verified. Packaging undamaged." },
      { id: "PKG002", name: "Books Parcel", category: "Books", status: "In Transit", location: "Bengaluru", date: "Oct 5, 2026", scenarioId: "example_2_wrong_item", verdict: "STOP_FIX", verdictText: "Variant mismatch: Cap color detected RED, expected BLUE. Packaging integrity compromised." },
      { id: "PKG003", name: "Clothes Package", category: "Fashion", status: "Pending", location: "Chennai", date: "Oct 4, 2026", scenarioId: "example_3_missing_item", verdict: "STOP_FIX", verdictText: "Missing item: 1x User Manual missing from box contents." },
      { id: "PKG004", name: "Home Essentials", category: "Home", status: "Delivered", location: "Mumbai", date: "Oct 4, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "All items verified against manifest. Box sealed for dispatch." },
      { id: "PKG005", name: "Grocery Items", category: "Grocery", status: "In Transit", location: "Delhi", date: "Oct 3, 2026", scenarioId: "example_4_extra_item", verdict: "STOP_FIX", verdictText: "Unauthorized extra item: 1x Red Scarf detected in box not present on order." },
      { id: "PKG006", name: "Camera Lens Kit", category: "Electronics", status: "Delivered", location: "Hyderabad", date: "Oct 2, 2026", scenarioId: "example_7_electronics_order", verdict: "SEAL", verdictText: "Verified 100% item match." },
      { id: "PKG007", name: "Designer Shoes", category: "Fashion", status: "Delivered", location: "Pune", date: "Oct 2, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "Verified 100% item match." },
      { id: "PKG008", name: "Kitchen Blender", category: "Home", status: "In Transit", location: "Ahmedabad", date: "Oct 1, 2026", scenarioId: "example_6_damaged_goods", verdict: "STOP_FIX", verdictText: "Packaging damage detected: Physical compression on box carton." }
    ];

    var activePkgIndex = 0;
    var currentView = "dashboard";

    /* Global Interactive Inspection State */
    var currentInspectionPkg = null;
    var currentInspectionIdx = 0;
    var currentInspectionScenario = "example_1_correct_order";
    var currentInspectionPhotoB64 = "";
    var currentInspectionPhotoUrl = "";
    var currentInspectionOrder = null;
    var liveMediaStream = null;

    function init() {
      renderTables();
    }

    /* Navigation between Sidebar Views */
    function navigateTo(viewName) {
      currentView = viewName;
      var views = ["dashboard", "packages", "categories", "inventory", "reports", "settings"];
      views.forEach(function(v) {
        var el = document.getElementById("view-" + v);
        if (el) el.classList.remove("active");
        var nav = document.getElementById("nav-" + v);
        if (nav) nav.classList.remove("active");
      });

      var targetView = document.getElementById("view-" + viewName);
      if (targetView) targetView.classList.add("active");
      var targetNav = document.getElementById("nav-" + viewName);
      if (targetNav) targetNav.classList.add("active");

      var titles = {
        dashboard: { title: "Dashboard", sub: "Welcome back, Vivek! Here's an overview of your packages." },
        packages: { title: "Packages Management", sub: "Inspect, track, and dispatch your outbound parcels." },
        categories: { title: "Package Categories", sub: "Distribution of shipments grouped by product classification." },
        inventory: { title: "Warehouse Inventory", sub: "Live stock counts across warehouse fulfillment stations." },
        reports: { title: "Logistics Analytics & Reports", sub: "Performance KPIs, packing speed, and on-time fulfillment rates." },
        settings: { title: "System & Station Settings", sub: "Configure preferences, terminal parameters, and credentials." }
      };

      var t = titles[viewName] || titles["dashboard"];
      var elTitle = document.getElementById("view-title");
      if (elTitle) elTitle.innerText = t.title;
      var elSub = document.getElementById("view-subtitle");
      if (elSub) elSub.innerText = t.sub;

      closeAllPopovers();
    }

    /* Render Tables */
    function renderTables(filterStatus, filterCategory, searchQuery) {
      var filtered = packagesData.slice();

      if (filterStatus && filterStatus !== "all") {
        filtered = filtered.filter(function(p) { return p.status === filterStatus; });
      }
      if (filterCategory) {
        filtered = filtered.filter(function(p) { return p.category.toLowerCase() === filterCategory.toLowerCase(); });
      }
      if (searchQuery) {
        filtered = filtered.filter(function(p) {
          var text = (p.id + " " + p.name + " " + p.category + " " + p.status + " " + p.location).toLowerCase();
          return text.includes(searchQuery.toLowerCase());
        });
      }

      var dashTbody = document.getElementById("packages-table-body");
      if (dashTbody) {
        var dashRows = filtered.slice(0, 5);
        dashTbody.innerHTML = dashRows.map(function(p, idx) {
          return createRowHtml(p, idx);
        }).join("");
      }

      var fullTbody = document.getElementById("full-packages-table-body");
      if (fullTbody) {
        fullTbody.innerHTML = filtered.map(function(p, idx) {
          return createRowHtml(p, idx);
        }).join("");
      }
    }

    function createRowHtml(p, idx) {
      var catClass = p.category.toLowerCase();
      var statClass = p.status.toLowerCase().replace(/\s+/g, '');
      return '<tr onclick="openPackageDetails(' + idx + ')" title="Click to view Open Box Photograph & AI Verification">' +
        '<td class="tracking-code">' + p.id + '</td>' +
        '<td class="pkg-name-text">' + p.name + ' <span title="Click to view Open Box Photograph" style="cursor:pointer;font-size:0.9rem;margin-left:4px;">📸</span></td>' +
        '<td><span class="pill-category ' + catClass + '">' + p.category + '</span></td>' +
        '<td><span class="pill-status ' + statClass + '">' + p.status + '</span></td>' +
        '<td>' + p.location + '</td>' +
        '<td style="color: var(--text-muted);">' + p.date + '</td>' +
        '<td style="text-align: right;">' +
          '<button class="btn-action-more" onclick="openRowActionMenu(event, ' + idx + ')">⋮</button>' +
        '</td>' +
      '</tr>';
    }

    /* Modal Form for Add Package */
    function openAddPackageModal() {
      var idField = document.getElementById("new-pkg-id");
      if (idField) idField.value = "PKG0" + (packagesData.length + 10);
      var nameField = document.getElementById("new-pkg-name");
      if (nameField) nameField.value = "";
      var locField = document.getElementById("new-pkg-location");
      if (locField) locField.value = "";
      var m = document.getElementById("add-pkg-modal");
      if (m) m.classList.add("open");
    }

    function closeAddPackageModal() {
      var m = document.getElementById("add-pkg-modal");
      if (m) m.classList.remove("open");
    }

    function submitAddPackage() {
      var id = document.getElementById("new-pkg-id").value.trim() || ("PKG0" + (packagesData.length + 10));
      var name = document.getElementById("new-pkg-name").value.trim() || "General Goods";
      var cat = document.getElementById("new-pkg-category").value;
      var stat = document.getElementById("new-pkg-status").value;
      var loc = document.getElementById("new-pkg-location").value.trim() || "Bengaluru Hub";

      packagesData.unshift({
        id: id,
        name: name,
        category: cat,
        status: stat,
        location: loc,
        date: "Oct 5, 2026",
        scenarioId: "example_1_correct_order",
        verdict: "SEAL",
        verdictText: "Verified outbound packing complete. Station Cam RGB feed inspected against manifest."
      });

      var totalEl = document.getElementById("kpi-val-total");
      if (totalEl) totalEl.innerText = parseInt(totalEl.innerText) + 1;

      renderTables();
      closeAddPackageModal();
      showToast("Package " + id + " registered successfully!");
    }

    /* Row Action Dropdown Menu */
    function openRowActionMenu(event, idx) {
      event.stopPropagation();
      activePkgIndex = idx;
      var menu = document.getElementById("row-action-menu");
      if (!menu) return;
      var btnRect = event.currentTarget.getBoundingClientRect();
      menu.style.top = (btnRect.bottom + window.scrollY + 4) + "px";
      menu.style.left = (btnRect.right + window.scrollX - 160) + "px";
      menu.classList.add("open");
    }

    function closeAllPopovers(event) {
      var menu = document.getElementById("row-action-menu");
      if (menu && (!event || !event.target.closest(".btn-action-more"))) menu.classList.remove("open");
      var notif = document.getElementById("notif-popover");
      if (notif && (!event || !event.target.closest(".icon-btn"))) notif.classList.remove("open");
      var userPop = document.getElementById("user-popover");
      if (userPop && (!event || !event.target.closest(".user-avatar-circle") && !event.target.closest(".sidebar-user"))) userPop.classList.remove("open");
    }

    function changeStatus(idx, newStatus) {
      if (packagesData[idx]) {
        packagesData[idx].status = newStatus;
        renderTables();
        showToast(packagesData[idx].id + " status changed to " + newStatus);
      }
      closeAllPopovers();
    }

    function deletePackage(idx) {
      if (packagesData[idx]) {
        var id = packagesData[idx].id;
        packagesData.splice(idx, 1);
        renderTables();
        showToast("Package " + id + " deleted");
      }
      closeAllPopovers();
    }

    /* =========================================================================
       PACKAGE INSPECTION MODAL:
       - Open Box Photograph Viewer
       - Live Capture Camera Feed (Webcam)
       - Allow Select Files from Local Folder
       - Multimodal AI Agent Analyzer
       ========================================================================= */

    async function openPackageDetails(idx) {
      if (idx === undefined || idx === null || idx < 0 || idx >= packagesData.length) {
        idx = 0;
      }
      var p = packagesData[idx];
      if (!p) return;
      currentInspectionPkg = p;
      currentInspectionIdx = idx;
      currentInspectionScenario = p.scenarioId || "example_1_correct_order";

      var titleEl = document.getElementById("modal-pkg-title");
      if (titleEl) {
        titleEl.innerText = p.id + " — " + p.name + " (" + p.category + ")";
      }
      var body = document.getElementById("modal-pkg-body");

      body.innerHTML = `
        <div class="modal-two-col">
          <!-- LEFT COLUMN: Product Picture, Live Camera, Folder Files, Manifest -->
          <div>
            <!-- Main Photo & Live Camera Viewport -->
            <div class="photo-viewport-card" id="photo-viewport-container">
              <div class="photo-viewport-header">
                <span id="cam-status-label">📸 Station Cam #03 (24-bit RGB)</span>
                <span id="cam-status-badge" style="display:flex;align-items:center;gap:5px;color:#10b981;">
                  <span style="width:7px;height:7px;border-radius:50%;background:#10b981;display:inline-block;"></span> READY
                </span>
              </div>
              <div class="photo-viewport-img-wrap" id="dropzone-area">
                <div class="scan-laser-line" id="scan-laser-line"></div>
                <!-- Normal Photo Image -->
                <img id="details-box-photo" src="" alt="Open Box Photograph" style="opacity:0;">
                <div id="bbox-overlay-container" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;z-index:20;"></div>
                <!-- Live Video Element for Camera Feed -->
                <video id="live-camera-feed" autoplay playsinline style="display:none;"></video>
                <div id="details-img-loader" style="position: absolute; color: #94a3b8; font-size: 0.85rem; font-weight: 500;">Loading product photo...</div>
              </div>
            </div>

            <!-- Input Controls: Live Camera & Folder Files -->
            <div style="margin-top: 10px; display: flex; gap: 8px; flex-wrap: wrap;">
              <!-- 1. Live Camera Button -->
              <button class="action-pill-btn" id="btn-toggle-camera" onclick="toggleLiveCamera()">
                <span>📹</span>
                <span id="btn-toggle-cam-text">Open Live Camera</span>
              </button>

              <!-- 2. Capture Photo Button (Visible when camera active) -->
              <button class="action-pill-btn" id="btn-snap-camera" onclick="captureLiveSnapshot()" style="display:none; background:#10b981; color:#ffffff; border-color:#059669; font-weight:700;">
                <span>📸</span>
                <span>Capture Photo</span>
              </button>

              <!-- 3. Local Folder File Picker Button -->
              <label class="action-pill-btn" style="cursor: pointer; background: #f8fafc;">
                <span>📁</span>
                <span>Select from Folder</span>
                <input type="file" id="local-file-input" accept="image/*" style="display:none;" onchange="handleFolderFileSelect(event)">
              </label>

              <!-- Drop hint -->
              <span style="font-size: 0.72rem; color: var(--text-muted); align-self: center; margin-left: auto;">or drag &amp; drop image</span>
            </div>

            <!-- Preset Scenarios -->
            <div style="margin-top: 10px;">
              <div style="font-size:0.73rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-bottom:6px;">Preset Inspection Scenarios:</div>
              <div style="display:flex;flex-wrap:wrap;gap:6px;" id="scenario-pill-container">
                <button class="action-pill-btn ${currentInspectionScenario === 'example_7_electronics_order' ? 'active' : ''}" id="btn-scen-example_7_electronics_order" onclick="switchInspectionScenario('example_7_electronics_order')">📷 Electronics Order</button>
                <button class="action-pill-btn ${currentInspectionScenario === 'example_1_correct_order' ? 'active' : ''}" id="btn-scen-example_1_correct_order" onclick="switchInspectionScenario('example_1_correct_order')">👕 Apparel Order (T-Shirt &amp; Cap)</button>
                <button class="action-pill-btn" id="btn-scen-example_2_wrong_item" onclick="switchInspectionScenario('example_2_wrong_item')">❌ Wrong Color</button>
                <button class="action-pill-btn" id="btn-scen-example_3_missing_item" onclick="switchInspectionScenario('example_3_missing_item')">❌ Missing Item</button>
                <button class="action-pill-btn" id="btn-scen-example_4_extra_item" onclick="switchInspectionScenario('example_4_extra_item')">❌ Extra Item</button>
                <button class="action-pill-btn" id="btn-scen-example_6_damaged_goods" onclick="switchInspectionScenario('example_6_damaged_goods')">⚠️ Damaged Box</button>
              </div>
            </div>

            <!-- Expected Manifest Card -->
            <div style="margin-top: 12px; padding: 12px; background: #f8fafc; border-radius: 10px; border: 1px solid #e2e8f0;">
              <div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;">
                <span>📋 Expected Order Manifest</span>
                <span id="manifest-order-id" style="font-family:'JetBrains Mono',monospace;color:#2563eb;font-weight:700;">ORD-2024-001</span>
              </div>
              <div id="manifest-items-list" style="font-size:0.82rem;color:var(--text-main);line-height:1.5;">
                Loading manifest items...
              </div>

              <!-- Quick Target Customizer for Live Camera & Folder Uploads -->
              <div style="margin-top: 8px; padding-top: 8px; border-top: 1px dashed #cbd5e1; font-size: 0.74rem;">
                <span style="font-weight: 700; color: #475569;">Target Item in Camera:</span>
                <div style="display: flex; gap: 6px; margin-top: 4px;">
                  <input type="text" id="custom-target-name" placeholder="e.g. Action Camera, Baseball Cap, Phone" value="${p.category === 'Electronics' ? 'Action Camera' : 'Blue Baseball Cap'}" style="flex: 1; padding: 4px 8px; font-size: 0.76rem; border: 1px solid #cbd5e1; border-radius: 6px;" oninput="updateManifestTarget()">
                  <select id="custom-target-color" style="padding: 4px 6px; font-size: 0.76rem; border: 1px solid #cbd5e1; border-radius: 6px;" onchange="updateManifestTarget()">
                    <option value="black" ${p.category === 'Electronics' ? 'selected' : ''}>Black / Neutral</option>
                    <option value="blue" ${p.category !== 'Electronics' ? 'selected' : ''}>Blue Color</option>
                    <option value="red">Red Color</option>
                    <option value="standard">Any / Standard</option>
                  </select>
                </div>
              </div>
            </div>
          </div>

          <!-- RIGHT COLUMN: Package Info, Action Button & Real-time AI Agent Report -->
          <div>
            <!-- Shipment Badges -->
            <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:12px;font-size:0.8rem;">
              <div><strong>Tracking ID:</strong> <span class="tracking-code">${p.id}</span></div>
              <div><strong>Status:</strong> <span class="pill-status ${p.status.toLowerCase().replace(/\s+/g,'')}" id="detail-pill-status">${p.status}</span></div>
              <div><strong>Hub:</strong> ${p.location}</div>
            </div>

            <!-- Vision AI Engine Selector -->
            <div style="margin-bottom: 10px; padding: 8px 12px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-size:0.73rem;font-weight:700;color:var(--text-muted);text-transform:uppercase;">Vision AI Engine:</span>
                <select id="select-ai-engine" onchange="toggleAiEngine(this.value)" style="font-size:0.76rem;font-weight:600;padding:3px 8px;border:1px solid #cbd5e1;border-radius:6px;background:#ffffff;color:#1e293b;">
                  <option value="simulation" selected>⚡ Smart Computer Vision (Offline)</option>
                  <option value="gemini">✨ Google Gemini 2.0 Flash Vision</option>
                  <option value="openai">🧠 OpenAI GPT-4o Vision</option>
                </select>
              </div>
              <div id="api-key-container" style="display:block;margin-top:8px;">
                <div id="offline-engine-badge" style="font-size:0.72rem;color:#047857;background:#ecfdf5;padding:6px 10px;border-radius:6px;border:1px solid #a7f3d0;">
                  ⚡ <strong>Offline Engine Active:</strong> Onboard 24-bit RGB Computer Vision inspecting physical contours, colors, and SKU brand signatures. No API key required.
                </div>
                <div id="cloud-api-inputs" style="display:none;margin-top:4px;">
                  <input type="password" id="input-api-key" placeholder="Paste your API Key here..." style="width:100%;padding:5px 8px;font-size:0.76rem;border:1px solid #93c5fd;border-radius:6px;" oninput="saveApiKey(this.value)">
                  <div style="font-size:0.7rem;color:#64748b;margin-top:3px;">Stored in browser. Get free Gemini key at <a href="https://aistudio.google.com/app/apikey" target="_blank" style="color:#2563eb;text-decoration:underline;">aistudio.google.com</a>.</div>
                </div>
              </div>
            </div>

            <!-- Primary Action Button: Analyze with AI Agent -->
            <button class="btn-analyze-ai" id="btn-run-analysis" onclick="runAIAgentAnalysis()">
              <span style="font-size:1.15rem;">⚡</span>
              <span id="btn-analyze-text">Analyze with AI Agent</span>
            </button>

            <!-- Dynamic AI Analysis Container -->
            <div id="ai-analysis-results" style="margin-top: 12px;">
              <div style="padding: 20px 16px; text-align: center; background: #f8fafc; border-radius: 12px; border: 1px dashed #cbd5e1;">
                <div style="font-size: 1.8rem; margin-bottom: 6px;">🤖</div>
                <div style="font-weight: 700; color: #1e293b; font-size: 0.92rem;">Ready for Inspection</div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px; line-height: 1.4;">
                  Click <strong>"Analyze with AI Agent"</strong> to verify the open box photograph against the order manifest.
                </div>
              </div>
            </div>
          </div>
        </div>
      `;

      setupDropzone();
      document.getElementById("pkg-details-modal").classList.add("open");
      await loadInspectionScenario(currentInspectionScenario);
    }

    function closeDetailsModal() {
      stopLiveCamera();
      var m = document.getElementById("pkg-details-modal");
      if (m) m.classList.remove("open");
    }

    /* =========================================================================
       Live Camera Capture (Webcam Feed)
       ========================================================================= */

    async function toggleLiveCamera() {
      if (liveMediaStream) {
        stopLiveCamera();
      } else {
        await startLiveCamera();
      }
    }

    async function startLiveCamera() {
      var video = document.getElementById("live-camera-feed");
      var img = document.getElementById("details-box-photo");
      var btnText = document.getElementById("btn-toggle-cam-text");
      var snapBtn = document.getElementById("btn-snap-camera");
      var statusLabel = document.getElementById("cam-status-label");
      var statusBadge = document.getElementById("cam-status-badge");

      try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          throw new Error("Webcam access not supported in this browser environment.");
        }
        var stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" },
          audio: false
        });

        liveMediaStream = stream;
        video.srcObject = stream;
        video.style.display = "block";
        if (img) img.style.display = "none";

        if (btnText) btnText.innerText = "Stop Camera";
        if (snapBtn) snapBtn.style.display = "inline-flex";
        if (statusLabel) statusLabel.innerText = "📹 Live Camera Feed (Webcam)";
        if (statusBadge) {
          statusBadge.style.color = "#ef4444";
          statusBadge.innerHTML = '<span style="width:7px;height:7px;border-radius:50%;background:#ef4444;display:inline-block;animation:pulse 1s infinite;"></span> LIVE STREAM';
        }

        showToast("Live camera stream started");
      } catch (err) {
        console.warn("Camera access error:", err);
        showToast("Camera not accessible: " + err.message + ". Please select file from folder.");
      }
    }

    function stopLiveCamera() {
      if (liveMediaStream) {
        liveMediaStream.getTracks().forEach(function(t) { t.stop(); });
        liveMediaStream = null;
      }
      var video = document.getElementById("live-camera-feed");
      var img = document.getElementById("details-box-photo");
      var btnText = document.getElementById("btn-toggle-cam-text");
      var snapBtn = document.getElementById("btn-snap-camera");
      var statusLabel = document.getElementById("cam-status-label");
      var statusBadge = document.getElementById("cam-status-badge");

      if (video) video.style.display = "none";
      if (img) img.style.display = "block";
      if (btnText) btnText.innerText = "Open Live Camera";
      if (snapBtn) snapBtn.style.display = "none";
      if (statusLabel) statusLabel.innerText = "📸 Station Cam #03 (24-bit RGB)";
      if (statusBadge) {
        statusBadge.style.color = "#10b981";
        statusBadge.innerHTML = '<span style="width:7px;height:7px;border-radius:50%;background:#10b981;display:inline-block;"></span> READY';
      }
    }

    function captureLiveSnapshot() {
      var video = document.getElementById("live-camera-feed");
      if (!video || !liveMediaStream) return;

      var canvas = document.createElement("canvas");
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      var ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      var dataUrl = canvas.toDataURL("image/png");
      currentInspectionPhotoUrl = dataUrl;
      currentInspectionPhotoB64 = dataUrl.split(",")[1];

      // Stop camera and show captured image
      stopLiveCamera();

      var img = document.getElementById("details-box-photo");
      if (img) {
        img.src = dataUrl;
        img.style.display = "block";
        var boxCont = document.getElementById("bbox-overlay-container");
        if (boxCont) boxCont.innerHTML = "";
        img.style.opacity = "1";
      }

      // De-select scenario pills
      var pills = document.querySelectorAll(".action-pill-btn");
      pills.forEach(function(b) { b.classList.remove("active"); });

      var resEl = document.getElementById("ai-analysis-results");
      if (resEl) {
        resEl.innerHTML = `
          <div style="padding: 16px; text-align: center; background: #ecfdf5; border-radius: 12px; border: 1px solid #a7f3d0;">
            <div style="font-weight: 700; color: #065f46; font-size: 0.9rem;">📸 Live Snapshot Captured!</div>
            <div style="font-size: 0.8rem; color: #047857; margin-top: 4px;">Click <strong>"Analyze with AI Agent"</strong> to verify this captured photo.</div>
          </div>
        `;
      }

      showToast("Live photo captured successfully!");
    }

    /* =========================================================================
       Allow Select Files from Local Folder
       ========================================================================= */

    function handleFolderFileSelect(e) {
      var file = e.target.files && e.target.files[0];
      if (!file) return;
      loadLocalImageFile(file);
    }

    function loadLocalImageFile(file) {
      var reader = new FileReader();
      reader.onload = function(evt) {
        var dataUrl = evt.target.result;
        currentInspectionPhotoUrl = dataUrl;
        currentInspectionPhotoB64 = dataUrl.split(",")[1];

        stopLiveCamera();

        var img = document.getElementById("details-box-photo");
        if (img) {
          img.src = dataUrl;
          img.style.display = "block";
          img.style.opacity = "1";
        }
        var loader = document.getElementById("details-img-loader");
        if (loader) loader.style.display = "none";

        var pills = document.querySelectorAll(".action-pill-btn");
        pills.forEach(function(b) { b.classList.remove("active"); });

        var resEl = document.getElementById("ai-analysis-results");
        if (resEl) {
          resEl.innerHTML = `
            <div style="padding: 16px; text-align: center; background: #eff6ff; border-radius: 12px; border: 1px solid #bfdbfe;">
              <div style="font-weight: 700; color: #1e40af; font-size: 0.9rem;">📁 Selected: ${file.name}</div>
              <div style="font-size: 0.8rem; color: #2563eb; margin-top: 4px;">Photo loaded from folder. Click <strong>"Analyze with AI Agent"</strong> to inspect.</div>
            </div>
          `;
        }

        showToast("Loaded " + file.name + " from folder");
      };
      reader.readAsDataURL(file);
    }

    function setupDropzone() {
      var zone = document.getElementById("dropzone-area");
      if (!zone) return;

      zone.ondragover = function(e) {
        e.preventDefault();
        zone.classList.add("dropzone-active");
      };
      zone.ondragleave = function(e) {
        e.preventDefault();
        zone.classList.remove("dropzone-active");
      };
      zone.ondrop = function(e) {
        e.preventDefault();
        zone.classList.remove("dropzone-active");
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
          loadLocalImageFile(e.dataTransfer.files[0]);
        }
      };
    }

    /* =========================================================================
       Preset Scenario Switching
       ========================================================================= */

    async function switchInspectionScenario(scenId) {
      stopLiveCamera();
      currentInspectionScenario = scenId;
      var pills = document.querySelectorAll("#scenario-pill-container .action-pill-btn");
      pills.forEach(function(b) { b.classList.remove("active"); });
      var activeBtn = document.getElementById("btn-scen-" + scenId);
      if (activeBtn) activeBtn.classList.add("active");

      var resEl = document.getElementById("ai-analysis-results");
      if (resEl) {
        resEl.innerHTML = `
          <div style="padding: 16px; text-align: center; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
            <div style="font-weight: 700; color: #1e293b; font-size: 0.9rem;">Scenario Updated!</div>
            <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px;">Click <strong>"Analyze with AI Agent"</strong> to evaluate this product picture.</div>
          </div>
        `;
      }

      await loadInspectionScenario(scenId);
    }

    async function loadInspectionScenario(scenId) {
      try {
        var resp = await fetch("/api/scenario/" + scenId);
        if (resp.ok) {
          var data = await resp.json();
          currentInspectionPhotoB64 = data.photo_base64;
          currentInspectionPhotoUrl = data.photo_data_url;
          currentInspectionOrder = data.order;

          var img = document.getElementById("details-box-photo");
          var loader = document.getElementById("details-img-loader");
          if (img && data.photo_data_url) {
            img.src = data.photo_data_url;
            img.style.display = "block";
            img.style.opacity = "1";
            if (loader) loader.style.display = "none";
          }

          var mList = document.getElementById("manifest-items-list");
          var mId = document.getElementById("manifest-order-id");
          if (data.order) {
            if (mId) mId.innerText = data.order.order_id || "ORD-2024-001";
            if (mList && data.order.items) {
              mList.innerHTML = data.order.items.map(function(it) {
                var varTxt = it.variant ? ' <span style="color:#64748b;font-size:0.75rem;">[' + it.variant + ']</span>' : '';
                return '<div style="display:flex;justify-content:space-between;padding:2px 0;">' +
                  '<span>&bull; ' + it.name + varTxt + '</span>' +
                  '<span style="font-weight:700;">Qty: ' + it.expected_qty + '</span>' +
                '</div>';
              }).join("");
            }
          }
        }
      } catch (err) {
        console.error("Failed to load scenario data", err);
      }
    }

    /* =========================================================================
       AI Agent Inspection Runner
       ========================================================================= */


    function toggleAiEngine(engine) {
      var badge = document.getElementById("offline-engine-badge");
      var cloudInputs = document.getElementById("cloud-api-inputs");
      var input = document.getElementById("input-api-key");
      if (engine === "simulation") {
        if (badge) badge.style.display = "block";
        if (cloudInputs) cloudInputs.style.display = "none";
      } else {
        if (badge) badge.style.display = "none";
        if (cloudInputs) cloudInputs.style.display = "block";
        if (input) {
          input.value = localStorage.getItem("pack_manager_api_key_" + engine) || "";
          input.placeholder = "Paste your " + (engine === "gemini" ? "Google Gemini" : "OpenAI") + " API Key...";
        }
      }
    }

    function saveApiKey(val) {
      var engine = document.getElementById("select-ai-engine") ? document.getElementById("select-ai-engine").value : "simulation";
      if (engine !== "simulation") {
        localStorage.setItem("pack_manager_api_key_" + engine, val.trim());
      }
    }

    function updateManifestTarget() {
      var name = document.getElementById("custom-target-name") ? document.getElementById("custom-target-name").value.trim() : "Blue Baseball Cap";
      var color = document.getElementById("custom-target-color") ? document.getElementById("custom-target-color").value : "blue";
      
      currentInspectionOrder = {
        order_id: (currentInspectionPkg ? currentInspectionPkg.id : "ORD-2024-001"),
        items: [
          {
            name: name || "Product Item",
            expected_qty: 1,
            variant: color,
            sku: "SKU-" + (color.toUpperCase())
          }
        ]
      };

      var mList = document.getElementById("manifest-items-list");
      if (mList) {
        mList.innerHTML = '<div style="display:flex;justify-content:space-between;padding:2px 0;">' +
          '<span>&bull; ' + (name || "Product Item") + ' <span style="color:#64748b;font-size:0.75rem;">[' + color + ']</span></span>' +
          '<span style="font-weight:700;">Qty: 1</span>' +
        '</div>';
      }
    }

    async function runAIAgentAnalysis() {
      var btn = document.getElementById("btn-run-analysis");
      var btnText = document.getElementById("btn-analyze-text");
      var laser = document.getElementById("scan-laser-line");
      var resultsEl = document.getElementById("ai-analysis-results");

      if (!currentInspectionPhotoB64) {
        showToast("Please wait for product photo to finish loading or capture camera");
        return;
      }

      btn.disabled = true;
      if (btnText) btnText.innerText = "Inspecting with AI Agent...";
      if (laser) laser.style.display = "block";

      resultsEl.innerHTML = `
        <div style="padding: 24px 16px; text-align: center; background: #f8fafc; border-radius: 12px; border: 1px solid #e2e8f0;">
          <div style="display:inline-block;width:32px;height:32px;border:3px solid #bfdbfe;border-top-color:#2563eb;border-radius:50%;animation:spin 0.8s linear infinite;margin-bottom:12px;"></div>
          <div style="font-weight: 700; color: #1e293b; font-size: 0.94rem;">AI Vision Agent Inspecting Contents...</div>
          <div style="font-size: 0.8rem; color: #64748b; margin-top: 4px; line-height: 1.4;">
            Analyzing 24-bit RGB photograph, detecting SKU attributes, cross-referencing order quantities and packaging integrity.
          </div>
        </div>
      `;

      try {
        var orderPayload = currentInspectionOrder || {
          order_id: currentInspectionPkg.id,
          items: [{ name: currentInspectionPkg.name, expected_qty: 1 }]
        };

        var engine = document.getElementById("select-ai-engine") ? document.getElementById("select-ai-engine").value : "simulation";
        var apiKey = (engine !== "simulation") ? (localStorage.getItem("pack_manager_api_key_" + engine) || "") : "";

        if (engine !== "simulation" && !apiKey) {
          showToast("No API key entered; auto-inspecting with onboard Smart Computer Vision.");
        }

        var resp = await fetch("/api/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            order: orderPayload,
            photo_base64: currentInspectionPhotoB64,
            provider: engine,
            api_key: apiKey
          })
        });

        var result;
        try {
          result = await resp.json();
        } catch (jsonErr) {
          throw new Error("HTTP error " + resp.status + " (Server did not return JSON)");
        }

        if (!resp.ok) {
          throw new Error(result.error || ("HTTP error " + resp.status));
        }

        renderAnalysisResult(result);
      } catch (err) {
        console.error("AI Analysis failed:", err);
        resultsEl.innerHTML = `
          <div style="padding: 16px; background: #fef2f2; border: 1px solid #fecaca; border-radius: 10px; color: #991b1b; font-size: 0.85rem;">
            <strong>Analysis Error:</strong> ${err.message}
          </div>
        `;
      } finally {
        btn.disabled = false;
        if (btnText) btnText.innerText = "Analyze with AI Agent";
        if (laser) laser.style.display = "none";
      }
    }

    function renderAnalysisResult(res) {
      var resultsEl = document.getElementById("ai-analysis-results");
      var isSeal = (res.decision === "SEAL");
      var isStop = (res.decision === "STOP_FIX");
      
      var badgeBg = isSeal ? "#ecfdf5" : (isStop ? "#fef2f2" : "#fffbeb");
      var badgeBorder = isSeal ? "#a7f3d0" : (isStop ? "#fecaca" : "#fde68a");
      var badgeColor = isSeal ? "#065f46" : (isStop ? "#991b1b" : "#92400e");
      var badgeTitle = isSeal ? "✅ SEAL FOR SHIPPING (PASS)" : (isStop ? "❌ STOP & FIX (HOLD)" : "⚠️ UNCERTAIN (MANUAL REVIEW)");
      var confColor = res.confidence === "high" ? "#10b981" : (res.confidence === "medium" ? "#f59e0b" : "#ef4444");

      // Draw bounding boxes on the photo viewport
      var bboxContainer = document.getElementById("bbox-overlay-container");
      if (bboxContainer) {
        bboxContainer.innerHTML = "";
        var records = res.evidence_records || [];
        records.forEach(function(rec) {
          if (!rec.bbox) return;
          var b = rec.bbox;
          var ymin = (b.ymin * 100);
          var xmin = (b.xmin * 100);
          var width = ((b.xmax - b.xmin) * 100);
          var height = ((b.ymax - b.ymin) * 100);

          var col = "#10b981"; // green
          if (rec.status === "AMBIGUOUS" || rec.status === "UNRESOLVED") col = "#f59e0b"; // yellow
          if (res.decision === "STOP_FIX" && res.extra_items && res.extra_items.some(function(e){ return e.sku === rec.matched_sku; })) {
            col = "#ef4444"; // red
          }

          var box = document.createElement("div");
          box.style.position = "absolute";
          box.style.top = ymin + "%";
          box.style.left = xmin + "%";
          box.style.width = width + "%";
          box.style.height = height + "%";
          box.style.border = "2px solid " + col;
          box.style.borderRadius = "4px";
          box.style.boxShadow = "0 0 6px " + col + "66";
          box.style.pointerEvents = "auto";
          box.title = (rec.object_id || "") + ": " + (rec.matched_sku || rec.observed_label || "");

          var tag = document.createElement("div");
          tag.style.position = "absolute";
          tag.style.top = "-16px";
          tag.style.left = "0";
          tag.style.background = col;
          tag.style.color = "#ffffff";
          tag.style.fontSize = "9px";
          tag.style.fontWeight = "bold";
          tag.style.padding = "1px 4px";
          tag.style.borderRadius = "3px";
          tag.style.whiteSpace = "nowrap";
          tag.innerText = (rec.object_id || "") + (rec.matched_sku ? " (" + rec.matched_sku + ")" : "");
          box.appendChild(tag);
          bboxContainer.appendChild(box);
        });
      }

      // 1. Expected Items
      var expItems = res.expected_items || [];
      var expRows = expItems.map(function(it) {
        return '<tr>' +
          '<td style="padding: 4px 6px; font-weight: 600;">' + (it.sku || '-') + '</td>' +
          '<td style="padding: 4px 6px;">' + (it.product_name || '-') + '</td>' +
          '<td style="padding: 4px 6px; text-align: center; font-weight: 700;">' + (it.expected_qty || 1) + '</td>' +
        '</tr>';
      }).join("") || '<tr><td colspan="3" style="padding:4px 6px;color:#94a3b8;">None</td></tr>';

      // 2. Observed Items
      var obsItems = res.observed_items || [];
      var obsRows = obsItems.map(function(it) {
        var conf = (typeof it.confidence === "number") ? (it.confidence * 100).toFixed(0) + "%" : it.confidence;
        return '<tr>' +
          '<td style="padding: 4px 6px; font-weight: 600;">' + (it.object_id || '-') + '</td>' +
          '<td style="padding: 4px 6px;">' + (it.label || '-') + '</td>' +
          '<td style="padding: 4px 6px; text-align: right; color: #2563eb; font-weight: 600;">' + conf + '</td>' +
        '</tr>';
      }).join("") || '<tr><td colspan="3" style="padding:4px 6px;color:#94a3b8;">None</td></tr>';

      // 3. Results (Matched, Missing, Extra, Quantity Mismatches, Unverified)
      var matchedCount = (res.matched_items || []).reduce(function(a, b){ return a + (b.observed_qty || 1); }, 0);
      var missingCount = (res.missing_items || []).reduce(function(a, b){ return a + (b.expected_qty || 1); }, 0);
      var extraCount = (res.extra_items || []).reduce(function(a, b){ return a + (b.observed_qty || 1); }, 0);
      var qtyMismatchCount = (res.quantity_mismatches || []).length;
      var unverifiedCount = (res.unverified_items || []).length;

      var resultBadges = `
        <div style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;">
          <span style="padding:3px 8px;border-radius:6px;font-size:0.74rem;font-weight:700;background:#ecfdf5;color:#065f46;border:1px solid #a7f3d0;">MATCHED: ${matchedCount}</span>
          <span style="padding:3px 8px;border-radius:6px;font-size:0.74rem;font-weight:700;background:${missingCount > 0 ? '#fef2f2' : '#f8fafc'};color:${missingCount > 0 ? '#991b1b' : '#64748b'};border:1px solid ${missingCount > 0 ? '#fecaca' : '#e2e8f0'};">MISSING: ${missingCount}</span>
          <span style="padding:3px 8px;border-radius:6px;font-size:0.74rem;font-weight:700;background:${extraCount > 0 ? '#fef2f2' : '#f8fafc'};color:${extraCount > 0 ? '#991b1b' : '#64748b'};border:1px solid ${extraCount > 0 ? '#fecaca' : '#e2e8f0'};">EXTRA: ${extraCount}</span>
          <span style="padding:3px 8px;border-radius:6px;font-size:0.74rem;font-weight:700;background:${qtyMismatchCount > 0 ? '#fffbeb' : '#f8fafc'};color:${qtyMismatchCount > 0 ? '#92400e' : '#64748b'};border:1px solid ${qtyMismatchCount > 0 ? '#fde68a' : '#e2e8f0'};">QTY MISMATCH: ${qtyMismatchCount}</span>
          <span style="padding:3px 8px;border-radius:6px;font-size:0.74rem;font-weight:700;background:${unverifiedCount > 0 ? '#fffbeb' : '#f8fafc'};color:${unverifiedCount > 0 ? '#92400e' : '#64748b'};border:1px solid ${unverifiedCount > 0 ? '#fde68a' : '#e2e8f0'};">UNVERIFIED: ${unverifiedCount}</span>
        </div>
      `;

      // 4. Grounded Evidence Records Table
      var evidenceRecords = res.evidence_records || [];
      var evidenceRows = evidenceRecords.map(function(ev) {
        var bboxTxt = ev.bbox ? `[${ev.bbox.ymin.toFixed(2)}, ${ev.bbox.xmin.toFixed(2)}, ${ev.bbox.ymax.toFixed(2)}, ${ev.bbox.xmax.toFixed(2)}]` : '-';
        var evText = (ev.matching_evidence && ev.matching_evidence.length) ? ev.matching_evidence.join("; ") : (ev.notes || '-');
        return '<tr>' +
          '<td style="padding: 4px 6px; font-weight: 700; color: #1e293b;">' + (ev.object_id || '-') + '</td>' +
          '<td style="padding: 4px 6px; font-family: monospace; font-size: 0.70rem;">' + (ev.image_id || '-') + '</td>' +
          '<td style="padding: 4px 6px; font-family: monospace; font-size: 0.70rem; color: #64748b;">' + bboxTxt + '</td>' +
          '<td style="padding: 4px 6px; font-weight: 600; color: #2563eb;">' + (ev.matched_sku || ev.observed_label || '-') + '</td>' +
          '<td style="padding: 4px 6px; font-size: 0.70rem; color: #475569;">' + evText + '</td>' +
        '</tr>';
      }).join("") || '<tr><td colspan="5" style="padding:4px 6px;color:#94a3b8;">No grounded records</td></tr>';

      resultsEl.innerHTML = `
        <!-- Verdict & Operator Action Header -->
        <div style="padding: 12px 14px; background: ${badgeBg}; border: 1px solid ${badgeBorder}; border-radius: 10px; margin-bottom: 10px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight: 800; color: ${badgeColor}; font-size: 0.94rem;">${badgeTitle}</span>
            <span style="font-size: 0.72rem; font-weight: 700; padding: 2px 8px; border-radius: 12px; background: #ffffff; color: ${confColor}; border: 1px solid #e2e8f0;">
              ${(res.confidence ? res.confidence.toUpperCase() : 'HIGH')} CONFIDENCE
            </span>
          </div>
          <div style="font-size: 0.82rem; font-weight: 700; color: ${badgeColor}; margin-bottom: 4px;">
            Operator Action: <span style="text-decoration: underline;">${res.operator_action || 'Inspect Carton'}</span>
          </div>
          <div style="font-size: 0.78rem; color: ${badgeColor}; line-height: 1.4;">
            ${(res.decision_reason || 'Package contents inspected against expected manifest.')}
          </div>
        </div>

        <!-- Result Breakdown Badges -->
        ${resultBadges}

        <!-- Expected vs Observed Manifest Panels -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;">
          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;font-size:0.74rem;">
            <div style="font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-bottom:4px;">Expected Manifest</div>
            <table style="width:100%;border-collapse:collapse;">
              <thead><tr style="color:#64748b;border-bottom:1px solid #e2e8f0;"><th style="text-align:left;padding:2px 4px;">SKU</th><th style="text-align:left;padding:2px 4px;">Product</th><th style="text-align:center;padding:2px 4px;">Qty</th></tr></thead>
              <tbody>${expRows}</tbody>
            </table>
          </div>
          <div style="background:#ffffff;border:1px solid #e2e8f0;border-radius:8px;padding:8px;font-size:0.74rem;">
            <div style="font-weight:700;color:var(--text-muted);text-transform:uppercase;margin-bottom:4px;">Observed Items</div>
            <table style="width:100%;border-collapse:collapse;">
              <thead><tr style="color:#64748b;border-bottom:1px solid #e2e8f0;"><th style="text-align:left;padding:2px 4px;">ID</th><th style="text-align:left;padding:2px 4px;">Visual Label</th><th style="text-align:right;padding:2px 4px;">Conf</th></tr></thead>
              <tbody>${obsRows}</tbody>
            </table>
          </div>
        </div>

        <!-- Grounded Evidence Audit Trail -->
        <div style="margin-bottom: 10px; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden;">
          <div style="padding: 6px 10px; background: #f8fafc; font-size: 0.74rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; border-bottom: 1px solid #e2e8f0;">
            Grounded Evidence &amp; Spatial Alignment
          </div>
          <div style="max-height: 140px; overflow-y: auto;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.74rem;">
              <thead style="background: #f1f5f9; color: var(--text-muted); font-size: 0.70rem;">
                <tr>
                  <th style="padding: 4px 6px; text-align: left;">Object</th>
                  <th style="padding: 4px 6px; text-align: left;">Image</th>
                  <th style="padding: 4px 6px; text-align: left;">Bounding Box</th>
                  <th style="padding: 4px 6px; text-align: left;">Resolved SKU</th>
                  <th style="padding: 4px 6px; text-align: left;">Matching Evidence</th>
                </tr>
              </thead>
              <tbody>${evidenceRows}</tbody>
            </table>
          </div>
        </div>

        <!-- Operator Action Buttons -->
        <div style="display:flex;gap:8px;margin-bottom:8px;">
          ${isSeal ? 
            '<button onclick="applyDecisionDelivered()" style="flex:1;padding:8px;background:#059669;color:#ffffff;border:none;border-radius:8px;font-weight:700;font-size:0.84rem;cursor:pointer;">✅ Confirm &amp; Seal Carton</button>' :
            '<button onclick="applyDecisionPending()" style="flex:1;padding:8px;background:#dc2626;color:#ffffff;border:none;border-radius:8px;font-weight:700;font-size:0.84rem;cursor:pointer;">❌ Route to Remediation Station</button>'
          }
        </div>

        <!-- Collapsible Audit JSON -->
        <details style="margin-top: 4px;">
          <summary style="font-size: 0.74rem; font-weight: 700; color: #2563eb; cursor: pointer;">🔍 View Stage 5 PackVerificationReport JSON</summary>
          <pre style="background: #0f172a; color: #38bdf8; padding: 10px; border-radius: 8px; font-size: 0.70rem; overflow-x: auto; margin-top: 6px; max-height: 160px;">${JSON.stringify(res, null, 2)}</pre>
        </details>
      `;

      if (currentInspectionPkg) {
        currentInspectionPkg.verdict = res.decision;
        currentInspectionPkg.verdictText = res.decision_reason;
      }
    }

    function applyDecisionDelivered() {
      applyDecisionToPackage("Delivered");
    }

    function applyDecisionPending() {
      applyDecisionToPackage("Pending");
    }

    function applyDecisionToPackage(newStatus) {
      if (currentInspectionPkg) {
        currentInspectionPkg.status = newStatus;
        renderTables();
        var pill = document.getElementById("detail-pill-status");
        if (pill) {
          pill.className = "pill-status " + newStatus.toLowerCase().replace(/\s+/g, '');
          pill.innerText = newStatus;
        }
        showToast("Package " + currentInspectionPkg.id + " updated to " + newStatus);
      }
      closeDetailsModal();
    }

    /* Popover toggles */
    function toggleNotifications(e) {
      if (e) e.stopPropagation();
      var notif = document.getElementById("notif-popover");
      if (notif) notif.classList.toggle("open");
    }

    function toggleUserMenu(e) {
      if (e) e.stopPropagation();
      var pop = document.getElementById("user-popover");
      if (pop) pop.classList.toggle("open");
    }

    function clearNotifs() {
      var badge = document.getElementById("notif-badge");
      if (badge) badge.style.display = "none";
      var list = document.getElementById("notif-list");
      if (list) list.innerHTML = '<div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 0.8rem;">No new notifications</div>';
      showToast("All notifications marked as read");
    }

    function setTimeframe(tf, label) {
      var tfBtn = document.getElementById("btn-timeframe");
      if (tfBtn) tfBtn.innerText = label;
      showToast("Timeframe set to " + label);
    }

    function filterByStatus(status) {
      renderTables(status);
      showToast("Filtered by status: " + status);
    }

    function handleTableSearch(e) {
      var query = e.target.value.trim();
      renderTables(null, null, query);
    }

    function applyStatusFilter(btn, status) {
      var tabs = document.querySelectorAll(".filter-tab");
      tabs.forEach(function(t) { t.classList.remove("active"); });
      if (btn) btn.classList.add("active");
      renderTables(status);
    }

    function showToast(msg) {
      var toast = document.getElementById("toast-msg");
      var text = document.getElementById("toast-text");
      if (!toast || !text) return;
      text.innerText = msg;
      toast.classList.add("show");
      setTimeout(function() { toast.classList.remove("show"); }, 3000);
    }

    // Initialize on load
    document.addEventListener("DOMContentLoaded", init);

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
                {"id": "example_7_electronics_order", "title": "Example 7: Electronics Order (SEAL)"},
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

            try:
                vlm = get_vlm_client(provider=provider, api_key=api_key)
            except ValueError as ve:
                self._send_json(400, {"error": f"API Key Required: {ve}"})
                return

            catalogue_payload = req.get("catalogue")
            agent = PackManagerAIAgent(vlm_client=vlm)

            try:
                report = agent.verify_full(
                    order=order_payload,
                    catalogue=catalogue_payload,
                    photo=photo_bytes,
                )
                report_dict = report.model_dump()
                # Backward-compatibility bridges for frontend / existing clients
                report_dict["matches"] = [
                    {
                        "item_name": m.get("product_name") or m.get("sku"),
                        "expected_qty": m.get("expected_qty", 1),
                        "detected_qty": m.get("observed_qty", 1),
                        "status": "PASS",
                        "variant_match": True,
                    }
                    for m in report_dict.get("matched_items", [])
                ]
                has_damage = any(
                    "damage" in str(b).lower()
                    for b in report_dict.get("blocking_issues", [])
                )
                report_dict["product_condition"] = {
                    "visible_damage": has_damage,
                    "notes": "Packaging damaged" if has_damage else "Carton intact",
                }
                self._send_json(200, report_dict)
            except Exception as e:
                logger.exception("Verification execution failed")
                self._send_json(400, {"error": f"Inspection failed: {e}"})
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
