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
      max-width: 640px;
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
            <span class="chart-title">Recent Packages</span>
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
      <div class="modal-header-row">
        <span class="modal-heading" id="modal-pkg-title">Package Details</span>
        <button class="btn-close-modal" onclick="closeDetailsModal()">&times;</button>
      </div>
      <div id="modal-pkg-body" style="font-size: 0.88rem; line-height: 1.8; color: var(--text-main);">
        <!-- Filled by JS -->
      </div>
      <div class="modal-footer">
        <button class="btn-cancel" onclick="closeDetailsModal()">Close</button>
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
      { id: "PKG001", name: "Electronics Items", category: "Electronics", status: "Delivered", location: "Hyderabad", date: "Oct 5, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "All items present, correct quantities & colors (1x Blue Baseball Cap). Packaging undamaged." },
      { id: "PKG002", name: "Books Parcel", category: "Books", status: "In Transit", location: "Bengaluru", date: "Oct 5, 2026", scenarioId: "example_2_wrong_item", verdict: "STOP_FIX", verdictText: "Variant mismatch: Cap color detected RED, expected BLUE. Packaging integrity compromised." },
      { id: "PKG003", name: "Clothes Package", category: "Fashion", status: "Pending", location: "Chennai", date: "Oct 4, 2026", scenarioId: "example_3_missing_item", verdict: "STOP_FIX", verdictText: "Missing item: 1x User Manual missing from box contents." },
      { id: "PKG004", name: "Home Essentials", category: "Home", status: "Delivered", location: "Mumbai", date: "Oct 4, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "All items verified against manifest. Box sealed for dispatch." },
      { id: "PKG005", name: "Grocery Items", category: "Grocery", status: "In Transit", location: "Delhi", date: "Oct 3, 2026", scenarioId: "example_4_extra_item", verdict: "STOP_FIX", verdictText: "Unauthorized extra item: 1x Red Scarf detected in box not present on order." },
      { id: "PKG006", name: "Camera Lens Kit", category: "Electronics", status: "Delivered", location: "Hyderabad", date: "Oct 2, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "Verified 100% item match." },
      { id: "PKG007", name: "Designer Shoes", category: "Fashion", status: "Delivered", location: "Pune", date: "Oct 2, 2026", scenarioId: "example_1_correct_order", verdict: "SEAL", verdictText: "Verified 100% item match." },
      { id: "PKG008", name: "Kitchen Blender", category: "Home", status: "In Transit", location: "Ahmedabad", date: "Oct 1, 2026", scenarioId: "example_6_damaged_goods", verdict: "STOP_FIX", verdictText: "Packaging damage detected: Physical compression on box carton." }
    ];

    var activePkgIndex = 0;
    var currentView = "dashboard";

    function init() {
      renderTables();
    }

    /* Navigation between Sidebar Views */
    function navigateTo(viewName) {
      currentView = viewName;
      // Hide all views
      var views = ["dashboard", "packages", "categories", "inventory", "reports", "settings"];
      views.forEach(function(v) {
        var el = document.getElementById("view-" + v);
        if (el) el.classList.remove("active");
        var nav = document.getElementById("nav-" + v);
        if (nav) nav.classList.remove("active");
      });

      // Activate selected view
      var targetView = document.getElementById("view-" + viewName);
      if (targetView) targetView.classList.add("active");
      var targetNav = document.getElementById("nav-" + viewName);
      if (targetNav) targetNav.classList.add("active");

      // Update page title & subtitle
      var titles = {
        dashboard: { title: "Dashboard", sub: "Welcome back, Vivek! Here's an overview of your packages." },
        packages: { title: "Packages Management", sub: "Inspect, track, and dispatch your outbound parcels." },
        categories: { title: "Package Categories", sub: "Distribution of shipments grouped by product classification." },
        inventory: { title: "Warehouse Inventory", sub: "Live stock counts across warehouse fulfillment stations." },
        reports: { title: "Logistics Analytics & Reports", sub: "Performance KPIs, packing speed, and on-time fulfillment rates." },
        settings: { title: "System & Station Settings", sub: "Configure preferences, terminal parameters, and credentials." }
      };

      var t = titles[viewName] || titles["dashboard"];
      document.getElementById("view-title").innerText = t.title;
      document.getElementById("view-subtitle").innerText = t.sub;

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

      // 1. Render Dashboard Table (first 5)
      var dashTbody = document.getElementById("packages-table-body");
      if (dashTbody) {
        var dashRows = filtered.slice(0, 5);
        dashTbody.innerHTML = dashRows.map(function(p, idx) {
          return createRowHtml(p, idx);
        }).join("");
      }

      // 2. Render Full Packages Table
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
      return '<tr onclick="openPackageDetails(' + idx + ')">' +
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

    /* Search Handler */
    function handleTableSearch(e) {
      var query = e.target.value;
      renderTables(null, null, query);
    }

    /* Status Filter Pills in Packages View */
    function applyStatusFilter(btn, status) {
      document.querySelectorAll(".filter-tab").forEach(function(b) { b.classList.remove("active"); });
      btn.classList.add("active");
      renderTables(status);
    }

    function filterByStatus(status) {
      navigateTo("packages");
      var tabs = document.querySelectorAll(".filter-tab");
      tabs.forEach(function(b) {
        if (b.innerText.includes(status) || (status === "all" && b.innerText.includes("All"))) {
          b.classList.add("active");
        } else {
          b.classList.remove("active");
        }
      });
      renderTables(status);
    }

    function filterByCategory(cat) {
      navigateTo("packages");
      renderTables(null, cat);
      showToast("Filtered by " + cat);
    }

    /* Change Timeframe on Bar Chart */
    function updateChartTimeframe(e) {
      var val = e.target.value;
      var bars = document.getElementById("barchart-bars-container");
      if (val === "30") {
        showToast("Loaded 30-Day Aggregated Data");
      } else if (val === "month") {
        showToast("Loaded Monthly Metrics");
      } else {
        showToast("Loaded Last 7 Days");
      }
    }

    /* Add Package Modal & Logic */
    function openAddPackageModal() {
      document.getElementById("new-pkg-id").value = "PKG00" + (packagesData.length + 1);
      document.getElementById("new-pkg-name").value = "";
      document.getElementById("new-pkg-location").value = "";
      document.getElementById("add-pkg-modal").classList.add("open");
    }
    function closeAddPackageModal() {
      document.getElementById("add-pkg-modal").classList.remove("open");
    }
    function submitAddPackage() {
      var id = document.getElementById("new-pkg-id").value.trim() || ("PKG00" + (packagesData.length + 1));
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
        date: "Oct 5, 2026"
      });

      // Update KPI Counter
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
      var rect = event.target.getBoundingClientRect();
      menu.style.top = (rect.bottom + window.scrollY + 4) + "px";
      menu.style.left = (rect.left - 130) + "px";
      menu.classList.add("show");
    }

    function closeAllPopovers(event) {
      var menu = document.getElementById("row-action-menu");
      if (menu) menu.classList.remove("show");
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

    /* Package Details Modal - Displays Open Box Photograph & AI Verification */
    async function openPackageDetails(idx) {
      var p = packagesData[idx];
      if (!p) return;
      document.getElementById("modal-pkg-title").innerText = p.id + " — " + p.name;
      var body = document.getElementById("modal-pkg-body");
      var scenId = p.scenarioId || "example_1_correct_order";
      var isSeal = (p.verdict === "SEAL");

      body.innerHTML = '<div style="margin-bottom: 14px;">' +
        '<div style="font-size: 0.78rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">' +
          '<span>📸 Open Box Photograph (Packing Station RGB Camera)</span>' +
          '<span style="font-size: 0.74rem; color: #2563eb; background: #eff6ff; padding: 2px 8px; border-radius: 12px; font-weight: 600;">Station Cam #03</span>' +
        '</div>' +
        '<div style="position: relative; width: 100%; height: 260px; background: #060911; border-radius: 12px; overflow: hidden; display: flex; align-items: center; justify-content: center; border: 2px dashed #cbd5e1;">' +
          '<img id="details-box-photo" src="" alt="Open Box Photograph" style="max-width: 100%; max-height: 100%; object-fit: contain; opacity: 0; transition: opacity 0.3s ease;">' +
          '<div id="details-img-loader" style="position: absolute; color: #94a3b8; font-size: 0.85rem; font-weight: 500;">Loading Open Box Photograph...</div>' +
        '</div>' +
      '</div>' +
      '<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px; font-size: 0.84rem;">' +
        '<div><strong>Tracking ID:</strong> <span class="tracking-code">' + p.id + '</span></div>' +
        '<div><strong>Category:</strong> <span class="pill-category ' + p.category.toLowerCase() + '">' + p.category + '</span></div>' +
        '<div><strong>Status:</strong> <span class="pill-status ' + p.status.toLowerCase().replace(/\s+/g,'') + '">' + p.status + '</span></div>' +
        '<div><strong>Hub Location:</strong> ' + p.location + '</div>' +
      '</div>' +
      '<div style="padding: 12px 14px; background: ' + (isSeal ? '#ecfdf5' : '#fef2f2') + '; border: 1px solid ' + (isSeal ? '#a7f3d0' : '#fecaca') + '; border-radius: 10px; margin-bottom: 14px;">' +
        '<div style="font-weight: 700; color: ' + (isSeal ? '#065f46' : '#991b1b') + '; margin-bottom: 2px;">' +
          'AI Pack Verification Verdict: ' + (isSeal ? '✅ SEAL FOR SHIPPING (PASS)' : '❌ STOP & FIX (HOLD)') +
        '</div>' +
        '<div style="font-size: 0.82rem; color: ' + (isSeal ? '#047857' : '#b91c1c') + ';">' +
          (p.verdictText || 'Package contents inspected against expected manifest.') +
        '</div>' +
      '</div>' +
      '<div style="padding: 10px 12px; background: #f8fafc; border-radius: 8px; border: 1px solid #e2e8f0; font-size: 0.8rem; color: var(--text-muted);">' +
        '<strong>Visual Verification Note:</strong> Photograph captured at packing station under 24-bit RGB inspection lighting before box seal.' +
      '</div>';

      document.getElementById("pkg-details-modal").classList.add("open");

      // Asynchronously fetch and display the Open Box Photograph from API
      try {
        var resp = await fetch("/api/scenario/" + scenId);
        if (resp.ok) {
          var data = await resp.json();
          var img = document.getElementById("details-box-photo");
          var loader = document.getElementById("details-img-loader");
          if (img && data.photo_data_url) {
            img.src = data.photo_data_url;
            img.style.opacity = "1";
            if (loader) loader.style.display = "none";
          }
        }
      } catch (err) {
        console.error("Failed to load box photograph", err);
      }
    }
    function closeDetailsModal() {
      document.getElementById("pkg-details-modal").classList.remove("open");
    }

    /* Popover toggles */
    function toggleNotifications(e) {
      e.stopPropagation();
      var notif = document.getElementById("notif-popover");
      notif.classList.toggle("open");
      document.getElementById("user-popover").classList.remove("open");
    }
    function clearNotifs() {
      document.getElementById("notif-list").innerHTML = '<div style="padding: 16px; text-align: center; color: var(--text-muted); font-size: 0.84rem;">No unread notifications</div>';
      var badge = document.getElementById("notif-badge");
      if (badge) badge.style.display = "none";
      showToast("All notifications marked as read");
    }
    function toggleUserDropdown(e) {
      e.stopPropagation();
      var userPop = document.getElementById("user-popover");
      userPop.classList.toggle("open");
      document.getElementById("notif-popover").classList.remove("open");
    }

    /* Export CSV */
    function exportDataCsv() {
      var csv = "Tracking ID,Package Name,Category,Status,Location,Date\\n";
      packagesData.forEach(function(p) {
        csv += p.id + ',"' + p.name + '",' + p.category + ',' + p.status + ',"' + p.location + '",' + p.date + "\\n";
      });
      var blob = new Blob([csv], { type: "text/csv" });
      var url = URL.createObjectURL(blob);
      var a = document.createElement("a");
      a.href = url;
      a.download = "pack_manager_shipments.csv";
      a.click();
      showToast("Exported " + packagesData.length + " packages to CSV");
    }

    /* Toast Notification Helper */
    function showToast(msg) {
      var toast = document.getElementById("toast-msg");
      var text = document.getElementById("toast-text");
      text.innerText = msg;
      toast.style.display = "flex";
      setTimeout(function() {
        toast.style.display = "none";
      }, 2600);
    }

    // Auto-init
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", init);
    } else {
      init();
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
