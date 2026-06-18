PAGE_CONFIG = {
    "page_title": "SSSP Network Simulator | Social Graph Analysis",
    "page_icon": "🔗",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

COLORS_LIGHT = {
    "source": "#2ECC71",
    "destination": "#E74C3C",
    "path": "#3498DB",
    "normal": "#85C1E9",
    "unreachable": "#BDC3C7",
    "path_edge": "#2980B9",
    "fig_bg": "#FFFFFF",
    "ax_bg": "#F8F9FA",
    "edge_color": "#95A5A6",
    "edge_label_color": "#555555",
    "node_edge": "#2C3E50",
    "label_color": "#1A1A2E",
    "title_color": "#2C3E50",
}

COLORS_DARK = {
    "source": "#2ECC71",
    "destination": "#E74C3C",
    "path": "#5DADE2",
    "normal": "#5B7FA5",
    "unreachable": "#4A4A5A",
    "path_edge": "#3498DB",
    "fig_bg": "#0E1117",
    "ax_bg": "#1A1A2E",
    "edge_color": "#4A5568",
    "edge_label_color": "#A0AEC0",
    "node_edge": "#E2E8F0",
    "label_color": "#E2E8F0",
    "title_color": "#E2E8F0",
}

GITHUB_URL = "https://github.com/deepanshu-prog/Searching-and-Recommendation-system-For-Social-Media"

LIGHT_CSS = """
<style>
    .stMetric {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e9f2 100%);
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 4px solid #2ECC71;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #2C3E50;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #7F8C8D;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fafbfc 0%, #f0f2f6 100%);
    }
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
        border: 1px solid #ccc;
    }
    .graph-container {
        background: #F8F9FA;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 8px;
    }
    .rec-card {
        background: linear-gradient(135deg, #f5f7fa, #e4e9f2);
        padding: 16px; border-radius: 12px; text-align: center;
        border: 1px solid #dfe6e9; min-height: 160px;
    }
    .rec-card .name { font-weight: 700; font-size: 1rem; margin: 4px 0; color: #2C3E50; }
    .rec-card .meta { font-size: 0.8rem; color: #7F8C8D; }
    .rec-card .strength { color: #2ECC71; font-weight: 600; }
</style>
"""

DARK_CSS = """
<style>
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        color: #E2E8F0;
    }
    [data-testid="stHeader"] {
        background-color: #0E1117;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22 0%, #0D1117 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #C9D1D9 !important;
    }
    section[data-testid="stSidebar"] label {
        color: #C9D1D9 !important;
    }
    .stMetric {
        background: linear-gradient(135deg, #161B22 0%, #1A1F2B 100%);
        padding: 12px 16px;
        border-radius: 10px;
        border-left: 4px solid #2ECC71;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.4rem;
        font-weight: 700;
        color: #E2E8F0;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #8B949E;
    }
    .legend-box {
        display: inline-block;
        width: 14px;
        height: 14px;
        border-radius: 3px;
        margin-right: 6px;
        vertical-align: middle;
        border: 1px solid #4A5568;
    }
    .graph-container {
        background: #1A1A2E;
        border: 1px solid #2D3748;
        border-radius: 12px;
        padding: 8px;
    }
    .rec-card {
        background: linear-gradient(135deg, #161B22, #1A1F2B);
        padding: 16px; border-radius: 12px; text-align: center;
        border: 1px solid #2D3748; min-height: 160px;
    }
    .rec-card .name { font-weight: 700; font-size: 1rem; margin: 4px 0; color: #E2E8F0; }
    .rec-card .meta { font-size: 0.8rem; color: #8B949E; }
    .rec-card .strength { color: #2ECC71; font-weight: 600; }
    h1, h2, h3, h4, p, span, div {
        color: #E2E8F0;
    }
    [data-testid="stDataFrame"] {
        background: #161B22;
    }
</style>
"""

COLOR_LEGEND_HTML = """
<div style="display:flex; gap:16px; flex-wrap:wrap; padding:8px 0; font-size:0.85rem;">
    <span><span class="legend-box" style="background:#2ECC71;"></span>Source</span>
    <span><span class="legend-box" style="background:#E74C3C;"></span>Destination</span>
    <span><span class="legend-box" style="background:#3498DB;"></span>Path</span>
    <span><span class="legend-box" style="background:#85C1E9;"></span>Reachable</span>
    <span><span class="legend-box" style="background:#BDC3C7;"></span>Unreachable</span>
</div>
"""
