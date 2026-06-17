PAGE_CONFIG = {
    "page_title": "SSSP Network Simulator | Social Graph Analysis",
    "page_icon": "🔗",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

COLORS = {
    "source": "#2ECC71",
    "destination": "#E74C3C",
    "path": "#3498DB",
    "normal": "#85C1E9",
    "unreachable": "#BDC3C7",
    "path_edge": "#2980B9",
}

GITHUB_URL = "https://github.com/deepanshu-prog/Searching-and-Recommendation-system-For-Social-Media"

CUSTOM_CSS = """
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
