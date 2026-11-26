"""Streamlit uygulaması için ortak yardımcılar: veri yükleme, renkler, grafik fonksiyonları."""
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DOCS_DIR = ROOT / "docs"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

POS, NEG, LINE, BASELINE, MUTED = "#eb6834", "#2a78d6", "#2a78d6", "#898781", "#898781"
H = 17

MACRO_PANELS = [
    ("cpi", "TÜFE düzeyi (%)"),
    ("pi", "TÜFE enflasyonu (yıllık puan)"),
    ("y", "GSYH (%)"),
    ("inv", "Yatırım (%)"),
    ("i", "Politika faizi (yıllık puan)"),
    ("q", "Reel kur (%, + = TL değer kaybı)"),
]


@st.cache_data
def load_explorer():
    with open(MODEL_DIR / "explorer_data.json", encoding="utf-8") as f:
        return json.load(f)


def fmt(x, d=2):
    return f"{x:+.{d}f}".replace(".", ",")


def decimals(vals):
    m = float(np.max(np.abs(vals))) if len(vals) else 0.0
    return 4 if m < 0.05 else 3 if m < 0.5 else 2


def sector_bar(sectors, labels, vals, ghost=None, ghost_name="Ağsız model", title="", xlabel="%"):
    order = np.argsort(np.abs(vals))  # plotly alttan üste çizer: en büyük mutlak etki en üstte
    v = np.asarray(vals)[order]
    names = [f"{sectors[i]} {labels[i]}" for i in order]
    d = decimals(v)
    fig = go.Figure()
    fig.add_bar(x=v, y=names, orientation="h", marker_color=[POS if x >= 0 else NEG for x in v],
                text=[fmt(x, d) for x in v], textposition="outside", cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.4f}%<extra></extra>", name="Tepki")
    if ghost is not None:
        g = np.asarray(ghost)[order]
        fig.add_scatter(x=g, y=names, mode="markers", marker=dict(symbol="line-ns-open", size=14, color="#0b0b0b", line=dict(width=2)),
                        name=ghost_name, hovertemplate="%{y}<br>" + ghost_name + ": %{x:.4f}%<extra></extra>")
    fig.add_vline(x=0, line_color=MUTED, line_width=1)
    fig.update_layout(height=max(420, 22 * len(v) + 80), margin=dict(l=10, r=40, t=40, b=30), title=title,
                      xaxis_title=xlabel, showlegend=ghost is not None, legend=dict(orientation="h", y=1.02, x=0),
                      bargap=0.25, plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(gridcolor="#e1e0d9", zeroline=False)
    return fig


def macro_lines(series_list, h_mark=None, height=190):
    """series_list: [(etiket, dict(key->list), renk, kesikli)]"""
    figs = []
    for key, title in MACRO_PANELS:
        fig = go.Figure()
        for name, data, color, dash in series_list:
            fig.add_scatter(x=list(range(len(data[key]))), y=data[key], mode="lines", name=name,
                            line=dict(color=color, width=2, dash="dash" if dash else "solid"),
                            hovertemplate=name + "<br>Çeyrek %{x}: %{y:.3f}<extra></extra>")
        fig.add_hline(y=0, line_color=MUTED, line_width=1)
        if h_mark is not None:
            fig.add_vline(x=h_mark, line_color=MUTED, line_dash="dot")
        fig.update_layout(title=dict(text=title, font=dict(size=13)), height=height, margin=dict(l=10, r=10, t=35, b=25),
                          showlegend=len(series_list) > 1 and key == "cpi", legend=dict(orientation="h", y=-0.25),
                          plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(title=None, gridcolor="#eeeeea", dtick=4)
        fig.update_yaxes(gridcolor="#eeeeea")
        figs.append(fig)
    return figs


FOOTNOTE = ("Birimler: sektörel fiyat düzeyi ve TÜFE düzeyi kümülatif % sapma; enflasyon ve politika faizi yıllıklandırılmış puan; "
            "üretim ve GSYH % sapma; reel kur % (+ = TL reel değer kaybı). Faiz şoku etki anında +1 puana, kur primi şoku TL'nin "
            "reel %1 değer kaybına, Brent şoku +%10'a, dış talep şoku AB-4 öncü göstergesinin etki anında %1 artışına, yurt içi talep "
            "şoku özel tüketimin etki anında %1 artışına, yerli maliyet şoku yerli üretici fiyatlarının etki anında %1 artışına, "
            "yatırım etkinliği şoku yatırımın etki anında %1 artışına, finansal şok dış finansman priminin etki anında yıllık 1 puan "
            "artışına normalize edilmiştir. Sektörel şoklar "
            "ilgili sektörün marjinal maliyetinde %1'lik, yarı ömrü yaklaşık 1 çeyrek olan artıştır. \"Ağsız model\", sektörlerin yurt içi "
            "ara girdilerini ortalama yerli fiyattan aldığı karşı-olgusaldır.")
