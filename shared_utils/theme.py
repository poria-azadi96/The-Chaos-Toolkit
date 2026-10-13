"""
The Chaos Toolkit - Shared Visual Engine & Modern GUI Styling Utilities.
"""
import matplotlib.pyplot as plt

PALETTE = {
    "bg_dark": "#080C14", "surface": "#0F172A", "border": "#1E293B",
    "text_main": "#F8FAFC", "text_muted": "#94A3B8", "cyan": "#00F5FF",
    "blue": "#0284C7", "emerald": "#39FF14", "crimson": "#FF0055", "amber": "#FFB800"
}

def apply_sleek_theme():
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg_dark"], "axes.facecolor": PALETTE["surface"],
        "axes.edgecolor": PALETTE["border"], "axes.labelcolor": PALETTE["text_muted"],
        "axes.titlecolor": PALETTE["text_main"], "xtick.color": PALETTE["text_muted"],
        "ytick.color": PALETTE["text_muted"], "grid.color": PALETTE["border"],
        "grid.linestyle": "--", "grid.alpha": 0.5, "legend.facecolor": PALETTE["surface"],
        "legend.edgecolor": PALETTE["border"], "legend.labelcolor": PALETTE["text_main"]
    })

def create_dashboard_header(title, subtitle):
    return """
    <div style="background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%); 
                padding: 16px 20px; border-radius: 10px; border-left: 5px solid #00F5FF; 
                margin-bottom: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
        <h3 style="color: #F8FAFC; margin: 0; font-family: sans-serif; font-size: 18px;">{}</h3>
        <p style="color: #94A3B8; margin: 5px 0 0 0; font-family: sans-serif; font-size: 13px;">{}</p>
    </div>
    """.format(title, subtitle)
