import io
import math
import base64
from typing import Dict, List, Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless cloud servers
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from jinja2 import Template
from weasyprint import HTML

# ==========================================
# 1. EMBEDDED KNOWLEDGE BASE & RULES
# ==========================================
DMIT_RULES = {
    "fingerMappings": {
        "L1": {"fingerName": "Left Thumb", "brainHemisphere": "Right", "brainLobe": "Prefrontal Lobe", "primaryIntelligence": "Interpersonal Intelligence"},
        "R1": {"fingerName": "Right Thumb", "brainHemisphere": "Left", "brainLobe": "Prefrontal Lobe", "primaryIntelligence": "Intrapersonal Intelligence"},
        "L2": {"fingerName": "Left Index", "brainHemisphere": "Right", "brainLobe": "Frontal Lobe", "primaryIntelligence": "Spatial Intelligence"},
        "R2": {"fingerName": "Right Index", "brainHemisphere": "Left", "brainLobe": "Frontal Lobe", "primaryIntelligence": "Logical-Mathematical Intelligence"},
        "L3": {"fingerName": "Left Middle", "brainHemisphere": "Right", "brainLobe": "Parietal Lobe", "primaryIntelligence": "Kinesthetic Intelligence (Gross Motor)"},
        "R3": {"fingerName": "Right Middle", "brainHemisphere": "Left", "brainLobe": "Parietal Lobe", "primaryIntelligence": "Kinesthetic Intelligence (Fine Motor)"},
        "L4": {"fingerName": "Left Ring", "brainHemisphere": "Right", "brainLobe": "Temporal Lobe", "primaryIntelligence": "Musical Intelligence"},
        "R4": {"fingerName": "Right Ring", "brainHemisphere": "Left", "brainLobe": "Temporal Lobe", "primaryIntelligence": "Linguistic Intelligence"},
        "L5": {"fingerName": "Left Pinky", "brainHemisphere": "Right", "brainLobe": "Occipital Lobe", "primaryIntelligence": "Visual-Aesthetic Intelligence"},
        "R5": {"fingerName": "Right Pinky", "brainHemisphere": "Left", "brainLobe": "Occipital Lobe", "primaryIntelligence": "Visual-Observation Intelligence"}
    },
    "patternDefinitions": {
        "WT": {"name": "Target Whorl", "group": "Whorl", "learningStyle": "Self-Directed / Cognitive", "traits": ["Goal-oriented", "Decisive", "Strong willpower"]},
        "WS": {"name": "Spiral Whorl", "group": "Whorl", "learningStyle": "Self-Directed / Cognitive", "traits": ["Ambitious", "Self-starter", "Curious"]},
        "WD": {"name": "Double Loop", "group": "Whorl", "learningStyle": "Deliberate / Analytical", "traits": ["Multi-angle thinker", "Cautious", "Perfectionist"]},
        "WC": {"name": "Composite", "group": "Whorl", "learningStyle": "Multi-Faceted Cognitive", "traits": ["Adaptable", "Complex problem solver"]},
        "WP": {"name": "Peacock Eye", "group": "Whorl", "learningStyle": "Expressive / Creative", "traits": ["Influential", "Artistic", "Spontaneous"]},
        "U":  {"name": "Ulnar Loop", "group": "Loop", "learningStyle": "Imitative Learner", "traits": ["Sociable", "Flexible", "Team player"]},
        "R":  {"name": "Radial Loop", "group": "Loop", "learningStyle": "Reverse Thinker", "traits": ["Out of the box", "Critical thinker", "Innovative"]},
        "A":  {"name": "Plain Arch", "group": "Arch", "learningStyle": "Open Learning (Sponge)", "traits": ["Absorptive", "Methodical", "Needs encouragement"]},
        "AT": {"name": "Tented Arch", "group": "Arch", "learningStyle": "Impulsive / High Energy", "traits": ["Enthusiastic", "Fast learner", "Emotionally engaged"]}
    }
}

# ==========================================
# 2. DATA MODELS & ENGINE
# ==========================================
class FingerInput(BaseModel):
    pattern_code: str
    left_rc: int = 0
    right_rc: int = 0

    @property
    def primary_rc(self) -> int:
        return max(self.left_rc, self.right_rc)


class SubjectInput(BaseModel):
    subject_id: str
    fingers: Dict[str, FingerInput]

    @field_validator("fingers")
    def check_10_fingers(cls, v):
        req = {"L1", "L2", "L3", "L4", "L5", "R1", "R2", "R3", "R4", "R5"}
        if req - set(v.keys()):
            raise ValueError(f"Missing fingers: {req - set(v.keys())}")
        return v


def analyze_dmit_data(data: SubjectInput) -> dict:
    tfrc = sum(f.primary_rc for f in data.fingers.values())
    safe_tfrc = max(tfrc, 1)

    breakdown = []
    rankings = []
    left_brain_rc = 0
    right_brain_rc = 0
    vak_rc = {"Visual": 0, "Auditory": 0, "Kinesthetic": 0}

    for code, f_input in data.fingers.items():
        mapping = DMIT_RULES["fingerMappings"][code]
        p_def = DMIT_RULES["patternDefinitions"].get(f_input.pattern_code.upper(), {
            "name": "Ulnar Loop", "group": "Loop", "learningStyle": "Imitative", "traits": ["Adaptive"]
        })

        if code.startswith("R"):
            left_brain_rc += f_input.primary_rc
        else:
            right_brain_rc += f_input.primary_rc

        if code in ["L5", "R5"]: vak_rc["Visual"] += f_input.primary_rc
        elif code in ["L4", "R4"]: vak_rc["Auditory"] += f_input.primary_rc
        elif code in ["L3", "R3"]: vak_rc["Kinesthetic"] += f_input.primary_rc

        pct = round((f_input.primary_rc / safe_tfrc) * 100, 2)

        item = {
            "finger_code": code,
            "finger_name": mapping["fingerName"],
            "brain_hemisphere": mapping["brainHemisphere"],
            "brain_lobe": mapping["brainLobe"],
            "intelligence_area": mapping["primaryIntelligence"],
            "pattern_name": p_def["name"],
            "learning_style": p_def["learningStyle"],
            "primary_ridge_count": f_input.primary_rc,
            "contribution_percentage": pct,
            "personality_traits": p_def["traits"]
        }
        breakdown.append(item)
        rankings.append({
            "finger": code,
            "intelligence": mapping["primaryIntelligence"],
            "lobe": mapping["brainLobe"],
            "contribution_pct": pct
        })

    rankings = sorted(rankings, key=lambda x: x["contribution_pct"], reverse=True)

    # Brain Balance
    total_hemi = max(left_brain_rc + right_brain_rc, 1)
    left_pct = round((left_brain_rc / total_hemi) * 100, 1)
    right_pct = round((right_brain_rc / total_hemi) * 100, 1)
    dominance = "Balanced / Whole Brain" if abs(left_pct - right_pct) <= 4 else ("Left Brain Dominant" if left_pct > right_pct else "Right Brain Dominant")

    # VAK Modality
    total_vak = max(sum(vak_rc.values()), 1)
    vak_pcts = {k: round((v / total_vak) * 100, 1) for k, v in vak_rc.items()}
    dom_vak = max(vak_pcts, key=vak_pcts.get)

    # Quotients
    f_map = {f["finger_code"]: f["primary_ridge_count"] for f in breakdown}
    eq_r = f_map.get("L1", 0) + f_map.get("R1", 0)
    iq_r = f_map.get("R2", 0) + f_map.get("R4", 0) + f_map.get("R5", 0)
    cq_r = f_map.get("L2", 0) + f_map.get("L5", 0)
    aq_r = f_map.get("L3", 0) + f_map.get("R3", 0) + (0.5 * f_map.get("R1", 0))
    sq_r = f_map.get("L4", 0) + (0.5 * f_map.get("L1", 0))
    q_tot = max(eq_r + iq_r + cq_r + aq_r + sq_r, 1)

    quotients = [
        {"code": "EQ", "name": "Emotional", "score": round((eq_r / q_tot) * 100, 1), "color": "#E53E3E"},
        {"code": "IQ", "name": "Intelligence", "score": round((iq_r / q_tot) * 100, 1), "color": "#3182CE"},
        {"code": "AQ", "name": "Adversity", "score": round((aq_r / q_tot) * 100, 1), "color": "#DD6B20"},
        {"code": "CQ", "name": "Creativity", "score": round((cq_r / q_tot) * 100, 1), "color": "#805AD5"},
        {"code": "SQ", "name": "Spiritual", "score": round((sq_r / q_tot) * 100, 1), "color": "#38A169"}
    ]

    return {
        "subject_id": data.subject_id,
        "tfrc": tfrc,
        "intelligence_rankings": rankings,
        "finger_breakdown": breakdown,
        "hemisphere_data": {"left_pct": left_pct, "right_pct": right_pct, "dominance": dominance},
        "vak_data": {"dominant": dom_vak, "scores": vak_pcts},
        "quotient_data": {"list": quotients, "dominant": max(quotients, key=lambda x: x["score"])}
    }

# ==========================================
# 3. CHART GENERATION (BASE64 INLINE)
# ==========================================
def make_radar_chart(rankings: list) -> str:
    order = ["L1", "L2", "L3", "L4", "L5", "R5", "R4", "R3", "R2", "R1"]
    r_map = {item["finger"]: item["contribution_pct"] for item in rankings}
    vals = [r_map.get(k, 0) for k in order]
    
    angles = np.linspace(0, 2 * np.pi, len(order), endpoint=False).tolist()
    vals += [vals[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(4.5, 4.5), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], order, color="#2D3748", size=8, weight="bold")
    ax.plot(angles, vals, color="#3182CE", linewidth=2)
    ax.fill(angles, vals, color="#3182CE", alpha=0.25)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def make_hemisphere_bar(left_pct: float, right_pct: float) -> str:
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh(0, left_pct, color="#2B6CB0", height=0.6)
    ax.barh(0, right_pct, left=left_pct, color="#805AD5", height=0.6)
    ax.set_xlim(0, 100)
    ax.axis("off")
    ax.text(left_pct / 2, 0, f"Left Brain\n{left_pct}%", ha="center", va="center", color="white", weight="bold", size=8)
    ax.text(left_pct + (right_pct / 2), 0, f"Right Brain\n{right_pct}%", ha="center", va="center", color="white", weight="bold", size=8)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

# ==========================================
# 4. EMBEDDED HTML TEMPLATE
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page { size: A4; margin: 15mm; }
  body { font-family: sans-serif; color: #2D3748; font-size: 9pt; }
  .header { border-bottom: 2px solid #2B6CB0; padding-bottom: 6px; margin-bottom: 12px; }
  h1 { color: #1A365D; margin: 0; font-size: 16pt; }
  .box { background: #EDF2F7; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 8pt; }
  th { background: #2B6CB0; color: white; padding: 5px; text-align: left; }
  td { padding: 5px; border-bottom: 1px solid #E2E8F0; }
  .tag { background: #E2E8F0; padding: 2px 4px; border-radius: 3px; font-size: 6.5pt; margin-right: 2px; }
</style>
</head>
<body>
  <div class="header">
    <h1>DMIT Comprehensive Analysis Report</h1>
    <p style="margin:0; color:#4A5568;">Subject ID: <strong>{{ d.subject_id }}</strong> | TFRC: <strong>{{ d.tfrc }}</strong></p>
  </div>

  <div class="box">
    <strong>Brain Dominance:</strong> {{ d.hemisphere_data.dominance }} | 
    <strong>VAK Style:</strong> {{ d.vak_data.dominant }} ({{ d.vak_data.scores[d.vak_data.dominant] }}%) |
    <strong>Dominant Quotient:</strong> {{ d.quotient_data.dominant.name }} ({{ d.quotient_data.dominant.score }}%)
  </div>

  <div style="text-align: center; margin-bottom: 10px;">
    <img src="{{ hemi_img }}" style="width: 70%;" />
  </div>

  <div style="display: flex; justify-content: space-between; align-items: center;">
    <div style="width: 48%; text-align: center;">
      <img src="{{ radar_img }}" style="width: 100%; max-width: 250px;" />
    </div>
    <div style="width: 48%;">
      <strong>Top Intelligences:</strong>
      <table>
        <tr><th>Rank</th><th>Intelligence</th><th>Score %</th></tr>
        {% for item in d.intelligence_rankings[:5] %}
        <tr><td>#{{ loop.index }}</td><td>{{ item.intelligence }}</td><td>{{ item.contribution_pct }}%</td></tr>
        {% endfor %}
      </table>
    </div>
  </div>

  <h3 style="color: #2C5282; margin-top: 15px; margin-bottom: 4px;">10-Finger Dermatoglyphic Breakdown</h3>
  <table>
    <tr><th>Code</th><th>Finger</th><th>Pattern</th><th>RC</th><th>Learning Style</th><th>Traits</th></tr>
    {% for f in d.finger_breakdown %}
    <tr>
      <td><strong>{{ f.finger_code }}</strong></td>
      <td>{{ f.finger_name }}</td>
      <td>{{ f.pattern_name }}</td>
      <td>{{ f.primary_ridge_count }}</td>
      <td>{{ f.learning_style }}</td>
      <td>{% for t in f.personality_traits %}<span class="tag">{{ t }}</span>{% endfor %}</td>
    </tr>
    {% endfor %}
  </table>
</body>
</html>
"""

# ==========================================
# 5. FASTAPI APPLICATION & ENDPOINTS
# ==========================================
app = FastAPI(title="DMIT Automated Cloud API")

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>DMIT Analysis API is Online</h2><p>Visit <a href='/docs'>/docs</a> for the interactive interface.</p>"

@app.post("/api/v1/generate-report")
async def generate_report(
    subject_id: str = Form("STUDENT_001"),
    l1_pattern: str = Form("WP"), l1_rc: int = Form(18),
    l2_pattern: str = Form("R"),  l2_rc: int = Form(14),
    l3_pattern: str = Form("U"),  l3_rc: int = Form(12),
    l4_pattern: str = Form("WD"), l4_rc: int = Form(16),
    l5_pattern: str = Form("U"),  l5_rc: int = Form(11),
    r1_pattern: str = Form("WT"), r1_rc: int = Form(20),
    r2_pattern: str = Form("WT"), r2_rc: int = Form(17),
    r3_pattern: str = Form("U"),  r3_rc: int = Form(13),
    r4_pattern: str = Form("U"),  r4_rc: int = Form(15),
    r5_pattern: str = Form("U"),  r5_rc: int = Form(12),
):
    fingers = {
        "L1": FingerInput(pattern_code=l1_pattern, left_rc=l1_rc),
        "L2": FingerInput(pattern_code=l2_pattern, left_rc=l2_rc),
        "L3": FingerInput(pattern_code=l3_pattern, left_rc=l3_rc),
        "L4": FingerInput(pattern_code=l4_pattern, left_rc=l4_rc),
        "L5": FingerInput(pattern_code=l5_pattern, left_rc=l5_rc),
        "R1": FingerInput(pattern_code=r1_pattern, right_rc=r1_rc),
        "R2": FingerInput(pattern_code=r2_pattern, right_rc=r2_rc),
        "R3": FingerInput(pattern_code=r3_pattern, right_rc=r3_rc),
        "R4": FingerInput(pattern_code=r4_pattern, right_rc=r4_rc),
        "R5": FingerInput(pattern_code=r5_pattern, right_rc=r5_rc),
    }

    results = analyze_dmit_data(SubjectInput(subject_id=subject_id, fingers=fingers))
    radar_img = make_radar_chart(results["intelligence_rankings"])
    hemi_img = make_hemisphere_bar(results["hemisphere_data"]["left_pct"], results["hemisphere_data"]["right_pct"])

    rendered_html = Template(HTML_TEMPLATE).render(d=results, radar_img=radar_img, hemi_img=hemi_img)

    pdf_buffer = io.BytesIO()
    HTML(string=rendered_html).write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=DMIT_Report_{subject_id}.pdf"}
    )
