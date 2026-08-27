import io
import os
import time
import math
import base64
import json
import traceback
from typing import Dict, List, Optional
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jinja2 import Template
from weasyprint import HTML
from google import genai
from google.genai import types
from starlette.datastructures import UploadFile as StarletteUploadFile

# ==============================================================================
# 1. BMD KNOWLEDGE BASE & EXACT DROPDOWN MAPPING
# ==============================================================================
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
        "WT": {"name": "Target / Plain Whorl", "group": "Whorl", "learningStyle": "Self-Directed / Cognitive", "traits": ["Goal-oriented", "Decisive", "Strong willpower"]},
        "WC": {"name": "Double Loop / Composite Whorl", "group": "Whorl", "learningStyle": "Deliberate / Analytical", "traits": ["Multi-angle thinker", "Cautious", "Perfectionist"]},
        "WP": {"name": "Peacock's Eye", "group": "Whorl", "learningStyle": "Expressive / Creative", "traits": ["Influential", "Artistic", "Spontaneous"]},
        "U":  {"name": "Ulnar Loop", "group": "Loop", "learningStyle": "Imitative Learner", "traits": ["Sociable", "Flexible", "Team player"]},
        "R":  {"name": "Radial Loop", "group": "Loop", "learningStyle": "Reverse Thinker", "traits": ["Out of the box", "Critical thinker", "Innovative"]},
        "A":  {"name": "Simple Arch", "group": "Arch", "learningStyle": "Open Learning (Sponge)", "traits": ["Absorptive", "Methodical", "Needs encouragement"]},
        "AT": {"name": "Tented Arch", "group": "Arch", "learningStyle": "Impulsive / High Energy", "traits": ["Enthusiastic", "Fast learner", "Emotionally engaged"]}
    }
}

# ==============================================================================
# 2. AUTOMATED VISION CLASSIFIER
# ==============================================================================
def classify_fingerprint_image_ai(image_bytes: bytes, api_key: str, finger_code: str = "Unknown", max_retries: int = 4) -> tuple[str, int, int]:
    client = genai.Client(api_key=api_key)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    prompt = f"""
    You are an expert Dermatoglyphics and Fingerprint Classification specialist based on the BMD Counseling system.
    Analyze this fingerprint image for finger position: {finger_code}.

    1. Classify the pattern into EXACTLY ONE of these valid system codes:
       - WT: Target / Plain Whorl (Concentric rings or spiral whorl with 2 deltas)
       - WC: Double Loop / Composite Whorl (S-shape / interlocking loops with 2 deltas)
       - WP: Peacock's Eye (Small center circular whorl inside a loop)
       - U: Ulnar Loop (Flows towards pinky side, 1 delta)
       - R: Radial Loop (Flows towards thumb side, 1 delta)
       - A: Simple Arch (Wave-like ridges, 0 deltas)
       - AT: Tented Arch (Sharp upward spike <90 degrees, 0 deltas)

    2. Calculate Ridge Counts (RC):
       - If Whorl (WT, WC, WP): Provide left delta ridge count and right delta ridge count (usually 8-22).
       - If Loop (U, R): Provide ridge count on the side with delta, the other side is 0.
       - If Arch (A, AT): Both counts are 0.

    Return ONLY a JSON object formatted exactly as:
    {{"pattern_code": "WT", "ridge_count_left": 14, "ridge_count_right": 15}}
    """

    for attempt in range(max_retries):
        try:
            time.sleep(1.0)
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.1
                ),
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)
            code = str(data.get("pattern_code", "U")).upper().strip()
            
            # Normalize code to the exact dropdown values
            if code in ["WS", "PLAIN", "TARGET"]:
                code = "WT"
            elif code in ["WD", "DOUBLE_LOOP", "COMPOSITE"]:
                code = "WC"
            elif code not in DMIT_RULES["patternDefinitions"]:
                code = "U"

            rc_l = int(data.get("ridge_count_left", data.get("rc_left", 0)))
            rc_r = int(data.get("ridge_count_right", data.get("rc_right", 0)))

            if rc_l == 0 and rc_r == 0:
                est = int(data.get("estimated_ridge_count", data.get("ridge_count", 0)))
                if code in ["WT", "WC", "WP"]:
                    rc_l, rc_r = est, est
                elif code == "U":
                    rc_l, rc_r = est, 0
                elif code == "R":
                    rc_l, rc_r = 0, est

            return code, max(rc_l, 0), max(rc_r, 0)

        except Exception as e:
            err_msg = str(e)
            print(f"[Attempt {attempt+1}] Vision classification error: {err_msg}")
            if attempt < max_retries - 1:
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    time.sleep(12.0)
                else:
                    time.sleep((2 ** attempt) * 2)
                continue
            raise e

# ==============================================================================
# 3. METRICS, CHARTS & PDF COMPILER
# ==============================================================================
def compile_dmit_report(subject_id: str, finger_results: dict) -> bytes:
    tfrc = sum(f["rc"] for f in finger_results.values())
    safe_tfrc = max(tfrc, 1)

    breakdown = []
    rankings = []
    left_brain_rc, right_brain_rc = 0, 0
    vak_rc = {"Visual": 0, "Auditory": 0, "Kinesthetic": 0}

    for code, f_data in finger_results.items():
        mapping = DMIT_RULES["fingerMappings"].get(code, {
            "fingerName": code, "brainHemisphere": "Unknown", "brainLobe": "Unknown", "primaryIntelligence": "General"
        })
        p_def = DMIT_RULES["patternDefinitions"].get(f_data["pattern"], DMIT_RULES["patternDefinitions"]["U"])

        rc = f_data["rc"]
        if code.startswith("R"):
            left_brain_rc += rc
        else:
            right_brain_rc += rc

        if code in ["L5", "R5"]: vak_rc["Visual"] += rc
        elif code in ["L4", "R4"]: vak_rc["Auditory"] += rc
        elif code in ["L3", "R3"]: vak_rc["Kinesthetic"] += rc

        pct = round((rc / safe_tfrc) * 100, 2)

        breakdown.append({
            "finger_code": code,
            "finger_name": mapping["fingerName"],
            "brain_hemisphere": mapping["brainHemisphere"],
            "brain_lobe": mapping["brainLobe"],
            "intelligence_area": mapping["primaryIntelligence"],
            "pattern_name": p_def["name"],
            "learning_style": p_def["learningStyle"],
            "primary_ridge_count": rc,
            "contribution_percentage": pct,
            "personality_traits": p_def["traits"]
        })

        rankings.append({
            "finger": code,
            "intelligence": mapping["primaryIntelligence"],
            "lobe": mapping["brainLobe"],
            "contribution_pct": pct
        })

    rankings = sorted(rankings, key=lambda x: x["contribution_pct"], reverse=True)

    tot_hemi = max(left_brain_rc + right_brain_rc, 1)
    left_pct = round((left_brain_rc / tot_hemi) * 100, 1)
    right_pct = round((right_brain_rc / tot_hemi) * 100, 1)
    dominance = "Balanced Brain" if abs(left_pct - right_pct) <= 4 else ("Left Brain Dominant" if left_pct > right_pct else "Right Brain Dominant")

    tot_vak = max(sum(vak_rc.values()), 1)
    vak_scores = {k: round((v / tot_vak) * 100, 1) for k, v in vak_rc.items()}
    dom_vak = max(vak_scores, key=vak_scores.get)

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
    buf_radar = io.BytesIO()
    plt.savefig(buf_radar, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    radar_img = f"data:image/png;base64,{base64.b64encode(buf_radar.getvalue()).decode()}"

    fig2, ax2 = plt.subplots(figsize=(6, 1.2))
    ax2.barh(0, left_pct, color="#2B6CB0", height=0.6)
    ax2.barh(0, right_pct, left=left_pct, color="#805AD5", height=0.6)
    ax2.set_xlim(0, 100)
    ax2.axis("off")
    ax2.text(left_pct / 2, 0, f"Left Brain\n{left_pct}%", ha="center", va="center", color="white", weight="bold", size=8)
    ax2.text(left_pct + (right_pct / 2), 0, f"Right Brain\n{right_pct}%", ha="center", va="center", color="white", weight="bold", size=8)
    buf_hemi = io.BytesIO()
    plt.savefig(buf_hemi, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig2)
    hemi_img = f"data:image/png;base64,{base64.b64encode(buf_hemi.getvalue()).decode()}"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"><style>
      @page { size: A4; margin: 15mm; }
      body { font-family: sans-serif; color: #2D3748; font-size: 9pt; }
      .header { border-bottom: 2px solid #2B6CB0; padding-bottom: 6px; margin-bottom: 12px; }
      h1 { color: #1A365D; margin: 0; font-size: 16pt; }
      .box { background: #EDF2F7; padding: 8px 12px; border-radius: 4px; margin-bottom: 12px; }
      table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 8pt; }
      th { background: #2B6CB0; color: white; padding: 5px; text-align: left; }
      td { padding: 5px; border-bottom: 1px solid #E2E8F0; }
      .tag { background: #E2E8F0; padding: 2px 4px; border-radius: 3px; font-size: 6.5pt; margin-right: 2px; }
    </style></head>
    <body>
      <div class="header">
        <h1>DMIT Automated AI Analysis Report</h1>
        <p style="margin:0; color:#4A5568;">Subject ID: <strong>{{ subject_id }}</strong> | Total Ridge Count (TFRC): <strong>{{ tfrc }}</strong></p>
      </div>

      <div class="box">
        <strong>Brain Balance:</strong> {{ dominance }} | 
        <strong>VAK Style:</strong> {{ dom_vak }} ({{ vak_scores[dom_vak] }}%)
      </div>

      <div style="text-align: center; margin-bottom: 10px;">
        <img src="{{ hemi_img }}" style="width: 70%;" />
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div style="width: 48%; text-align: center;"><img src="{{ radar_img }}" style="width: 100%; max-width: 250px;" /></div>
        <div style="width: 48%;">
          <strong>Top Innate Intelligences:</strong>
          <table>
            <tr><th>Rank</th><th>Intelligence</th><th>Score %</th></tr>
            {% for item in rankings[:5] %}
            <tr><td>#{{ loop.index }}</td><td>{{ item.intelligence }}</td><td>{{ item.contribution_pct }}%</td></tr>
            {% endfor %}
          </table>
        </div>
      </div>

      <h3 style="color: #2C5282; margin-top: 15px; margin-bottom: 4px;">AI-Detected 10-Finger Pattern Breakdown</h3>
      <table>
        <tr><th>Code</th><th>Finger</th><th>Detected Pattern</th><th>RC</th><th>Learning Style</th><th>Traits</th></tr>
        {% for f in breakdown %}
        <tr>
          <td><strong>{{ f.finger_code }}</strong></td>
          <td>{{ f.finger_name }}</td>
          <td><strong>{{ f.pattern_name }}</strong></td>
          <td>{{ f.primary_ridge_count }}</td>
          <td>{{ f.learning_style }}</td>
          <td>{% for t in f.personality_traits %}<span class="tag">{{ t }}</span>{% endfor %}</td>
        </tr>
        {% endfor %}
      </table>
    </body></html>
    """
    rendered = Template(html_template).render(
        subject_id=subject_id, tfrc=tfrc, dominance=dominance,
        dom_vak=dom_vak, vak_scores=vak_scores, rankings=rankings,
        breakdown=breakdown, radar_img=radar_img, hemi_img=hemi_img
    )
    pdf_buf = io.BytesIO()
    HTML(string=rendered).write_pdf(pdf_buf)
    return pdf_buf.getvalue()

# ==============================================================================
# 4. FASTAPI APP & UNIVERSAL RESPONSE ROUTER
# ==============================================================================
app = FastAPI(title="DMIT Automated AI Scanner API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>DMIT Automated Vision API is Online</h2><p>Visit <a href='/docs'>/docs</a> to upload fingerprint photos.</p>"

@app.head("/")
def head_root():
    return JSONResponse(content={"status": "ok"})

@app.get("/classify")
def test_classify_connection():
    return JSONResponse(content={"status": "ok", "message": "Classification service is reachable"})

@app.post("/classify")
@app.post("/")
async def classify_single_finger_safe(request: Request):
    api_key = request.headers.get("x-api-key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key missing. Set GEMINI_API_KEY in Render Environment Variables.")

    image_bytes = None
    finger_code = "Unknown"

    try:
        body_bytes = await request.body()
        content_type = request.headers.get("content-type", "").lower()

        if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
            form_data = await request.form()
            for key, val in form_data.items():
                if isinstance(val, StarletteUploadFile):
                    image_bytes = await val.read()
                elif isinstance(val, (bytes, bytearray)):
                    image_bytes = bytes(val)
                elif key.lower() in ["finger", "finger_code", "position", "id", "name"]:
                    finger_code = str(val)

        elif "application/json" in content_type:
            body_json = json.loads(body_bytes.decode("utf-8", errors="ignore") or "{}")
            finger_code = body_json.get("finger", body_json.get("finger_code", "Unknown"))
            data_str = body_json.get("image") or body_json.get("file") or body_json.get("image_base64")
            if data_str:
                if "," in data_str:
                    data_str = data_str.split(",")[1]
                image_bytes = base64.b64decode(data_str)

        if not image_bytes and len(body_bytes) > 100:
            image_bytes = body_bytes

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed reading payload: {str(e)}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No fingerprint image data received.")

    try:
        pattern_code, rc_left, rc_right = classify_fingerprint_image_ai(
            image_bytes=image_bytes,
            api_key=api_key,
            finger_code=finger_code
        )

        display_name = DMIT_RULES["patternDefinitions"].get(pattern_code, {}).get("name", "Ulnar Loop")
        primary_rc = max(rc_left, rc_right)
        code_upper = pattern_code.upper()
        code_lower = pattern_code.lower()

        # Build base payload with all possible key and casing combinations
        core_data = {
            # Uppercase option value codes (WT, WC, WP, U, R, A, AT)
            "pattern": code_upper,
            "pattern_type": code_upper,
            "pattern_code": code_upper,
            "patternType": code_upper,
            "patternCode": code_upper,
            "code": code_upper,
            "value": code_upper,
            "type": code_upper,
            "finger_pattern": code_upper,
            "fingerPattern": code_upper,
            "classification": code_upper,
            "detected_pattern": code_upper,

            # Lowercase codes (wt, wc, wp, u, r, a, at)
            "pattern_lower": code_lower,
            "pattern_slug": code_lower,

            # Display name labels
            "pattern_name": display_name,
            "patternName": display_name,
            "name": display_name,
            "label": display_name,

            # Ridge counts
            "ridge_count": primary_rc,
            "ridgeCount": primary_rc,
            "rc": primary_rc,
            "ridge_count_l": rc_left,
            "ridge_count_left": rc_left,
            "ridgeCountL": rc_left,
            "ridgeCountLeft": rc_left,
            "rc_l": rc_left,
            "rc_left": rc_left,
            "rcL": rc_left,
            "left_rc": rc_left,
            "leftRc": rc_left,
            "left_ridge_count": rc_left,
            "leftRidgeCount": rc_left,
            "ridge_count_r": rc_right,
            "ridge_count_right": rc_right,
            "ridgeCountR": rc_right,
            "ridgeCountRight": rc_right,
            "rc_r": rc_right,
            "rc_right": rc_right,
            "rcR": rc_right,
            "right_rc": rc_right,
            "rightRc": rc_right,
            "right_ridge_count": rc_right,
            "rightRidgeCount": rc_right
        }

        # Provide flat root keys AND nested wrapper objects
        result_payload = {
            **core_data,
            "data": core_data,
            "result": core_data,
            "prediction": core_data,
            "fingerprint": core_data
        }

        print(f"[Success] Set {finger_code} -> value='{code_upper}' | RC(L): {rc_left}, RC(R): {rc_right}")
        return JSONResponse(content=result_payload)

    except Exception as e:
        print(f"[Classification Error]: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/api/v1/auto-scan-and-generate-report")
async def auto_scan_and_generate_report(
    api_key: Optional[str] = Form(None, description="Your Google AI Studio Gemini API Key (or set via Render Env)"),
    subject_id: str = Form("STUDENT_001", description="Subject/Client Name or ID"),
    l1: UploadFile = File(..., description="Left Thumb Image"),
    l2: UploadFile = File(..., description="Left Index Image"),
    l3: UploadFile = File(..., description="Left Middle Image"),
    l4: UploadFile = File(..., description="Left Ring Image"),
    l5: UploadFile = File(..., description="Left Pinky Image"),
    r1: UploadFile = File(..., description="Right Thumb Image"),
    r2: UploadFile = File(..., description="Right Index Image"),
    r3: UploadFile = File(..., description="Right Middle Image"),
    r4: UploadFile = File(..., description="Right Ring Image"),
    r5: UploadFile = File(..., description="Right Pinky Image"),
):
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    uploads = {
        "L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5,
        "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5
    }

    finger_results = {}

    for code, file_obj in uploads.items():
        try:
            image_bytes = await file_obj.read()
            pattern_code, rc_l, rc_r = classify_fingerprint_image_ai(image_bytes, effective_api_key, code)
            finger_results[code] = {"pattern": pattern_code, "rc": max(rc_l, rc_r)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to analyze image for {code}: {str(e)}")

    pdf_bytes = compile_dmit_report(subject_id, finger_results)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=DMIT_Report_{subject_id}.pdf"}
    )
