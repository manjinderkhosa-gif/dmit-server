import io
import os
import time
import math
import base64
import json
import traceback
from typing import Dict, List, Optional
from PIL import Image, ImageEnhance, ImageFilter
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
# 1. BMD KNOWLEDGE BASE & EXACT FORMULA DEFINITIONS
# ==============================================================================
DMIT_RULES = {
    "fingerMappings": {
        "L1": {"fingerName": "Left Thumb", "brainHemisphere": "Right", "brainLobe": "Prefrontal Lobe", "primaryIntelligence": "Interpersonal Intelligence"},
        "R1": {"fingerName": "Right Thumb", "brainHemisphere": "Left", "brainLobe": "Prefrontal Lobe", "primaryIntelligence": "Intrapersonal Intelligence"},
        "L2": {"fingerName": "Left Index", "brainHemisphere": "Right", "brainLobe": "Frontal Lobe", "primaryIntelligence": "Spatial Intelligence"},
        "R2": {"fingerName": "Right Index", "brainHemisphere": "Left", "brainLobe": "Frontal Lobe", "primaryIntelligence": "Logical Intelligence"},
        "L3": {"fingerName": "Left Middle", "brainHemisphere": "Right", "brainLobe": "Parietal Lobe", "primaryIntelligence": "Kinesthetic Intelligence"},
        "R3": {"fingerName": "Right Middle", "brainHemisphere": "Left", "brainLobe": "Parietal Lobe", "primaryIntelligence": "Kinesthetic Intelligence"},
        "L4": {"fingerName": "Left Ring", "brainHemisphere": "Right", "brainLobe": "Temporal Lobe", "primaryIntelligence": "Musical Intelligence"},
        "R4": {"fingerName": "Right Ring", "brainHemisphere": "Left", "brainLobe": "Temporal Lobe", "primaryIntelligence": "Linguistic Intelligence"},
        "L5": {"fingerName": "Left Pinky", "brainHemisphere": "Right", "brainLobe": "Occipital Lobe", "primaryIntelligence": "Visual Intelligence"},
        "R5": {"fingerName": "Right Pinky", "brainHemisphere": "Left", "brainLobe": "Occipital Lobe", "primaryIntelligence": "Naturalistic Intelligence"}
    },
    "patternDefinitions": {
        "WT": {"name": "Target / Plain Whorl", "group": "Whorl", "learningStyle": "Cognitive", "traits": ["Goal-driven", "High concentration", "Decisive"]},
        "WC": {"name": "Double Loop / Composite Whorl", "group": "Whorl", "learningStyle": "Cognitive", "traits": ["Multi-perspective", "Adaptable negotiator", "Analytical"]},
        "WP": {"name": "Peacock's Eye", "group": "Whorl", "learningStyle": "Cognitive", "traits": ["Artistic flair", "Intuitive perception", "Refined leadership"]},
        "U":  {"name": "Ulnar Loop", "group": "Loop", "learningStyle": "Imitative", "traits": ["Cooperative", "Environment-absorbent", "Empathetic team worker"]},
        "R":  {"name": "Radial Loop", "group": "Loop", "learningStyle": "Reverse Thinking", "traits": ["Unconventional thinker", "Critical inquiry", "Independent logic"]},
        "A":  {"name": "Simple Arch", "group": "Arch", "learningStyle": "Open Learning", "traits": ["Absorptive sponge", "Methodical", "Step-by-step learner"]},
        "AT": {"name": "Tented Arch", "group": "Arch", "learningStyle": "Open Learning", "traits": ["Enthusiastic", "High energy", "Fast burst learner"]}
    }
}

# ==============================================================================
# 2. IMAGE PREPROCESSING (CONTRAST & RIDGE SHARPENING)
# ==============================================================================
def preprocess_fingerprint_image(image_bytes: bytes) -> Image.Image:
    """Enhance micro-ridge visibility and delta contrast before sending to AI."""
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    enhancer = ImageEnhance.Contrast(img)
    img_contrasted = enhancer.enhance(2.2)
    img_sharpened = img_contrasted.filter(ImageFilter.SHARPEN)
    return img_sharpened.convert("RGB")

# ==============================================================================
# 3. HIGH-PRECISION VISION CLASSIFIER & RIDGE COUNTER
# ==============================================================================
def classify_fingerprint_image_ai(image_bytes: bytes, api_key: str, finger_code: str = "Unknown", max_retries: int = 4) -> tuple[str, int, int]:
    client = genai.Client(api_key=api_key)
    processed_image = preprocess_fingerprint_image(image_bytes)

    is_left_hand = finger_code.upper().startswith("L") or "LEFT" in finger_code.upper()

    prompt = f"""
    You are an expert Forensic Dermatoglyphics and Henry Classification specialist analyzing this fingerprint image for position: {finger_code}.

    ANALYTICAL PROCEDURE:
    1. LOCATE ANATOMICAL LANDMARKS:
       - Find the CORE (center loop crest, spiral vortex, or central ring).
       - Find the DELTA(S) (triangular triradius forks where ridges diverge).

    2. CLASSIFY PATTERN CODE:
       - WHORL (2 Deltas present):
         * WP: Peacock's Eye (Small circle/eye nestled inside a loop).
         * WT: Target / Plain Whorl (Concentric rings or continuous spiral).
         * WC: Double Loop / Composite (Two distinct interlocking loops / S-shape).
       - LOOP (1 Delta present, 180° core recurve):
         * Hand context: This is the {'LEFT' if is_left_hand else 'RIGHT'} Hand.
         * For Left Hand: Ridges opening to the right (pinky side) = 'U' (Ulnar Loop); opening to the left (thumb side) = 'R' (Radial Loop).
         * For Right Hand: Ridges opening to the left (pinky side) = 'U' (Ulnar Loop); opening to the right (thumb side) = 'R' (Radial Loop).
       - ARCH (0 Deltas present):
         * A: Simple Arch (Wave-like ridges across print).
         * AT: Tented Arch (Sharp upward tent spike < 90°).

    3. PRECISION RIDGE COUNTING (RC):
       - Count every single friction ridge line crossing the straight vector from Core to Delta.
       - Include fine micro-ridges near the delta junction (typical loop/whorl counts range between 14 and 22).
       - For Whorls (WT, WC, WP): Provide left delta count and right delta count.
       - For Loops (U, R): Provide the count for the side with delta; the other side MUST be 0.
       - For Arches (A, AT): Both counts MUST be 0.

    Return ONLY a JSON object:
    {{"pattern_code": "WT", "ridge_count_left": 16, "ridge_count_right": 17}}
    """

    for attempt in range(max_retries):
        try:
            time.sleep(1.0)
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=[prompt, processed_image],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0
                ),
            )

            raw_text = response.text.strip()
            data = json.loads(raw_text)
            code = str(data.get("pattern_code", "U")).upper().strip()

            if code in ["WS", "PLAIN", "TARGET"]:
                ui_pattern = "WT"
            elif code == "WP":
                ui_pattern = "WP"
            elif code in ["WC", "WD", "DOUBLE_LOOP", "COMPOSITE"]:
                ui_pattern = "WC"
            elif code in ["U", "R", "A", "AT"]:
                ui_pattern = code
            else:
                ui_pattern = "U"

            rc_l = int(data.get("ridge_count_left", data.get("rc_left", 0)))
            rc_r = int(data.get("ridge_count_right", data.get("rc_right", 0)))

            if ui_pattern in ["A", "AT"]:
                rc_l, rc_r = 0, 0
            elif ui_pattern in ["WT", "WC", "WP"]:
                if rc_l == 0 and rc_r > 0: rc_l = rc_r
                if rc_r == 0 and rc_l > 0: rc_r = rc_l

            return ui_pattern, max(rc_l, 0), max(rc_r, 0)

        except Exception as e:
            err_msg = str(e)
            print(f"[Attempt {attempt+1}] Vision error for {finger_code}: {err_msg}")
            if attempt < max_retries - 1:
                time.sleep(12.0 if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) else 2.0)
                continue
            raise e

# ==============================================================================
# 4. BMD METRICS, CHARTS & PDF COMPILER
# ==============================================================================
def compile_dmit_report(subject_id: str, finger_results: dict) -> bytes:
    tfrc = sum(f["rc"] for f in finger_results.values())
    safe_tfrc = max(tfrc, 1)

    breakdown = []
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

    # BMD 4 Core Quotients
    r1_rc = finger_results.get("R1", {}).get("rc", 0)
    l1_rc = finger_results.get("L1", {}).get("rc", 0)
    r4_rc = finger_results.get("R4", {}).get("rc", 0)
    r2_rc = finger_results.get("R2", {}).get("rc", 0)
    l2_rc = finger_results.get("L2", {}).get("rc", 0)
    l4_rc = finger_results.get("L4", {}).get("rc", 0)
    l3_rc = finger_results.get("L3", {}).get("rc", 0)
    r3_rc = finger_results.get("R3", {}).get("rc", 0)
    r5_rc = finger_results.get("R5", {}).get("rc", 0)

    quotients = {
        "EQ": r1_rc + l1_rc,
        "IQ": r4_rc + r2_rc,
        "CQ": l2_rc + l4_rc,
        "AQ": l3_rc + r3_rc + r5_rc
    }
    q_sum = max(sum(quotients.values()), 1)
    q_pcts = {k: round((v / q_sum) * 100, 1) for k, v in quotients.items()}

    tot_hemi = max(left_brain_rc + right_brain_rc, 1)
    left_pct = round((left_brain_rc / tot_hemi) * 100, 1)
    right_pct = round((right_brain_rc / tot_hemi) * 100, 1)
    dominance = "Balanced Brain" if abs(left_pct - right_pct) <= 4 else ("Left Brain Dominant" if left_pct > right_pct else "Right Brain Dominant")

    tot_vak = max(sum(vak_rc.values()), 1)
    vak_scores = {k: round((v / tot_vak) * 100, 1) for k, v in vak_rc.items()}
    dom_vak = max(vak_scores, key=vak_scores.get)

    order = ["L1", "L2", "L3", "L4", "L5", "R5", "R4", "R3", "R2", "R1"]
    r_map = {item["finger_code"]: item["contribution_percentage"] for item in breakdown}
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
        <h1>DMIT Assessment Report (BMD Standard)</h1>
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
          <strong>Four Core Quotients:</strong>
          <table>
            <tr><th>Quotient</th><th>Score %</th></tr>
            <tr><td>EQ (Emotional)</td><td>{{ q_pcts['EQ'] }}%</td></tr>
            <tr><td>IQ (Intelligence)</td><td>{{ q_pcts['IQ'] }}%</td></tr>
            <tr><td>CQ (Creative)</td><td>{{ q_pcts['CQ'] }}%</td></tr>
            <tr><td>AQ (Adversity)</td><td>{{ q_pcts['AQ'] }}%</td></tr>
          </table>
        </div>
      </div>

      <h3 style="color: #2C5282; margin-top: 15px; margin-bottom: 4px;">10-Finger Dermatoglyphic Breakdown</h3>
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
        dom_vak=dom_vak, vak_scores=vak_scores, q_pcts=q_pcts,
        breakdown=breakdown, radar_img=radar_img, hemi_img=hemi_img
    )
    pdf_buf = io.BytesIO()
    HTML(string=rendered).write_pdf(pdf_buf)
    return pdf_buf.getvalue()

# ==============================================================================
# 5. FASTAPI APP & ENDPOINTS
# ==============================================================================
app = FastAPI(title="DMIT Automated Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def root():
    return "<h2>DMIT Automated Vision API is Online</h2><p>Visit <a href='/docs'>/docs</a> to view endpoints.</p>"

@app.head("/")
def head_root():
    return JSONResponse(content={"status": "ok"})

@app.get("/classify")
def test_classify_connection():
    return JSONResponse(content={"status": "ok", "message": "Service is reachable"})

@app.post("/classify")
@app.post("/")
async def classify_single_finger_safe(request: Request):
    api_key = request.headers.get("x-api-key") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API Key missing. Set GEMINI_API_KEY in Render.")

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
        raise HTTPException(status_code=400, detail=f"Failed reading upload: {str(e)}")

    if not image_bytes:
        raise HTTPException(status_code=400, detail="No image bytes received.")

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

        core_data = {
            "pattern": code_upper,
            "pattern_type": code_upper,
            "pattern_code": code_upper,
            "patternType": code_upper,
            "patternCode": code_upper,
            "code": code_upper,
            "value": code_upper,
            "type": code_upper,
            "finger_pattern": code_upper,
            "classification": code_upper,

            "pattern_lower": code_lower,
            "pattern_name": display_name,
            "name": display_name,
            "label": display_name,

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
            "ridge_count_r": rc_right,
            "ridge_count_right": rc_right,
            "ridgeCountR": rc_right,
            "ridgeCountRight": rc_right,
            "rc_r": rc_right,
            "rc_right": rc_right,
            "rcR": rc_right,
            "right_rc": rc_right,
            "rightRc": rc_right,
            "right_ridge_count": rc_right
        }

        result_payload = {
            **core_data,
            "data": core_data,
            "result": core_data,
            "prediction": core_data,
            "fingerprint": core_data
        }

        print(f"[Success] {finger_code} -> {code_upper} ({display_name}) | RC(L): {rc_left}, RC(R): {rc_right}")
        return JSONResponse(content=result_payload)

    except Exception as e:
        print(f"[Classification Error]: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/api/v1/auto-scan-and-generate-report")
async def auto_scan_and_generate_report(
    api_key: Optional[str] = Form(None),
    subject_id: str = Form("STUDENT_001"),
    l1: UploadFile = File(...), l2: UploadFile = File(...), l3: UploadFile = File(...),
    l4: UploadFile = File(...), l5: UploadFile = File(...), r1: UploadFile = File(...),
    r2: UploadFile = File(...), r3: UploadFile = File(...), r4: UploadFile = File(...),
    r5: UploadFile = File(...)
):
    effective_api_key = api_key or os.environ.get("GEMINI_API_KEY")
    if not effective_api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required.")

    uploads = {"L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5, "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5}
    finger_results = {}

    for code, file_obj in uploads.items():
        try:
            image_bytes = await file_obj.read()
            pattern_code, rc_l, rc_r = classify_fingerprint_image_ai(image_bytes, effective_api_key, code)
            finger_results[code] = {"pattern": pattern_code, "rc": max(rc_l, rc_r)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to analyze {code}: {str(e)}")

    pdf_bytes = compile_dmit_report(subject_id, finger_results)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=DMIT_Report_{subject_id}.pdf"}
    )
