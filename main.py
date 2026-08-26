import io
import json
import cv2
import numpy as np
from typing import Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from jinja2 import Environment, FileSystemLoader
from weasyprint import HTML

# Import models & generators created in previous steps
from chart_generator import generate_radar_chart_base64
from hemisphere_chart import calculate_hemisphere_split, generate_hemisphere_bar_base64
from vak_generator import calculate_vak_distribution, generate_vak_donut_chart_base64
from quotients_generator import calculate_quotients, generate_quotients_chart_base64
from dmit_engine import DMITAnalysisEngine, SubjectInput, FingerInput

app = FastAPI(
    title="DMIT Analysis & PDF Report API",
    description="Automated fingerprint analysis and PDF generation pipeline.",
    version="1.0.0"
)

# Load the DMIT JSON ruleset configuration on startup
with open("dmit_rules.json", "r") as f:
    RULES_CONFIG = json.load(f)

engine = DMITAnalysisEngine(ruleset_dict=RULES_CONFIG)
jinja_env = Environment(loader=FileSystemLoader("."))


# --- Computer Vision Preprocessing & Classification Stub ---
def process_fingerprint_image(image_bytes: bytes) -> tuple[str, int, int]:
    """
    Decodes the raw image, enhances ridge contrast using OpenCV,
    and runs classification/ridge counting.
    
    (In production, replace the heuristic/mock return with your trained ONNX/PyTorch model)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image file.")

    # 1. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # 2. Otsu's Thresholding / Ridge binarization
    _, binarized = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Placeholder for Model Inference: Returns (pattern_code, left_rc, right_rc)
    # e.g., model.predict(binarized) -> "WT", 16, 18
    # For demonstration, returning representative defaults:
    return "WT", 16, 17


# --- Main API Endpoint ---
@app.post(
    "/api/v1/dmit/generate-report",
    summary="Upload 10 finger photos and generate DMIT PDF report",
    response_class=StreamingResponse
)
async def generate_dmit_report(
    subject_id: str = Form(..., description="Unique Subject/Client ID"),
    l1: UploadFile = File(..., description="Left Thumb"),
    l2: UploadFile = File(..., description="Left Index"),
    l3: UploadFile = File(..., description="Left Middle"),
    l4: UploadFile = File(..., description="Left Ring"),
    l5: UploadFile = File(..., description="Left Pinky"),
    r1: UploadFile = File(..., description="Right Thumb"),
    r2: UploadFile = File(..., description="Right Index"),
    r3: UploadFile = File(..., description="Right Middle"),
    r4: UploadFile = File(..., description="Right Ring"),
    r5: UploadFile = File(..., description="Right Pinky"),
):
    uploads = {
        "L1": l1, "L2": l2, "L3": l3, "L4": l4, "L5": l5,
        "R1": r1, "R2": r2, "R3": r3, "R4": r4, "R5": r5,
    }

    finger_inputs: Dict[str, FingerInput] = {}

    # 1. Process all 10 uploaded images through OpenCV/Vision Pipeline
    for code, file_obj in uploads.items():
        try:
            content = await file_obj.read()
            pattern_code, left_rc, right_rc = process_fingerprint_image(content)
            finger_inputs[code] = FingerInput(
                pattern_code=pattern_code,
                left_rc=left_rc,
                right_rc=right_rc
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to process image for finger {code}: {str(e)}"
            )

    # 2. Run DMIT Engine Metrics & Calculations
    user_payload = SubjectInput(subject_id=subject_id, fingers=finger_inputs)
    report = engine.analyze(user_payload)
    report_dict = report.model_dump()

    # 3. Generate Analytical Visualizations (Base64)
    radar_chart = generate_radar_chart_base64(report_dict["intelligence_rankings"])
    
    hemisphere_data = calculate_hemisphere_split(report_dict["finger_breakdown"])
    hemisphere_chart = generate_hemisphere_bar_base64(
        hemisphere_data["left_brain_pct"], 
        hemisphere_data["right_brain_pct"]
    )
    
    vak_data = calculate_vak_distribution(report_dict["finger_breakdown"])
    vak_chart = generate_vak_donut_chart_base64(vak_data)
    
    quotient_data = calculate_quotients(report_dict["finger_breakdown"])
    quotients_chart = generate_quotients_chart_base64(quotient_data)

    # 4. Render HTML Template
    template = jinja_env.get_template("report_template.html")
    rendered_html = template.render(
        report=report_dict,
        radar_chart_base64=radar_chart,
        hemisphere_chart_base64=hemisphere_chart,
        hemisphere_data=hemisphere_data,
        vak_chart_base64=vak_chart,
        vak_data=vak_data,
        quotients_chart_base64=quotients_chart,
        quotient_data=quotient_data
    )

    # 5. Compile PDF in memory using WeasyPrint
    pdf_buffer = io.BytesIO()
    HTML(string=rendered_html, base_url=".").write_pdf(pdf_buffer)
    pdf_buffer.seek(0)

    # 6. Stream PDF response directly to client
    filename = f"DMIT_Report_{subject_id}.pdf"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}"
    }

    return StreamingResponse(
        pdf_buffer,
        media_type="application/pdf",
        headers=headers
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)