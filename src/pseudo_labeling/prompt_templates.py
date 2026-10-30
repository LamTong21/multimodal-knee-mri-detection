import json

LABEL_COLS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture"
]

SCHEMA_TEMPLATE = json.dumps({col: 0 for col in LABEL_COLS})

SYSTEM_PROMPT = """You are an expert board-certified musculoskeletal radiologist.
Analyze knee MRI clinical reports across multiple languages (English, Spanish, French, German, Turkish, Greek, Croatian/Serbian) and extract binary diagnostic labels (0 = negative/absent/normal, 1 = positive/present/abnormal) for 12 conditions.

Critical Guidelines:
1. High Sensitivity Directive:
- Any grade of chondromalacia, cartilage thinning, fissuring, defect, or osteophytes -> 1 for respective compartment (Medial OA, Lateral OA, PF OA).
- Bone marrow edema (BME), bone bruise, subchondral edema -> 1 for Contusion.
- Cortical disruption, trabecular impaction, stress injury, microfracture -> 1 for Fracture.
- Mild/borderline/partial tears, sprains, or signal alterations -> 1.
2. Output Format:
- Return ONLY a valid, single-line JSON object matching the exact keys provided.
- Do NOT output explanations, medical notes, or repeated report text."""

ALIAS_MAP = {
    "bonecontusion": "contusion",
    "bonemarrowedema": "contusion",
    "marrowedema": "contusion",
    "bonebruise": "contusion",
    "bonefracture": "fracture",
    "corticalfracture": "fracture",
    "bakerscyst": "bakers",
    "poplitealcyst": "bakers",
    "jointfluid": "effusion",
    "jointeffusion": "effusion",
    "medialmenisci": "medialmeniscus",
    "lateralmeniscust": "lateralmeniscus",
    "mediaoa": "medialoa",
    "patellofemoraloa": "pfoa",
    "synovits": "synovitis",
    "baker": "bakers",
    "fractur": "fracture",
    "fracutre": "fracture",
}

def format_user_prompt(report_text: str) -> str:
    """Định dạng văn bản báo cáo lâm sàng chuẩn prompt đầu vào cho LLM[cite: 1]."""
    return (
        f"Analyze this knee MRI clinical report (interpret English, Spanish, French, German, Turkish, Greek, Croatian/Serbian).\n"
        f"Apply high clinical sensitivity: classify mild/early abnormalities, cartilage wear, edema, and micro-tears as 1.\n\n"
        f"Report:\n\"\"\"\n{str(report_text).strip()}\n\"\"\"\n\n"
        f"CRITICAL RULES:\n"
        f"1. Output ONLY a valid single-line JSON with integer values (0 or 1), never lists or arrays.\n"
        f"2. Use exact key names as provided below. Do not translate keys, do not use hyphens.\n"
        f"Start output immediately with '{{':\n"
        f"{SCHEMA_TEMPLATE}"
    )