import json
import re
from .prompt_templates import LABEL_COLS, ALIAS_MAP

def clean_key(k: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(k).lower())

CLEAN_ALIAS_MAP = {clean_key(k): clean_key(v) for k, v in ALIAS_MAP.items()}


def parse_json_response(raw_text: str) -> dict:
    """
    Phân tích phản hồi text từ LLM thành dictionary nhãn nhị phân:
    1. Chuẩn hóa dấu phân cách Unicode toàn giác
    2. Thử giải mã JSON trực tiếp (kèm kỹ thuật tự đóng chuỗi khi chạm token limit)
    3. Fallback bóc tách regex theo từng cặp key-value và tra cứu aliases
    """
    default_dict = {col: 0 for col in LABEL_COLS}
    if not raw_text or not isinstance(raw_text, str):
        return default_dict

    # 1. Chuẩn hóa ký tự Unicode Á Đông
    cleaned = (
        raw_text.replace("，", ",")
        .replace("：", ":")
        .replace("\u3000", " ")
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
    )
    cleaned = re.sub(r'\}\s*,', ',', cleaned)

    # 2. Thử Parse JSON thông thường
    parsed = None
    match = re.search(r'(\{[\s\S]*?\})', cleaned)
    if match:
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    if parsed is None:
        start_idx = cleaned.find('{')
        if start_idx != -1:
            fragment = cleaned[start_idx:].strip()
            for suffix in ['"}', '}', ' 0}', ' 1}']:
                try:
                    parsed = json.loads(fragment + suffix)
                    break
                except json.JSONDecodeError:
                    continue

    if isinstance(parsed, dict):
        clean_parsed = {CLEAN_ALIAS_MAP.get(clean_key(k), clean_key(k)): v for k, v in parsed.items()}
        res = {}
        for col in LABEL_COLS:
            target_key = clean_key(col)
            val = clean_parsed.get(target_key, 0)
            if isinstance(val, bool):
                res[col] = 1 if val else 0
            elif isinstance(val, (int, float)):
                res[col] = 1 if val >= 0.5 else 0
            elif isinstance(val, str):
                res[col] = 1 if val.strip().lower() in ["1", "true", "yes", "positive"] else 0
            else:
                res[col] = 0
        return res

    # 3. Fallback Regex Parsing nếu JSON cấu trúc hỏng
    pattern = r'["\']?([a-zA-Z0-9_\-\s\']+)["\']?\s*:\s*(\[?\s*\d*\s*\]?)'
    matches = re.findall(pattern, cleaned)
    extracted = {}
    for raw_k, raw_v in matches:
        norm_k = clean_key(raw_k)
        digit_match = re.search(r'\d+', raw_v)
        val = int(digit_match.group(0)) if digit_match else 0
        extracted[norm_k] = 1 if val >= 1 else 0

    for k_norm, v in extracted.items():
        mapped_target = CLEAN_ALIAS_MAP.get(k_norm, k_norm)
        for target_col in LABEL_COLS:
            norm_target = clean_key(target_col)
            if norm_target == mapped_target or norm_target in k_norm:
                default_dict[target_col] = max(default_dict[target_col], v)

    return default_dict