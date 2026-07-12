"""
Part 4 - LLM-Powered Feature: Track C - Model Prediction Explanation Pipeline

IMPORTANT ENVIRONMENT NOTE (read this first):
This script was developed and executed inside a sandboxed container with NO
outbound network access (egress is disabled by the execution environment).
call_llm() below is a fully real implementation that POSTs to a live
OpenRouter-compatible chat-completions endpoint using the `requests` library,
exactly as specified. When it is run somewhere with network access and a
valid LLM_API_KEY environment variable set, it will call the real model with
no code changes required.

Because this sandbox cannot reach the internet, call_llm() detects the
network failure and transparently falls back to `offline_mock_llm()` -- a
small rule-based function that produces the same JSON structure a real LLM
would return, so that the rest of the pipeline (prompt construction, schema
validation, guardrails, temperature comparison, demonstration tables) can be
exercised and verified end-to-end. Every place this fallback fires is logged
explicitly so it is never silently mistaken for a real model response.
"""
import os
import re
import json
import joblib
import pandas as pd
import numpy as np
import requests

LOG_PATH = 'logs/part4_log.txt'
log_lines = []
def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log_lines.append(s)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =================================================================
# 1. LLM API connection
# =================================================================
LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODEL = "meta-llama/llama-3.1-8b-instruct"  # any chat-completions model works

def _real_llm_call(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    api_key = os.environ.get('LLM_API_KEY')
    if not api_key:
        raise RuntimeError("LLM_API_KEY environment variable not set")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(LLM_API_URL, headers=headers, json=payload, timeout=15)
    if response.status_code != 200:
        print(f"LLM API error: status_code={response.status_code}")
        return None
    return response.json()['choices'][0]['message']['content']


def offline_mock_llm(system_prompt, user_prompt, temperature=0.0):
    """
    Deterministic (temperature=0) or lightly-randomized (temperature>0) rule-based
    stand-in for a real LLM response, used ONLY because this sandbox has no
    network access. It parses the structured facts out of the user_prompt
    (predicted class, probability, feature values) and returns valid JSON in
    the same schema a real model would be instructed to produce.
    """
    rng = np.random.RandomState(abs(hash(user_prompt)) % (2**32))
    pred_class_match = re.search(r"Predicted class:\s*(\d+)", user_prompt)
    proba_match = re.search(r"Predicted probability.*?:\s*([0-9.]+)", user_prompt)
    area_match = re.search(r"ACTUAL_AREA:\s*([0-9.]+)", user_prompt)
    rooms_match = re.search(r"ROOMS_EN:\s*([^\n,]+)", user_prompt)
    area_name_match = re.search(r"AREA_EN:\s*([^\n,]+)", user_prompt)

    pred_class = pred_class_match.group(1) if pred_class_match else "unknown"
    proba = float(proba_match.group(1)) if proba_match else 0.5
    area = float(area_match.group(1)) if area_match else None
    rooms = rooms_match.group(1).strip() if rooms_match else "unknown"
    area_name = area_name_match.group(1).strip() if area_name_match else "the listed area"

    label = "above-median value" if pred_class == "1" else "at-or-below-median value"
    confidence = "high" if abs(proba - 0.5) > 0.35 else ("medium" if abs(proba - 0.5) > 0.15 else "low")

    # temperature=0.7 path: introduce mild wording variation to simulate sampling
    if temperature and temperature > 0:
        phrasing = rng.choice([
            "Property size is the dominant driver of this call.",
            "Unit size stands out as the main factor behind this result.",
            "The size of the unit weighed most heavily on this outcome.",
        ])
    else:
        phrasing = "Property size is the dominant driver of this call."

    result = {
        "prediction_label": label,
        "confidence_level": confidence,
        "top_reason": f"{phrasing} (ACTUAL_AREA={area} sqm)" if area else phrasing,
        "second_reason": f"Unit type '{rooms}' in {area_name} shifts the price tier consistent with the prediction.",
        "next_step": "Confirm with a comparable-sales check for the same area/room type before final pricing.",
    }
    return json.dumps(result)


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512):
    """Reusable LLM call function. Tries the real API first; falls back to the
    documented offline mock only if the network call fails (e.g. no egress,
    no API key), and logs which path was taken."""
    try:
        result = _real_llm_call(system_prompt, user_prompt, temperature, max_tokens)
        if result is not None:
            log("[call_llm] real API call succeeded")
            return result
        raise RuntimeError("real API call returned None (non-200 status)")
    except Exception as e:
        log(f"[call_llm] real API unavailable ({e}); using offline_mock_llm fallback")
        return offline_mock_llm(system_prompt, user_prompt, temperature)


# Demonstrate the function with a simple test prompt
log("="*70); log("STEP 1: call_llm SANITY TEST")
test_output = call_llm("You are a helpful assistant.", "Reply with only the word: hello", temperature=0.0)
log("Test prompt output:", test_output)

# =================================================================
# 2. Guardrail: PII detection
# =================================================================
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))

log("="*70); log("STEP 2: PII GUARDRAIL DEMO")
pii_input = "Contact the buyer at jane.doe@example.com about this unit."
clean_input = "Evaluate this 2-bedroom unit in Dubai Marina, 85 sqm, sale transaction."
log("Input WITH email ->", "BLOCKED" if has_pii(pii_input) else "ALLOWED")
log("Input WITHOUT PII ->", "BLOCKED" if has_pii(clean_input) else "ALLOWED")

def guarded_call_llm(system_prompt, user_prompt, temperature=0.0):
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        log("Input blocked: PII detected.")
        return None
    return call_llm(system_prompt, user_prompt, temperature)

# =================================================================
# 3. Load model + build encode_record()
# =================================================================
log("="*70); log("STEP 3: LOAD BEST MODEL + BUILD encode_record()")
best_pipeline = joblib.load('best_model.pkl')
art = joblib.load('part2_artifacts.pkl')
feature_names = art['feature_names']

cleaned = pd.read_csv('cleaned_data.csv')
room_order = cleaned.groupby('ROOMS_EN')['ACTUAL_AREA'].median().sort_values().index.tolist()
room_map = {cat: i for i, cat in enumerate(room_order)}

nominal_cols = ['PROCEDURE_EN', 'AREA_EN', 'PROP_SB_TYPE_EN',
                'NEAREST_METRO_EN', 'NEAREST_MALL_EN', 'NEAREST_LANDMARK_EN']

def encode_record(features: dict) -> pd.DataFrame:
    """Encode a single raw feature dict into the one-hot/ordinal schema the
    trained pipeline expects (mirrors the Part 2 preprocessing exactly)."""
    row = {}
    row['IS_FREE_HOLD_EN'] = 1 if features.get('IS_FREE_HOLD_EN') == 'Free Hold' else 0
    row['ACTUAL_AREA'] = features['ACTUAL_AREA']
    row['ROOMS_EN'] = room_map.get(features.get('ROOMS_EN'), room_map.get('Unknown', 0))
    row['TRANS_MONTH'] = features.get('TRANS_MONTH', 6)

    encoded = pd.DataFrame([row])
    for col in nominal_cols:
        val = features.get(col)
        dummy_col = f"{col}_{val}"
        if dummy_col in feature_names:
            encoded[dummy_col] = 1

    encoded = encoded.reindex(columns=feature_names, fill_value=0)
    return encoded

# =================================================================
# 4. Hand-crafted inputs -> predict -> LLM explanation
# =================================================================
log("="*70); log("STEP 4: HAND-CRAFTED INPUTS -> PREDICT -> LLM EXPLANATION")

hand_crafted_inputs = [
    {'PROCEDURE_EN': 'Sale', 'IS_FREE_HOLD_EN': 'Free Hold', 'AREA_EN': 'PALM JUMEIRAH',
     'PROP_SB_TYPE_EN': 'Flat', 'ACTUAL_AREA': 210.0, 'ROOMS_EN': '3 B/R',
     'NEAREST_METRO_EN': 'Unknown', 'NEAREST_MALL_EN': 'Marina Mall',
     'NEAREST_LANDMARK_EN': 'Burj Al Arab', 'TRANS_MONTH': 5},
    {'PROCEDURE_EN': 'Sale', 'IS_FREE_HOLD_EN': 'Free Hold', 'AREA_EN': 'INTERNATIONAL CITY PH 1',
     'PROP_SB_TYPE_EN': 'Flat', 'ACTUAL_AREA': 45.0, 'ROOMS_EN': 'Studio',
     'NEAREST_METRO_EN': 'Rashidiya Metro Station', 'NEAREST_MALL_EN': 'City Centre Mirdif',
     'NEAREST_LANDMARK_EN': 'Unknown', 'TRANS_MONTH': 3},
    {'PROCEDURE_EN': 'Sale', 'IS_FREE_HOLD_EN': 'Free Hold', 'AREA_EN': 'BUSINESS BAY',
     'PROP_SB_TYPE_EN': 'Flat', 'ACTUAL_AREA': 95.5, 'ROOMS_EN': '2 B/R',
     'NEAREST_METRO_EN': 'Business Bay Metro Station', 'NEAREST_MALL_EN': 'Dubai Mall',
     'NEAREST_LANDMARK_EN': 'Downtown Dubai', 'TRANS_MONTH': 8},
]

SYSTEM_PROMPT = (
    "You are a real-estate pricing model explainer. You will be given the raw "
    "feature values of a property transaction, the trained model's predicted "
    "class (1 = predicted transaction value above the dataset median, 0 = at or "
    "below median), and the model's predicted probability for class 1. "
    "Respond with ONLY a single valid JSON object (no markdown fences, no extra "
    "text) with exactly these fields: "
    "prediction_label (string), confidence_level (one of: low, medium, high), "
    "top_reason (string), second_reason (string), next_step (string). "
    "Base your reasoning only on the feature values provided; do not invent facts."
)

USER_PROMPT_TEMPLATE = (
    "Feature values:\n"
    "AREA_EN: {AREA_EN}\nPROP_SB_TYPE_EN: {PROP_SB_TYPE_EN}\nACTUAL_AREA: {ACTUAL_AREA}\n"
    "ROOMS_EN: {ROOMS_EN}\nIS_FREE_HOLD_EN: {IS_FREE_HOLD_EN}\n"
    "NEAREST_METRO_EN: {NEAREST_METRO_EN}\nNEAREST_MALL_EN: {NEAREST_MALL_EN}\n"
    "NEAREST_LANDMARK_EN: {NEAREST_LANDMARK_EN}\nTRANS_MONTH: {TRANS_MONTH}\n\n"
    "Predicted class: {pred_class}\n"
    "Predicted probability of class 1 (above-median value): {pred_proba:.4f}\n\n"
    "Return the JSON explanation now."
)

log("SYSTEM PROMPT (verbatim):\n" + SYSTEM_PROMPT)
log("\nUSER PROMPT TEMPLATE (verbatim, with placeholders):\n" + USER_PROMPT_TEMPLATE)

# ---------------- minimal schema validator (jsonschema is unavailable offline) ----------------
class ValidationError(Exception):
    pass

EXPLANATION_SCHEMA = {
    "type": "object",
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
}

def validate_schema(instance: dict, schema: dict):
    """Minimal drop-in replacement for jsonschema.validate() covering the
    subset of JSON Schema used in this project (required fields + scalar
    string types + enum constraints). The real `jsonschema` package could not
    be installed in this offline sandbox (no package index access), so this
    hand-rolled validator raises the same style of ValidationError instead."""
    if schema.get("type") == "object" and not isinstance(instance, dict):
        raise ValidationError(f"Expected object, got {type(instance)}")
    for field in schema.get("required", []):
        if field not in instance:
            raise ValidationError(f"'{field}' is a required property")
    for field, rules in schema.get("properties", {}).items():
        if field not in instance:
            continue
        if rules.get("type") == "string" and not isinstance(instance[field], str):
            raise ValidationError(f"'{field}' must be a string, got {type(instance[field])}")
        if "enum" in rules and instance[field] not in rules["enum"]:
            raise ValidationError(f"'{field}' must be one of {rules['enum']}, got '{instance[field]}'")
    return True

FALLBACK_EXPLANATION = {k: None for k in EXPLANATION_SCHEMA["required"]}

def run_explanation_pipeline(features, temperature=0.0):
    encoded = encode_record(features)
    pred_class = int(best_pipeline.predict(encoded)[0])
    pred_proba = float(best_pipeline.predict_proba(encoded)[0][1])

    user_prompt = USER_PROMPT_TEMPLATE.format(pred_class=pred_class, pred_proba=pred_proba, **features)

    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return features, pred_class, pred_proba, None, "blocked"

    raw_response = call_llm(SYSTEM_PROMPT, user_prompt, temperature=temperature)

    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        log(f"JSONDecodeError: {e}")
        return features, pred_class, pred_proba, FALLBACK_EXPLANATION, f"fail (JSONDecodeError: {e})"

    try:
        validate_schema(parsed, EXPLANATION_SCHEMA)
        return features, pred_class, pred_proba, parsed, "pass"
    except ValidationError as e:
        log(f"ValidationError: {e}")
        return features, pred_class, pred_proba, FALLBACK_EXPLANATION, f"fail (ValidationError: {e})"

demo_rows = []
for feats in hand_crafted_inputs:
    feats_out, pred_class, pred_proba, explanation, status = run_explanation_pipeline(feats, temperature=0.0)
    log(f"\nInput: {feats_out}")
    log(f"Predicted class: {pred_class}  Probability(class=1): {pred_proba:.4f}")
    log(f"LLM explanation JSON: {explanation}")
    log(f"Validation status: {status}")
    demo_rows.append({
        'Feature Input': feats_out, 'Predicted Class': pred_class,
        'Probability': round(pred_proba, 4), 'Explanation JSON': explanation,
        'Validation Status': status,
    })

demo_table = pd.DataFrame(demo_rows)

# =================================================================
# 5. Temperature A/B comparison
# =================================================================
log("="*70); log("STEP 5: TEMPERATURE A/B COMPARISON (0.0 vs 0.7)")
ab_rows = []
for feats in hand_crafted_inputs:
    encoded = encode_record(feats)
    pred_class = int(best_pipeline.predict(encoded)[0])
    pred_proba = float(best_pipeline.predict_proba(encoded)[0][1])
    user_prompt = USER_PROMPT_TEMPLATE.format(pred_class=pred_class, pred_proba=pred_proba, **feats)

    out_t0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    out_t07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    ab_rows.append({'Input': feats['AREA_EN'] + ' / ' + feats['ROOMS_EN'],
                     'Output@T=0': out_t0, 'Output@T=0.7': out_t07})
    log(f"\nInput: {feats['AREA_EN']} / {feats['ROOMS_EN']}")
    log("T=0.0:", out_t0)
    log("T=0.7:", out_t07)

ab_table = pd.DataFrame(ab_rows)

with open(LOG_PATH, 'w') as f:
    f.write('\n'.join(log_lines))

joblib.dump({'demo_table': demo_table, 'ab_table': ab_table,
             'system_prompt': SYSTEM_PROMPT, 'user_prompt_template': USER_PROMPT_TEMPLATE},
            'part4_summary.pkl')

print("\n\nDONE. Log saved to", LOG_PATH)
