#!/usr/bin/env python3
"""
Local Streaming Token — by Assisted Intel.

Prompt Validation & Evaluation logic (GUI-free, mirrors app.logic). Supports the
model-graded eval workflow: fill a prompt template from spreadsheet rows, then ask
a grader model to score each response against user-defined criteria and reply with
a single JSON object. The JSON is parsed tolerantly (the "standard JSON method"
from the Building with the Claude API course) and aggregated into per-criterion and
overall scores.

Nothing here touches Flask, the store, or provider clients — it takes plain dicts
so it can be unit-tested and reused from the SSE route.
"""

import csv
import io
import json
import re
import uuid
from datetime import datetime

# A field/placeholder is written {ColumnName} in the prompt template.
_FIELD = re.compile(r"\{([^{}]+)\}")

# Default grading criteria offered to a brand-new eval project.
DEFAULT_CRITERIA = [
    {"label": "Accuracy", "guidance": "Is the response factually correct and free of made-up details?",
     "mode": "score", "min": 1, "max": 10},
    {"label": "Instruction following", "guidance": "Does the response do exactly what the prompt asked for?",
     "mode": "score", "min": 1, "max": 10},
]


# --------------------------- project factory ---------------------------

def create_eval_dict(name="New Evaluation", server_url=None, model=None):
    """Create a new eval-project dict with every field the app tracks."""
    now = datetime.utcnow().isoformat()
    return {
        "id": uuid.uuid4().hex[:12],
        "name": name,
        "created": now,
        "updated": now,
        "columns": ["Input", "Response"],
        "rows": [],
        "gen_server_url": server_url or "",
        "gen_model": model or "",
        "prompt_template": "",
        "input_columns": ["Input"],
        "output_column": "Response",
        "grader_server_url": server_url or "",
        "grader_model": model or "",
        "criteria": [dict(c) for c in DEFAULT_CRITERIA],
        "batch_models": [],          # [{server_url, model}]
        "gen_instructions": {},      # {column name: how to generate that cell}
        "gen_num_rows": 10,          # default rows to synthesize
        "last_results": None,
    }


# --------------------------- prompt filling ---------------------------

def fill_prompt(template, row, input_columns=None):
    """Substitute {Column} placeholders with the row's cell values.

    Missing columns become empty strings (never raises). If ``input_columns`` is
    given, only those columns are substituted and any other {placeholder} is left
    verbatim — this keeps unrelated braces in the prompt intact.
    """
    template = template or ""
    row = row or {}
    allowed = set(input_columns) if input_columns else None

    def repl(m):
        key = m.group(1).strip()
        if allowed is not None and key not in allowed:
            return m.group(0)
        val = row.get(key)
        return "" if val is None else str(val)

    return _FIELD.sub(repl, template)


# --------------------------- test-data generation ---------------------------

def gen_target_columns(columns, gen_instructions, output_column):
    """The columns the generator should fill: every column except the output
    column, restricted to those with a non-empty instruction."""
    instr = gen_instructions or {}
    out = (output_column or "").strip()
    result = []
    for col in columns or []:
        if col == out:
            continue
        if (instr.get(col) or "").strip():
            result.append(col)
    return result


def build_gen_prompt(columns, gen_instructions, output_column, row_index=0, seed=""):
    """Build a single user message asking the model to invent ONE test row as a
    JSON object keyed by the target columns, honoring each column's instruction.

    The row_index/seed are surfaced so independent parallel calls produce distinct
    rows instead of repeating the same example.
    """
    targets = gen_target_columns(columns, gen_instructions, output_column)
    instr = gen_instructions or {}
    lines = []
    shape = {}
    for col in targets:
        lines.append(f'- "{col}": {instr[col].strip()}')
        shape[col] = "<value>"
    rubric = "\n".join(lines)
    example = json.dumps(shape, indent=2)
    hint = f"{seed}#{row_index}" if seed else str(row_index)
    content = (
        "You generate synthetic test data for evaluating an AI prompt. Invent ONE "
        "realistic, plausible test case as a single JSON object.\n\n"
        "=== FIELDS TO GENERATE (JSON keys) ===\n"
        f"{rubric}\n\n"
        f"This is test case #{row_index + 1} (variation token: {hint}). Make it "
        "distinct and varied — do not repeat a generic default example. Keep the "
        "fields internally consistent with each other.\n\n"
        "Reply with ONLY a single JSON object, no prose before or after, using "
        "exactly these keys and string values. Shape:\n"
        f"{example}"
    )
    return [{"role": "user", "content": content}]


def parse_gen_row(text, target_columns):
    """Parse a generated JSON object into a {column: cell string} dict, keeping only
    the requested columns (missing keys become empty strings). Reuses the tolerant
    grader JSON parser. Returns {} only when the columns request nothing."""
    obj = parse_grader_json(text)
    row = {}
    for col in target_columns or []:
        val = obj.get(col)
        if val is None:
            row[col] = ""
        elif isinstance(val, str):
            row[col] = val.strip()
        elif isinstance(val, (dict, list)):
            row[col] = json.dumps(val, ensure_ascii=False)
        else:
            row[col] = str(val)
    return row


# --------------------------- file import ---------------------------

def parse_csv(raw):
    """Parse CSV text into {columns, rows}. First row is the header; remaining
    rows become {column: cell} dicts. Blank trailing columns are tolerated."""
    reader = csv.reader(io.StringIO(raw))
    all_rows = [r for r in reader]
    # Drop leading fully-empty lines.
    while all_rows and not any((c or "").strip() for c in all_rows[0]):
        all_rows.pop(0)
    if not all_rows:
        return {"columns": [], "rows": []}
    header = [(c or "").strip() or f"Column {i + 1}" for i, c in enumerate(all_rows[0])]
    # De-duplicate header names so cells map unambiguously.
    seen = {}
    columns = []
    for name in header:
        if name in seen:
            seen[name] += 1
            name = f"{name} ({seen[name]})"
        else:
            seen[name] = 0
        columns.append(name)
    rows = []
    for raw_row in all_rows[1:]:
        if not any((c or "").strip() for c in raw_row):
            continue
        row = {}
        for i, col in enumerate(columns):
            row[col] = raw_row[i].strip() if i < len(raw_row) else ""
        rows.append(row)
    return {"columns": columns, "rows": rows}


def split_text(raw, delimiter="", is_regex=False):
    """Split a .txt file into a list of cell strings.

    Default (empty delimiter) splits on blank lines. A literal delimiter splits on
    that exact substring; ``is_regex`` treats the delimiter as a regular expression.
    Empty/whitespace-only chunks are dropped and each chunk is stripped.
    """
    raw = raw or ""
    if not (delimiter or "").strip() and not is_regex:
        parts = re.split(r"\r?\n\s*\r?\n", raw)      # blank-line separated
    elif is_regex:
        try:
            parts = re.split(delimiter, raw)
        except re.error:
            parts = raw.split(delimiter)
    else:
        parts = raw.split(delimiter)
    return [p.strip() for p in parts if p and p.strip()]


# --------------------------- grader messages ---------------------------

def _criteria_rubric(criteria):
    """Render the criteria list into a human-readable rubric + the exact JSON shape
    the grader must return."""
    lines = []
    shape = {}
    for c in criteria or []:
        label = (c.get("label") or "").strip()
        if not label:
            continue
        guidance = (c.get("guidance") or "").strip()
        if (c.get("mode") or "score") == "score":
            lo, hi = c.get("min", 1), c.get("max", 10)
            lines.append(f'- "{label}" (score {lo}-{hi}): {guidance}')
            shape[label] = {"score": f"<integer {lo}-{hi}>", "reasoning": "<one sentence>"}
        else:
            lines.append(f'- "{label}" (reasoning only): {guidance}')
            shape[label] = {"reasoning": "<your assessment>"}
    return "\n".join(lines), shape


def build_grader_messages(project, filled_prompt, response):
    """Assemble one isolated grader turn: the task the model was given, the response
    to grade, the rubric, and a strict instruction to reply with a single JSON
    object shaped exactly like the criteria."""
    rubric, shape = _criteria_rubric(project.get("criteria"))
    example = json.dumps(shape, indent=2)
    content = (
        "You are grading the output of another AI model. Evaluate the response "
        "strictly and objectively against the criteria below.\n\n"
        "=== TASK GIVEN TO THE MODEL ===\n"
        f"{filled_prompt}\n\n"
        "=== RESPONSE TO GRADE ===\n"
        f"{response}\n\n"
        "=== EVALUATION CRITERIA ===\n"
        f"{rubric}\n\n"
        "Reply with ONLY a single JSON object, no prose before or after, using "
        "exactly these keys. For score criteria return an integer within the stated "
        "range plus a short reasoning; for reasoning-only criteria return your "
        "assessment. Shape:\n"
        f"{example}"
    )
    return [{"role": "user", "content": content}]


# --------------------------- JSON parsing ---------------------------

def parse_grader_json(text):
    """Tolerantly extract the grader's JSON object. Tries json.loads first, then
    strips ```json fences and extracts the first balanced {...} block. Returns {}
    when nothing parses (the row is flagged ungraded)."""
    if not text:
        return {}
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    # Strip code fences.
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        try:
            obj = json.loads(fenced.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Extract the first balanced {...} block.
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return {}


def _coerce_score(val):
    """Pull a number out of a grade value that may be an int, float, or string."""
    if isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        m = re.search(r"-?\d+(?:\.\d+)?", val)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


def normalize_grades(raw_grades, criteria):
    """Turn a parsed grader object into a clean per-criterion dict:
    {label: {score: float|None, reasoning: str, mode, min, max}}. Handles graders
    that return a bare number instead of {score, reasoning}."""
    raw_grades = raw_grades or {}
    out = {}
    for c in criteria or []:
        label = (c.get("label") or "").strip()
        if not label:
            continue
        mode = c.get("mode") or "score"
        entry = {"mode": mode, "min": c.get("min", 1), "max": c.get("max", 10),
                 "score": None, "reasoning": ""}
        val = raw_grades.get(label)
        if isinstance(val, dict):
            if mode == "score":
                entry["score"] = _coerce_score(val.get("score"))
            entry["reasoning"] = str(val.get("reasoning") or "").strip()
        elif val is not None:
            if mode == "score":
                entry["score"] = _coerce_score(val)
            else:
                entry["reasoning"] = str(val).strip()
        out[label] = entry
    return out


# --------------------------- aggregation ---------------------------

def aggregate(row_grades, criteria):
    """Aggregate per-row normalized grades into per-criterion averages and one
    overall 0-100 prompt score.

    row_grades : list of normalize_grades() dicts (one per graded row)
    Returns {overall, per_criterion:{label:{avg, avg_pct, n, mode}}, graded, total}
    """
    per = {}
    for c in criteria or []:
        label = (c.get("label") or "").strip()
        if not label:
            continue
        mode = c.get("mode") or "score"
        lo = float(c.get("min", 1))
        hi = float(c.get("max", 10))
        span = (hi - lo) or 1.0
        raw_scores = []
        pct_scores = []
        for grades in row_grades:
            g = grades.get(label)
            if not g or g.get("score") is None:
                continue
            s = float(g["score"])
            raw_scores.append(s)
            pct_scores.append(max(0.0, min(1.0, (s - lo) / span)) * 100.0)
        avg = round(sum(raw_scores) / len(raw_scores), 2) if raw_scores else None
        avg_pct = round(sum(pct_scores) / len(pct_scores), 1) if pct_scores else None
        per[label] = {"avg": avg, "avg_pct": avg_pct, "n": len(raw_scores),
                      "mode": mode, "min": lo, "max": hi}

    # Overall = mean of every numeric criterion's percentage average.
    pcts = [v["avg_pct"] for v in per.values() if v["mode"] == "score" and v["avg_pct"] is not None]
    overall = round(sum(pcts) / len(pcts), 1) if pcts else None
    return {"overall": overall, "per_criterion": per,
            "graded": len(row_grades)}
