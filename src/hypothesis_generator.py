"""
hypothesis_generator.py
Call LLM to generate factor hypotheses based on feature classification + behavior definitions + historical experience.
"""
import json, re, logging
from typing import Optional
from src.llm_client import LLMClient
from pathlib import Path

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Prompt templates (v1.3: no feature preference, grouped by category, behavioral rules introduced)
# ------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert in animal behavior research, proficient in NumPy vectorized programming. Task: design "behavioral factors" for mouse behavior identification.

[Data Format]
- windows: np.ndarray [N, seq_length, D], full time window feature matrix. Each feature is normalized to [0,1].
- idx: dict, feature name → column index. Must use idx.get('name', -1) to check existence, return -1 if missing.
- np: numpy module, directly available.

[Code Specification — Batch Mode Only]
Must use batch mode (mode="batch"), processing all N windows at once with fully vectorized numpy operations.
Forbidden: for loops or per-window Python logic. Variables: windows [N,seq_length,D], idx, np.
Result must be assigned to _result (float array of shape [N]).
>=50% of windows must return valid values, otherwise the factor is discarded. No imports, no function definitions.

[Vectorized Approach for Conditional Logic]
If conditional logic is needed, use np.where:
  mask = speed_vals > _P1
  _result = np.where(mask, score * 2.0, score)

If piecewise computation is needed, use boolean indexing:
  _result = np.full(len(windows), np.nan)
  valid = (i_speed >= 0) & (i_tail >= 0)
  if valid.any():
      _result[valid] = _P0 * windows[valid, :, i_speed].mean(axis=1)

[Parameterization Convention — Important!]
Do not hardcode specific numeric constants (e.g., 0.4, 0.6, 0.35). Use _P0, _P1, _P2... as tunable parameters,
assigning reasonable default values at the top of the code. An automatic parameter tuner will find optimal values later.

Parameter types and defaults:
  - Weight type (linear combination coefficient): default 1.0, search range approx [0.05, 5.0]
  - Threshold type (condition threshold): default 0.5, search range approx [0.01, 0.95]
  - Scaling type (amplification/attenuation factor): default 1.0, search range approx [0.1, 5.0]
  - Power type (nonlinear exponent): default 1.0, search range approx [0.5, 3.0]

Example — weight+threshold parameterization (batch vectorized):
  _P0 = 1.0   # weight coefficient for speed
  _P1 = 0.3   # activity threshold
  i_speed = idx.get('speed', -1)
  if i_speed < 0:
      _result = np.full(len(windows), np.nan)
  else:
      speed_vals = windows[:, -1, i_speed]
      base = _P0 * speed_vals
      _result = np.where(speed_vals > _P1, base * 2.0, base)

Example — temporal aggregation (batch vectorized):
  _P0 = 1.0
  i_speed = idx.get('speed', -1)
  i_tail = idx.get('tail_motion', -1)
  if i_speed < 0 or i_tail < 0:
      _result = np.full(len(windows), np.nan)
  else:
      mean_speed = np.mean(windows[:, :, i_speed], axis=1)
      std_tail = np.std(windows[:, :, i_tail], axis=1)
      _result = _P0 * mean_speed * (1.0 - std_tail)

Rule: each independently tunable value uses its own _P{n}, do not share a parameter across multiple locations.
Use your best guess for initial parameter values; the tuner will override them.

[Output Format] Strict JSON array, each element with 5 fields:
"name" (factor_ prefix, snake_case), "description" (English), "target" (behavior class name),
"mode" (fixed as "batch"), "code" (Python code, batch vectorized only)
No extra fields, no text outside JSON.
"""

HYPOTHESIS_PROMPT_TEMPLATE = """[Feature Categories] (All features are normalized to [0,1], used equally, no preference)
{schema}

[Behavior Class Definitions]
{behavior_rules}

[Current Weak Classes] (Prioritize designing factors for these classes)
{weak_classes_hint}

[Window Configuration]
{window_context}

[Historical Experience]
{experience}

[Task] Design {n} new factors. Hard requirements:
1. Must use at least 2 features
2. Explore underused feature combinations, prioritize cross-group combinations (e.g., [Motion]+[Social], [Skeleton]+[Tail])
3. At least {n_weak} factors target weak classes
4. Do not duplicate existing factors
5. Do not repeat previously failed strategies
6. All numeric constants must use _P0, _P1... parameterization, no hardcoded values
7. Must use batch vectorized mode, no for loops or per-window Python logic
{temporal_requirements}
{n_start}. >=50% of windows return valid float
{n_end}. Every feature in code uses idx.get() for existence check

Output JSON array directly:"""

_SINGLE_FRAME_REQUIREMENTS = """\
8. Window has only 1 frame, use current frame features for mathematical combinations (ratios, differences, products, absolute values, nonlinear transformations), all vectorized"""

_TEMPORAL_REQUIREMENTS_TEMPLATE = """\
8. Window has {seq_length} frames, must utilize temporal information (e.g., np.mean/std, np.polyfit trend, first-last difference, np.max-np.min range), all vectorized"""

# ====================================================================
# Feature classification (synced with data_loader.py FEATURE_REGISTRY)
# ====================================================================
FEATURE_GROUPS = {
    "Skeleton(Self)": [
        "nose_to_head", "head_to_body", "body_to_tail",
        "body_orientation", "body_length", "body_compactness",
    ],
    "Skeleton(Other)": [
        "other_nose_to_head", "other_head_to_body", "other_body_to_tail",
        "other_body_orientation", "other_body_length", "other_body_compactness",
    ],
    "Motion(Self)": [
        "head_vel_x", "head_vel_y", "body_vel_x", "body_vel_y",
        "tail_vel_x", "tail_vel_y", "speed", "acceleration",
    ],
    "Motion(Other)": [
        "other_head_vel_x", "other_head_vel_y", "other_body_vel_x", "other_body_vel_y",
        "other_tail_vel_x", "other_tail_vel_y", "other_speed", "other_acceleration",
    ],
    "Tail(Self)": ["tail_angle", "tail_curve", "tail_motion"],
    "Tail(Other)": ["other_tail_angle", "other_tail_curve", "other_tail_motion"],
    "Social": [
        "dist_to_other", "self_facing_other", "other_facing_self",
        "mutual_facing", "approach_speed", "relative_speed",
        "heading_diff", "body_axis_align",
    ],
}


class HypothesisGenerator:
    """Call LLM to generate a list of factor hypotheses."""

    def __init__(self, llm_client: LLMClient, cfg: dict, seq_length: int = 1):
        self.llm = llm_client
        self.n_hypotheses = cfg["llm"].get("hypotheses_per_round", 5)
        self.behavior_classes = cfg.get("data", {}).get("behavior_classes", [])
        self.seq_length = seq_length
        # Load behavior rules
        rules_path = cfg.get("behavior_rules", "config/behavior_rules.json")
        self.behavior_rules = self._load_behavior_rules(rules_path)

    @staticmethod
    def _load_behavior_rules(path: str) -> dict:
        p = Path(path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f).get("rules", {})
        logger.warning(f"[HypothesisGenerator] Behavior rules file does not exist: {path}")
        return {}

    def generate(
        self,
        flat_attributes: list,
        experience_summary: str,
        n: Optional[int] = None,
        weak_classes: Optional[list] = None,
    ) -> list[dict]:
        n = n or self.n_hypotheses
        schema = self._build_schema_description(flat_attributes)
        rules_text = self._build_rules_text()
        behavior_str = "\n".join(f"  - {b}" for b in self.behavior_classes) if self.behavior_classes else ""

        if weak_classes:
            weak_hint = "The following classes currently have no valid factors, please prioritize designing factors for them:\n" + "\n".join(f"  - {c}" for c in weak_classes)
            n_weak = max(1, min(len(weak_classes), n // 2))
        else:
            weak_hint = "(No particularly weak classes currently, explore all behavior classes evenly)"
            n_weak = 0

        seq = self.seq_length
        if seq <= 1:
            window_context = "seq_length=1 (single frame mode, no historical frames)"
            temporal_requirements = _SINGLE_FRAME_REQUIREMENTS
            n_start, n_end = 9, 10
        else:
            window_context = f"seq_length={seq} (multi-frame temporal mode, windows[0] earliest frame, windows[-1] latest frame)"
            temporal_requirements = _TEMPORAL_REQUIREMENTS_TEMPLATE.format(seq_length=seq)
            n_start, n_end = 9, 10

        user_prompt = HYPOTHESIS_PROMPT_TEMPLATE.format(
            schema=schema,
            behavior_rules=rules_text,
            behavior_classes=behavior_str,
            experience=experience_summary,
            n=n,
            weak_classes_hint=weak_hint,
            window_context=window_context,
            temporal_requirements=temporal_requirements,
            n_weak=n_weak,
            n_start=n_start,
            n_end=n_end,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        logger.info(f"[HypothesisGenerator] Requesting LLM to generate {n} hypotheses (seq_length={seq})...")
        raw = self.llm.chat(messages)

        hypotheses = self._parse_response(raw)
        logger.info(f"[HypothesisGenerator] Successfully parsed {len(hypotheses)} hypotheses")
        return hypotheses

    # ------------------------------------------------------------------
    # Schema description (grouped by FEATURE_GROUPS semantics)
    # ------------------------------------------------------------------
    def _build_schema_description(self, flat_attributes: list) -> str:
        if not flat_attributes:
            return "(No feature information)"

        attr_set = set(flat_attributes)
        lines = [f"Total {len(flat_attributes)} dimensions, grouped by semantics as follows:\n"]

        # Output by FEATURE_GROUPS
        for group_name, feats in FEATURE_GROUPS.items():
            present = [f for f in feats if f in attr_set]
            if present:
                lines.append(f"  [{group_name}] {', '.join(present)}")
            else:
                lines.append(f"  [{group_name}] (No available features)")

        # Ungrouped features
        all_grouped: set[str] = set()
        for feats in FEATURE_GROUPS.values():
            all_grouped.update(feats)
        ungrouped = sorted(attr_set - all_grouped)
        if ungrouped:
            lines.append(f"\n  [Other] {', '.join(ungrouped)}")

        lines.append("")
        lines.append("Tip: prioritize cross-group feature combinations (e.g., [Motion(Self)]+[Social], [Skeleton(Self)]+[Tail(Self)]),")
        lines.append("these carry more information than permutations within the same group. Each factor must use >=2 features.")
        lines.append("Use idx.get('feature_name', -1) to check existence.")
        return "\n".join(lines)

    def _build_rules_text(self) -> str:
        if not self.behavior_rules:
            return "(No manually defined rules)"
        lines = []
        for cls, rule in self.behavior_rules.items():
            lines.append(f"  - {cls}: {rule}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing (same as before)
    # ------------------------------------------------------------------
    def _parse_response(self, raw: str) -> list[dict]:
        if not raw or not raw.strip():
            logger.warning("[HypothesisGenerator] LLM returned empty content")
            return []
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [h for h in [self._validate_hypothesis(x) for x in data] if h]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                if isinstance(data, list):
                    return [h for h in [self._validate_hypothesis(x) for x in data] if h]
            except json.JSONDecodeError:
                pass
        rescued = self._rescue_truncated_array(raw)
        if rescued:
            parsed = [h for h in [self._validate_hypothesis(x) for x in rescued] if h]
            if parsed:
                logger.warning(f"[HypothesisGenerator] JSON truncated, rescued {len(parsed)} items")
                return parsed
        logger.warning(f"[HypothesisGenerator] Unable to parse output: {raw[:300]}")
        return []

    @staticmethod
    def _rescue_truncated_array(raw: str) -> list[dict]:
        start = raw.find("[")
        if start < 0: return []
        i = start + 1; depth = 0; obj_start = -1
        in_str = False; escape = False; results = []
        for i in range(start + 1, len(raw)):
            ch = raw[i]
            if in_str:
                if escape: escape = False
                elif ch == "\\": escape = True
                elif ch == '"': in_str = False
            else:
                if ch == '"': in_str = True
                elif ch == "{":
                    if depth == 0: obj_start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and obj_start >= 0:
                        try:
                            obj = json.loads(raw[obj_start:i+1])
                            if isinstance(obj, dict): results.append(obj)
                        except json.JSONDecodeError: pass
                        obj_start = -1
        return results

    def _validate_hypothesis(self, h: dict) -> Optional[dict]:
        if not isinstance(h, dict): return None
        if "name" not in h and "factor_name" in h: h["name"] = h.pop("factor_name")
        if "target" not in h:
            for alt in ("target_behavior","target_class","behavior","behavior_class"):
                if alt in h: h["target"] = h.pop(alt); break
            else: h["target"] = "any"
        if "code" not in h and "formula" in h:
            h["code"] = self._formula_to_code(h.pop("formula"))
        if "code" not in h and "implementation" in h:
            h["code"] = h.pop("implementation")
        required = ["name","description","code"]
        if any(k not in h for k in required):
            return None
        h["name"] = re.sub(r"[^\w]", "_", h["name"]).strip("_")
        h.setdefault("mode", "batch")
        return h

    _KNOWN_FEATURES = [
        "speed", "dist_to_other", "approach_dot", "compactness",
        "sniff_min_dist", "wall_dist_score", "width", "height",
    ]

    def _formula_to_code(self, formula: str) -> str:
        import re as _re
        pattern = _re.compile(
            r'\b((?:Self|Other\d+)_\w+|'
            + '|'.join(_re.escape(f) for f in self._KNOWN_FEATURES)
            + r')\b'
        )
        used = list(dict.fromkeys(pattern.findall(formula)))
        lines = []
        for feat_name in used:
            var = feat_name.replace(".", "_")
            lines.append(f"_{var} = idx.get('{feat_name}', -1); {var}_ok = _{var} >= 0")
        for feat_name in used:
            var = "_" + feat_name.replace(".", "_")
            lines.append(f"if not {var}_ok: return np.nan")
        expr = formula
        for feat_name in sorted(used, key=len, reverse=True):
            var = "_" + feat_name.replace(".", "_")
            expr = _re.sub(r'\b' + _re.escape(feat_name) + r'\b', var, expr)
        lines.append(f"return float({expr})")
        return "\n".join(lines)
