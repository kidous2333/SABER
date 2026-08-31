"""
memory.py
Save valid factors and loop experiences, providing historical context for next-round LLM hypothesis generation.

Persistence structure:
  memory/
    valid_factors.json   — All validated factor definitions and metrics
    experience.json      — Experience summary per round (which directions worked/failed)
    run_log.json         — Complete run log per round
"""

import json
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manage persistent memory for the factor mining process."""

    def __init__(self, cfg: dict):
        memory_dir = Path(cfg["output"]["memory_dir"])
        memory_dir.mkdir(parents=True, exist_ok=True)
        self.factors_path = memory_dir / "valid_factors.json"
        self.experience_path = memory_dir / "experience.json"
        self.run_log_path = memory_dir / "run_log.json"
        self.combo_stats_path = memory_dir / "combo_stats.json"
        self.worker_id = cfg.get("loop", {}).get("worker_id", "")

        self.valid_factors: list[dict] = self._load_json(self.factors_path, default=[])
        self.experiences: list[dict] = self._load_json(self.experience_path, default=[])
        self.run_log: list[dict] = self._load_json(self.run_log_path, default=[])

        # Dedup set: names + normalized codes (including those synced from shared library)
        self._factor_names: set[str] = {f["name"] for f in self.valid_factors}
        self._known_codes: set[str] = {
            "".join(f.get("code", "").split()) for f in self.valid_factors
        }

        # Feature combination tracking (for diversity guidance)
        self._feat_combo_counter: dict[str, int] = self._load_json(
            self.combo_stats_path, default={}
        )
        if not self._feat_combo_counter:
            # Rebuild counter from existing factors
            for f in self.valid_factors:
                combo_key = str(self._extract_features_from_code(f.get("code", "")))
                self._feat_combo_counter[combo_key] = (
                    self._feat_combo_counter.get(combo_key, 0) + 1
                )

        # Names of all 42 features (synced with hypothesis_generator.FEATURE_GROUPS)
        self.ALL_42_FEATURES: set[str] = {
            "nose_to_head", "head_to_body", "body_to_tail",
            "body_orientation", "body_length", "body_compactness",
            "other_nose_to_head", "other_head_to_body", "other_body_to_tail",
            "other_body_orientation", "other_body_length", "other_body_compactness",
            "head_vel_x", "head_vel_y", "body_vel_x", "body_vel_y",
            "tail_vel_x", "tail_vel_y", "speed", "acceleration",
            "other_head_vel_x", "other_head_vel_y", "other_body_vel_x", "other_body_vel_y",
            "other_tail_vel_x", "other_tail_vel_y", "other_speed", "other_acceleration",
            "tail_angle", "tail_curve", "tail_motion",
            "other_tail_angle", "other_tail_curve", "other_tail_motion",
            "dist_to_other", "self_facing_other", "other_facing_self",
            "mutual_facing", "approach_speed", "relative_speed",
            "heading_diff", "body_axis_align",
        }

    # ------------------------------------------------------------------
    # Feature extraction (used by combo tracking / diversity)
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_features_from_code(code: str) -> tuple[str, ...]:
        """Extract feature names used from factor code (two modes: idx.get + names=[])."""
        import re as _re_feat
        feats: set[str] = set()
        # Mode 1: idx.get('feature_name', ...)
        for m in _re_feat.finditer(r"idx\.get\(['\"]([^'\"]+)['\"]", code):
            feat = m.group(1)
            if feat and not feat.startswith("_"):
                feats.add(feat)
        # Mode 2: names=['xxx','yyy']... then idx.get(n,-1)
        names_lists = _re_feat.findall(r"names\s*=\s*\[([^\]]+)\]", code)
        for nl in names_lists:
            for m in _re_feat.finditer(r"['\"]([^'\"]+)['\"]", nl):
                feats.add(m.group(1))
        # Mode 3: features = ['xxx','yyy']... then loop idx.get
        feat_lists = _re_feat.findall(r"features\s*=\s*\[([^\]]+)\]", code)
        for fl in feat_lists:
            for m in _re_feat.finditer(r"['\"]([^'\"]+)['\"]", fl):
                feats.add(m.group(1))
        return tuple(sorted(feats))

    # ------------------------------------------------------------------
    # Save valid factors
    # ------------------------------------------------------------------
    def save_valid_factor(self, hypothesis: dict, validation_result: dict, seq_length: int = 1):
        """Save a factor that passed validation. Perform code dedup check first."""
        name = hypothesis["name"]
        code = hypothesis.get("code", "").strip()

        # 1. Name dedup
        if name in self._factor_names:
            logger.debug(f"[Memory] Factor '{name}' already exists, skipping duplicate save.")
            return

        # 2. Code dedup: check own + shared library known codes
        new_code_norm = "".join(code.split())
        if new_code_norm in self._known_codes:
            logger.info(
                f"[Memory] Factor '{name}' code duplicates a known factor, skipping save."
            )
            self._factor_names.add(name)  # Register name to avoid future duplicate checks
            return

        record = {
            "name": name,
            "description": hypothesis.get("description", ""),
            "target": hypothesis.get("target", ""),
            "code": hypothesis["code"],
            "mode": hypothesis.get("mode", "row"),
            "seq_length": hypothesis.get("seq_length", seq_length),
            "best_auc": validation_result.get("best_auc", validation_result.get("auc", 0)),
            "best_f1": validation_result.get("best_f1", 0),
            "best_class": validation_result.get("best_class", ""),
            "valid_classes": validation_result.get("valid_classes", []),
            "per_class": validation_result.get("per_class", {}),
            "n_samples": validation_result.get("n_val", validation_result.get("n_samples", 0)),
            "saved_at": datetime.now().isoformat(),
        }
        self.valid_factors.append(record)
        self._factor_names.add(name)
        self._known_codes.add("".join(code.split()))

        # 3. Update feature combination counter
        combo_key = str(self._extract_features_from_code(code))
        self._feat_combo_counter[combo_key] = (
            self._feat_combo_counter.get(combo_key, 0) + 1
        )
        # Persist every 10 new combinations
        if len(self._feat_combo_counter) % 10 == 0:
            self._save_json(self.combo_stats_path, self._feat_combo_counter)

        self._save_json(self.factors_path, self.valid_factors)
        valid_cls_names = [v["class"] for v in record["valid_classes"]]
        logger.info(f"[Memory] Saved valid factor: '{name}' (best AUC={record['best_auc']:.4f}, valid classes={valid_cls_names})")

    # ------------------------------------------------------------------
    # Save current round experience
    # ------------------------------------------------------------------
    def save_round_experience(
        self,
        round_num: int,
        hypotheses: list[dict],
        results: list[dict],
    ):
        """
        Save the experience summary for the current round.
        results are validation result dicts in one-to-one correspondence with hypotheses (containing valid/auc/f1/reason).
        """
        valid_items = [
            (h, r) for h, r in zip(hypotheses, results) if r.get("valid", False)
        ]
        invalid_items = [
            (h, r) for h, r in zip(hypotheses, results) if not r.get("valid", False)
        ]

        experience = {
            "round": round_num,
            "worker_id": self.worker_id,
            "timestamp": datetime.now().isoformat(),
            "n_hypotheses": len(hypotheses),
            "n_valid": len(valid_items),
            "valid_factors": [
                {
                    "name": h["name"],
                    "target": h.get("target", ""),
                    "description": h.get("description", ""),
                    "best_auc": r.get("best_auc", r.get("auc", 0)),
                    "best_class": r.get("best_class", ""),
                    "valid_classes": [v["class"] for v in r.get("valid_classes", [])],
                }
                for h, r in valid_items
            ],
            "invalid_factors": [
                {
                    "name": h["name"],
                    "target": h.get("target", ""),
                    "description": h.get("description", ""),
                    "reason": r.get("reason", ""),
                }
                for h, r in invalid_items
            ],
        }
        self.experiences.append(experience)
        # Keep at most 300 experience entries, discard oldest when exceeded (prevent unbounded JSON growth)
        if len(self.experiences) > 300:
            self.experiences = self.experiences[-200:]
            logger.info(f"[Memory] Experience entries exceed limit, trimming to most recent 200 entries")
        self._save_json(self.experience_path, self.experiences)

    # ------------------------------------------------------------------
    # Generate experience summary for LLM
    # ------------------------------------------------------------------
    def get_experience_summary(
        self,
        max_rounds: int = 5,         # Deprecated, kept for backward compatibility
        max_chars: int = 2500,       # Hard upper limit for total summary characters
        llm=None,                    # Deprecated
        keep_recent_rounds: int = 3,
        keep_recent_factors: int = 20,
        earlier_max_chars: int = 800,
    ) -> str:
        """
        Layered strategy:
        - Most recent keep_recent_factors valid factors + most recent keep_recent_rounds exploration rounds: verbatim
        - Earlier parts: aggregate by target; if aggregation still exceeds earlier_max_chars, fall back to truncation
        - If recent itself exceeds the (max_chars - earlier_max_chars) budget, downgrade the oldest recent
          entries to earlier and include them in aggregation, until within budget (no information loss)
        """
        if not self.valid_factors and not self.experiences:
            return "This is the first round of exploration. Boldly try many different directions in factor design."

        # Count usage frequency for each feature → diversity guidance
        import re as _re_stats
        _feat_usage = {}
        for _f in self.valid_factors:
            _code = _f.get("code", "")
            _refs = set(_re_stats.findall(r"idx\.get\(['\"]([^'\"]+)['\"]", _code))
            _refs |= set(_re_stats.findall(r"idx\[['\"]([^'\"]+)['\"]\]", _code))
            for _r in _refs:
                _feat_usage[_r] = _feat_usage.get(_r, 0) + 1

        n_f_total = len(self.valid_factors)
        _diversity_hint = ""
        if n_f_total >= 10 and _feat_usage:
            _sorted_feats = sorted(_feat_usage.items(), key=lambda x: -x[1])
            _overused = [(n, c, 100.0*c/n_f_total) for n, c in _sorted_feats[:3] if c > n_f_total * 0.35]
            _underused = [(n, c, 100.0*c/n_f_total) for n, c in _sorted_feats[-8:] if c <= n_f_total * 0.05]

            _lines = ["\n[DIVERSITY — Feature Usage Report]"]
            if _overused:
                _lines.append("  ★ OVER-USED — try to use these LESS often:")
                for name, count, pct in _overused:
                    _lines.append(f"    {name}: {pct:.0f}% of factors ({count}/{n_f_total})")
                _lines.append("  → Avoid building new factors that rely ONLY on these. Combine with other features.")
            if _underused:
                _lines.append("  ☆ Rarely used (suggestions, NOT requirements — some may simply have limited signal):")
                for name, count, pct in _underused:
                    _lines.append(f"    {name}: {pct:.0f}% of factors ({count}/{n_f_total})")
            _diversity_hint = "\n".join(_lines)

        n_f = len(self.valid_factors)
        # Sort by AUC for top-N, not by time for most recent N
        sorted_factors = sorted(
            self.valid_factors,
            key=lambda f: float(f.get("best_auc", f.get("auc", 0)) or 0),
            reverse=True,
        )
        recent_factors = sorted_factors[:keep_recent_factors]
        earlier_factors = sorted_factors[keep_recent_factors:] if n_f > keep_recent_factors else []

        n_e = len(self.experiences)
        recent_exps = list(self.experiences[-keep_recent_rounds:]) if n_e else []
        earlier_exps = list(self.experiences[:-keep_recent_rounds]) if n_e > keep_recent_rounds else []

        # recent budget: reserve earlier_max_chars for earlier + 200 chars for separator/hint overhead
        recent_budget = max(500, max_chars - earlier_max_chars - 200)

        # Greedy downgrade: when recent exceeds budget, move oldest entries to earlier
        recent_block = self._render_recent_full(recent_factors, recent_exps)
        n_downgraded_exps = 0
        n_downgraded_factors = 0
        while len(recent_block) > recent_budget:
            if len(recent_exps) > 1:
                earlier_exps.append(recent_exps.pop(0))
                n_downgraded_exps += 1
            elif len(recent_factors) > 5:
                earlier_factors.append(recent_factors.pop(0))
                n_downgraded_factors += 1
            else:
                break
            recent_block = self._render_recent_full(recent_factors, recent_exps)

        if n_downgraded_exps or n_downgraded_factors:
            logger.info(
                f"[Memory] recent exceeds budget (budget={recent_budget}), "
                f"downgraded {n_downgraded_exps} rounds / {n_downgraded_factors} factors to earlier aggregation"
            )

        earlier_block = self._render_earlier_summary(earlier_factors, earlier_exps)
        if earlier_block and len(earlier_block) > earlier_max_chars:
            omitted = len(earlier_block) - earlier_max_chars
            logger.info(
                f"[Memory] Early aggregated summary ({len(earlier_block)} chars) exceeds {earlier_max_chars}, truncating by chars"
                f" (early factors: {len(earlier_factors)} / early rounds: {len(earlier_exps)})"
            )
            earlier_block = (
                earlier_block[:earlier_max_chars]
                + f"\n  ...(early aggregation truncated to {earlier_max_chars} chars, {omitted} chars omitted)"
            )

        summary = (earlier_block + "\n\n" + recent_block + _diversity_hint).strip() if earlier_block else (recent_block + _diversity_hint)

        # Append feature combination diversity hint
        diversity_hint = self._get_feature_combo_summary()
        if diversity_hint:
            summary = summary + "\n" + diversity_hint

        if len(summary) > max_chars:
            logger.warning(
                f"[Memory] Experience summary ({len(summary)} chars) still exceeds max_chars={max_chars} (approximate limit);"
                f" even after downgrading {n_downgraded_exps}/{n_downgraded_factors} entries, recent section is still too long,"
                f" consider decreasing keep_recent_* or earlier_max_chars"
            )
        return summary

    # ------------------------------------------------------------------
    # Render: recent verbatim block (no compression)
    # ------------------------------------------------------------------
    def _render_recent_full(self, recent_factors: list, recent_exps: list) -> str:
        lines = []
        if recent_factors:
            lines.append(f"Valid factors (recent top-{len(recent_factors)}):")
            for f in recent_factors:
                valid_cls = [v["class"] for v in f.get("valid_classes", [])]
                lines.append(
                    f"  {f['name']}|{valid_cls}|AUC={f.get('best_auc', 0):.2f}"
                )
        else:
            lines.append("Valid factors: none yet")

        if recent_exps:
            lines.append("Recent exploration:")
            for exp in recent_exps:
                valid_names = [f['name'] for f in exp.get("valid_factors", [])]
                invalid_names = [f['name'] for f in exp.get("invalid_factors", [])]
                lines.append(
                    f"  Round {exp['round']}: ✓{valid_names} ✗{invalid_names}"
                )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Render: earlier section (rule-based compression, aggregated by target)
    # ------------------------------------------------------------------
    def _render_earlier_summary(self, earlier_factors: list, earlier_exps: list) -> str:
        if not earlier_factors and not earlier_exps:
            return ""

        lines = ["Early experience (aggregated):"]

        if earlier_factors:
            by_target: dict[str, list] = {}
            for f in earlier_factors:
                cls_list = [v["class"] for v in f.get("valid_classes", [])] or ["any"]
                for cls in cls_list:
                    by_target.setdefault(cls, []).append(
                        (f["name"], float(f.get("best_auc", f.get("auc", 0)) or 0))
                    )
            for cls, items in sorted(by_target.items(), key=lambda x: -len(x[1])):
                items.sort(key=lambda x: -x[1])
                top2 = ", ".join(f"{n}" for n, _ in items[:2])
                lines.append(f"  {cls}×{len(items)}: {top2}")

        if earlier_exps:
            total_tried = sum(e.get("n_hypotheses", 0) for e in earlier_exps)
            total_valid = sum(e.get("n_valid", 0) for e in earlier_exps)
            lines.append(f"  Early {len(earlier_exps)} rounds: tried {total_tried}, {total_valid} valid")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Feature combination diversity guidance
    # ------------------------------------------------------------------
    def _get_feature_combo_summary(self, top_n_overused: int = 3) -> str:
        """
        Generate a compact diversity hint:
        1. Overused feature combinations (>=3 factors sharing) → suggest avoiding
        2. Unused or rarely used features → suggest trying
        """
        if not self._feat_combo_counter:
            return ""

        lines = ["[DIVERSITY]"]
        import ast as _ast

        # 1. Overused combinations
        overused: list[tuple[str, int]] = [
            (k, v) for k, v in self._feat_combo_counter.items() if v >= 3
        ]
        overused.sort(key=lambda x: -x[1])
        if overused[:top_n_overused]:
            parts = []
            for combo_str, cnt in overused[:top_n_overused]:
                try:
                    feats = _ast.literal_eval(combo_str)
                except Exception:
                    feats = combo_str.strip("()").split(",")
                feat_short = ", ".join(
                    list(feats)[:5] if isinstance(feats, (tuple, list)) else [str(feats)]
                )
                if isinstance(feats, (tuple, list)) and len(feats) > 5:
                    feat_short += f", ...(+{len(feats)-5})"
                parts.append(f"{{{feat_short}}}({cnt}×)")
            lines.append(
                f"Overused feature combinations (avoid): {' | '.join(parts)}"
            )

        # 2. All features that have been used
        all_used: set[str] = set()
        for combo_str in self._feat_combo_counter:
            try:
                feats = _ast.literal_eval(combo_str)
                if isinstance(feats, (tuple, list)):
                    all_used.update(feats)
            except Exception:
                pass

        unused = sorted(self.ALL_42_FEATURES - all_used)
        rarely_used = sorted(
            f for f in (self.ALL_42_FEATURES & all_used)
            if sum(1 for c in self._feat_combo_counter if f in c) <= 2
        )

        if unused:
            lines.append(f"Unused features: {', '.join(unused[:6])}")
        if rarely_used:
            lines.append(f"Rare features (worth exploring): {', '.join(rarely_used[:6])}")

        lines.append("Suggestion: prioritize combining features from different [semantic groups] (e.g., motion+social, skeleton+tail)")
        return "\n".join(lines)

    def get_diversity_direction(self, focus: str) -> str:
        """Return exploration direction suggestion based on diversity_focus."""
        FOCUS_HINTS = {
            "social": "Prioritize combining [Social] features (dist_to_other/self_facing_other/approach_speed) with [Motion(Self)] features",
            "skeleton_self": "Prioritize combining [Skeleton(Self)] features (body_length/body_orientation/nose_to_head) with [Motion] or [Tail] features",
            "skeleton_other": "Prioritize using [Skeleton(Other)] features (other_body_orientation/other_body_length) paired with [Social] features",
            "motion_self": "Prioritize combining contrasts/differences of multi-part velocities (head/body/tail vel) from [Motion(Self)]",
            "motion_other": "Prioritize combining [Motion(Other)] features with [Social] features",
            "tail_self": "Prioritize combining [Tail(Self)] features (tail_angle/tail_curve/tail_motion) with [Skeleton] or [Motion] features",
            "tail_other": "Prioritize trying [Tail(Other)] features paired with [Social] features",
            "cross_group": "Mandatorily use >=1 [Social] + >=1 [Motion] + >=1 [Skeleton] features",
        }
        hint = FOCUS_HINTS.get(focus, "")
        if hint:
            unused = sorted(self.ALL_42_FEATURES - {
                f for c in self._feat_combo_counter
                for f in (c.strip("()'").split("', '") if "," in c else [c.strip("()'")])
            })
            if unused:
                hint += f"\n  Unused features: {', '.join(unused[:5])}"
        return f"[DIRECTION:{focus}] {hint}" if hint else ""

    # ------------------------------------------------------------------
    # Run log
    # ------------------------------------------------------------------
    def log_run(self, record: dict):
        """Record metadata for one complete run."""
        self.run_log.append(record)
        self._save_json(self.run_log_path, self.run_log)

    # ------------------------------------------------------------------
    # Cross-process sync: load newly discovered factors from shared library
    # ------------------------------------------------------------------
    def reload_from_shared(self, shared_factors_path: str = None, shared_experience_path: str = None) -> int:
        """
        Sync discoveries from other workers via shared files. Only updates dedup sets, does not modify valid_factors.

        Important: does NOT add shared factors to self.valid_factors, otherwise save_valid_factor()
        would write the entire shared library back to the private file, causing count_factors_worker to be distorted.
        Deduplication still works — self._factor_names and code comparison prevent duplicate storage.
        """
        added = 0

        # ── Sync factors (dedup sets only) ──
        factors_src = Path(shared_factors_path) if shared_factors_path else None
        if factors_src and factors_src.exists():
            shared_factors = self._load_json(factors_src, default=[])
            for f in shared_factors:
                name = f.get("name", "")
                code_norm = "".join(f.get("code", "").split())
                # Name or code already exists → skip
                if name in self._factor_names or code_norm in self._known_codes:
                    continue
                # Only register dedup info, don't add to valid_factors
                self._factor_names.add(name)
                self._known_codes.add(code_norm)

                # Sync feature combination counter
                combo_key = str(self._extract_features_from_code(f.get("code", "")))
                self._feat_combo_counter[combo_key] = (
                    self._feat_combo_counter.get(combo_key, 0) + 1
                )
                added += 1

        if added:
            logger.info(
                f"[Memory] reload: synchronized {added} new factors from shared library "
                f"(own={len(self.valid_factors)}, known={len(self._factor_names)})"
            )

        # ── Sync experiences ──
        exp_src = Path(shared_experience_path) if shared_experience_path else None
        if exp_src and exp_src.exists():
            shared_exp = self._load_json(exp_src, default=[])
            # In multi-worker scenarios round numbers may overlap, dedup by (round, worker_id, n_hypotheses)
            existing_sigs = {
                (e.get("round"), e.get("worker_id", ""), e.get("n_hypotheses", 0))
                for e in self.experiences
            }
            exp_added = 0
            for e in shared_exp:
                sig = (e.get("round"), e.get("worker_id", ""), e.get("n_hypotheses", 0))
                if sig not in existing_sigs:
                    self.experiences.append(e)
                    existing_sigs.add(sig)
                    exp_added += 1
            if exp_added:
                logger.info(
                    f"[Memory] reload: synced {exp_added} rounds of experience from shared library"
                    f" (total {len(self.experiences)} rounds)"
                )

        return added

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def get_valid_factor_names(self) -> set[str]:
        return set(self._factor_names)

    def get_valid_factors(self) -> list[dict]:
        return list(self.valid_factors)

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------
    @staticmethod
    def _load_json(path: Path, default):
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"[Memory] Failed to read {path}: {e}, using default.")
        return default

    @staticmethod
    def _save_json(path: Path, data):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
