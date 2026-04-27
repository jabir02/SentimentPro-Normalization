from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

try:
    from bnunicodenormalizer import Normalizer as _BNUnicodeNormalizer
except Exception:
    _BNUnicodeNormalizer = None


NORM_VERSION = "bangla-ecom-hybrid-v1.0"

BANGLA_CHAR_RE = re.compile(r"[\u0980-\u09FF]")
LATIN_CHAR_RE = re.compile(r"[A-Za-z]")
BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002700-\U000027BF]",
    flags=re.UNICODE,
)

# Tokenizer keeps placeholders, URLs, SKU-like tokens, words, numbers, and punctuation.
TOKEN_RE = re.compile(
    r"URLTOKEN|EMAILTOKEN|PHONETOKEN|__KEEP_\d+__|"
    r"https?://\S+|www\.\S+|"
    r"[\w\.-]+@[\w\.-]+\.\w+|"
    r"\+?\d[\d\s().-]{6,}\d|"
    r"\d+\s*/\s*\d+(?:\s*(?:gb|tb))?|"
    r"\d+(?:\.\d+)?\s*(?:mah|w|gb|tb|mp|hz|mm|cm|inch|in|tk|৳)|"
    r"[A-Za-z]*\d+[A-Za-z\d-]*|"
    r"[A-Za-z]+(?:[-'][A-Za-z]+)*|"
    r"[\u0980-\u09FF]+(?:[-'][\u0980-\u09FF]+)*|"
    r"\d+(?:[./:]\d+)*|"
    r"[^\w\s]",
    flags=re.UNICODE | re.IGNORECASE,
)

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
ROMAN_WORD_RE = re.compile(r"^[a-z][a-z'-]*$", flags=re.IGNORECASE)

NEGATION_FORMS = {
    "না", "নাই", "নেই", "নয়", "না।",
    "na", "nai", "nei", "no", "not", "never", "without"
}

INTENSIFIER_FORMS = {
    "খুব", "অনেক", "বেশি", "অতি",
    "khub", "onek", "beshi", "very", "too", "super", "really"
}

ROMAN_BANGLISH_CUES = (
    "lam", "chilam", "chi", "si", "se", "bo", "ben", "echi", "ache", "ase",
    "ta", "tai", "gula", "gulo", "valo", "bhalo", "kharap", "nai", "na",
)


@dataclass
class RuleAction:
    rule_id: str
    source: str
    target: str
    method: str
    confidence: float


class ResearchBanglaEcomNormalizer:
    """
    Research-grade normalizer for Bangla e-commerce sentiment reviews.

    Main idea:
    - preserve semantic invariants
    - protect product/domain spans
    - normalize Banglish only when confidence is high
    - log every transformation for auditability
    """

    def __init__(
        self,
        resource_dir: str = "resources",
        fuzzy_cutoff: float = 0.91,
        punctuation_cap: int = 3,
        digit_policy: str = "western_numeric_spans",
    ) -> None:
        self.resource_dir = Path(resource_dir)
        self.fuzzy_cutoff = fuzzy_cutoff
        self.punctuation_cap = punctuation_cap
        self.digit_policy = digit_policy
        self.bn_unicode = _BNUnicodeNormalizer() if _BNUnicodeNormalizer else None

        self.banglish_map = self._load_json_map("banglish_map.json")
        self.dialect_map = self._load_json_map("dialect_map.json")
        self.phrase_map = self._load_json_map("phrase_map.json")
        self.keep_words = self._load_word_list("keep_words.txt")
        self.english_words = self._load_word_list("english_words.txt")

        self.lookup_map: Dict[str, str] = {}
        self.lookup_map.update(self.banglish_map)
        self.lookup_map.update(self.dialect_map)

        self.lookup_keys = list(self.lookup_map.keys())
        self.sorted_phrase_keys = sorted(self.phrase_map.keys(), key=len, reverse=True)

        self.keep_patterns = [
            re.compile(r"^URLTOKEN$|^EMAILTOKEN$|^PHONETOKEN$", re.IGNORECASE),
            re.compile(r"^__KEEP_\d+__$"),
            re.compile(r"^(https?://|www\.)", re.IGNORECASE),
            re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$"),
            re.compile(r"^\+?\d[\d\s().-]{6,}\d$"),
            re.compile(r"^\d+\s*/\s*\d+(?:\s*(?:gb|tb))?$", re.IGNORECASE),
            re.compile(r"^\d+(?:\.\d+)?\s*(?:mah|w|gb|tb|mp|hz|mm|cm|inch|in|tk|৳)$", re.IGNORECASE),
            re.compile(r"^[a-z]*\d+[a-z\d-]*$", re.IGNORECASE),
        ]

    def _load_json_map(self, filename: str) -> Dict[str, str]:
        p = self.resource_dir / filename
        if not p.exists():
            return {}
        data = json.loads(p.read_text(encoding="utf-8"))
        return {str(k).strip().lower(): str(v).strip() for k, v in data.items() if str(k).strip() and str(v).strip()}

    def _load_word_list(self, filename: str) -> set:
        p = self.resource_dir / filename
        if not p.exists():
            return set()
        return {line.strip().lower() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()}

    @staticmethod
    def is_bangla_token(token: str) -> bool:
        return bool(BANGLA_CHAR_RE.search(token))

    @staticmethod
    def is_punct(token: str) -> bool:
        return bool(PUNCT_RE.fullmatch(token))

    @staticmethod
    def md5_text(text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def script_stats(self, text: str) -> Dict[str, float]:
        bn = len(BANGLA_CHAR_RE.findall(text))
        lat = len(LATIN_CHAR_RE.findall(text))
        total = max(bn + lat, 1)
        return {
            "bn_chars": bn,
            "latin_chars": lat,
            "script_ratio_bn": round(bn / total, 4),
            "script_ratio_latin": round(lat / total, 4),
        }

    def detect_language_type(self, text: str) -> str:
        stats = self.script_stats(text)
        bn, lat = stats["bn_chars"], stats["latin_chars"]

        if bn > 0 and lat == 0:
            return "bangla"
        if bn > 0 and lat > 0:
            return "mixed"

        tokens = [t.lower() for t in self.tokenize(text) if not self.is_punct(t)]
        banglish_signals = sum(1 for t in tokens if self.looks_like_banglish(t))
        english_hits = sum(1 for t in tokens if t in self.english_words or t in self.keep_words)

        if banglish_signals >= 2:
            return "banglish"
        if banglish_signals >= 1 and english_hits >= 1:
            return "mixed"
        if banglish_signals >= 1 and english_hits == 0:
            return "banglish"
        return "english"

    def tokenize(self, text: str) -> List[str]:
        return TOKEN_RE.findall(text)

    def detokenize(self, tokens: Sequence[str]) -> str:
        out: List[str] = []
        for tok in tokens:
            if not out:
                out.append(tok)
            elif self.is_punct(tok):
                out[-1] = out[-1] + tok
            elif out[-1].endswith(("(", "[", "{", "“", '"', "'")):
                out.append(tok)
            else:
                out.append(" " + tok)
        return "".join(out).strip()

    def replace_sensitive_spans(self, text: str, actions: List[RuleAction], flags: Dict) -> str:
        def repl_url(m):
            flags["has_url"] = True
            actions.append(RuleAction("R03_URL", m.group(0), "URLTOKEN", "placeholder", 1.0))
            return " URLTOKEN "

        def repl_email(m):
            flags["has_email"] = True
            actions.append(RuleAction("R04_EMAIL", m.group(0), "EMAILTOKEN", "placeholder", 1.0))
            return " EMAILTOKEN "

        def repl_phone(m):
            flags["has_phone"] = True
            actions.append(RuleAction("R05_PHONE", m.group(0), "PHONETOKEN", "placeholder", 0.95))
            return " PHONETOKEN "

        text = re.sub(r"https?://\S+|www\.\S+", repl_url, text, flags=re.IGNORECASE)
        text = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", repl_email, text)
        text = re.sub(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)", repl_phone, text)
        return text

    def safe_cleanup(self, text: str, actions: List[RuleAction], flags: Dict) -> str:
        original = "" if text is None else str(text)

        # Unicode compatibility normalization, then Bangla-specific normalization if installed.
        text_norm = unicodedata.normalize("NFKC", original)
        if text_norm != original:
            actions.append(RuleAction("R01_UNICODE_NFKC", original[:80], text_norm[:80], "safe_cleanup", 1.0))

        if self.bn_unicode:
            try:
                bn_norm = self.bn_unicode(text_norm)
                if isinstance(bn_norm, dict) and "normalized" in bn_norm:
                    bn_norm = bn_norm["normalized"]
                if isinstance(bn_norm, str) and bn_norm != text_norm:
                    actions.append(RuleAction("R02_BN_UNICODE", text_norm[:80], bn_norm[:80], "safe_cleanup", 1.0))
                    text_norm = bn_norm
            except Exception:
                pass

        replacements = {
            "\u200b": " ",
            "\ufeff": " ",
            "\u00a0": " ",
            "“": '"',
            "”": '"',
            "‘": "'",
            "’": "'",
            "–": "-",
            "—": "-",
            "…": "...",
            "\t": " ",
            "\r": " ",
            "\n": " ",
        }
        for old, new in replacements.items():
            if old in text_norm:
                text_norm = text_norm.replace(old, new)
                flags["layout_fixed"] = True

        # HTML fragments are noise in review text.
        if re.search(r"<[^>]+>", text_norm):
            text_norm = re.sub(r"<[^>]+>", " ", text_norm)
            actions.append(RuleAction("R06_HTML", "<tag>", " ", "safe_cleanup", 1.0))

        text_norm = self.replace_sensitive_spans(text_norm, actions, flags)

        # Bengali digits only inside numeric spans.
        if re.search(r"[০-৯]", text_norm):
            converted = re.sub(r"[০-৯]+", lambda m: m.group(0).translate(BENGALI_DIGIT_MAP), text_norm)
            if converted != text_norm:
                text_norm = converted
                flags["digit_map_applied"] = True
                actions.append(RuleAction("R07_DIGITS", "bengali_digits", "western_digits", "digit_policy", 1.0))

        # Punctuation cap preserves emphasis without massive token noise.
        def cap_punct(m):
            src = m.group(0)
            tgt = src[0] * min(len(src), self.punctuation_cap)
            if src != tgt:
                flags["punctuation_capped"] = True
                actions.append(RuleAction("R08_PUNCT_CAP", src, tgt, "safe_cleanup", 0.98))
            return tgt

        text_norm = re.sub(r"([!?.,])\1{3,}", cap_punct, text_norm)

        if EMOJI_RE.search(text_norm):
            flags["has_emoji"] = True

        if re.search(r"([A-Za-z\u0980-\u09FF])\1{2,}", text_norm):
            flags["has_elongation"] = True

        text_norm = re.sub(r"\s+", " ", text_norm).strip()
        return text_norm

    def should_keep_token(self, token: str) -> bool:
        t = token.lower()
        if t in self.keep_words:
            return True
        if self.is_bangla_token(token):
            return False
        for pat in self.keep_patterns:
            if pat.fullmatch(token.strip()):
                return True
        return False

    def protect_tokens(self, text: str, actions: List[RuleAction], flags: Dict) -> Tuple[str, Dict[str, str]]:
        mapping: Dict[str, str] = {}
        tokens = self.tokenize(text)
        out: List[str] = []
        keep_id = 0

        for tok in tokens:
            if self.should_keep_token(tok):
                key = f"__KEEP_{keep_id}__"
                mapping[key] = tok
                out.append(key)
                keep_id += 1
            else:
                out.append(tok)

        if mapping:
            flags["protected_span_count"] = len(mapping)
            actions.append(RuleAction("R09_PROTECT", f"{len(mapping)} spans", "protected", "masking", 1.0))

        return self.detokenize(out), mapping

    def restore_tokens(self, text: str, protected_map: Dict[str, str]) -> str:
        for key, value in protected_map.items():
            text = text.replace(key, value)
        return text

    def apply_phrase_map(self, text: str, actions: List[RuleAction], flags: Dict) -> str:
        out = text
        count = 0
        for phrase in self.sorted_phrase_keys:
            target = self.phrase_map[phrase]
            pattern = re.compile(rf"(?<!\w){re.escape(phrase)}(?!\w)", flags=re.IGNORECASE)
            new_out, n = pattern.subn(target, out)
            if n > 0:
                count += n
                actions.append(RuleAction("R10_PHRASE", phrase, target, "phrase_map", 1.0))
            out = new_out
        if count:
            flags["phrase_map_count"] = count
        return out

    def simplify_roman_token(self, token: str) -> str:
        t = token.lower()
        t = t.replace("0", "o").replace("1", "i").replace("3", "e").replace("5", "s")
        # Only for matching. Main text elongation is preserved unless a reliable mapping exists.
        t = re.sub(r"(.)\1{2,}", r"\1", t)
        for src, tgt in [
            ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"),
            ("iy", "i"), ("yy", "y"), ("ph", "f")
        ]:
            t = t.replace(src, tgt)
        return t

    def variants(self, token: str) -> List[str]:
        base = self.simplify_roman_token(token)
        variants = {token.lower(), base}
        swaps = [
            ("bh", "v"), ("v", "bh"),
            ("sh", "s"), ("s", "sh"),
            ("kh", "k"), ("k", "kh"),
            ("ch", "c"), ("c", "ch"),
            ("j", "z"), ("z", "j"),
            ("oo", "u"), ("u", "oo"),
        ]
        for src, tgt in swaps:
            if src in base:
                variants.add(base.replace(src, tgt, 1))
        return [v for v in variants if v]

    def looks_like_banglish(self, token: str) -> bool:
        t = token.lower()
        if not ROMAN_WORD_RE.fullmatch(t):
            return False
        if t in self.keep_words or t in self.english_words:
            return False
        if t in self.lookup_map:
            return True
        simple = self.simplify_roman_token(t)
        if simple in self.lookup_map:
            return True
        if any(simple.endswith(suf) for suf in ROMAN_BANGLISH_CUES):
            return True
        for v in self.variants(t):
            if v in self.lookup_map:
                return True
        if len(simple) >= 4:
            return bool(get_close_matches(simple, self.lookup_keys, n=1, cutoff=self.fuzzy_cutoff))
        return False

    def normalize_token(self, token: str) -> RuleAction:
        lower = token.lower()

        if self.is_punct(token) or self.is_bangla_token(token):
            return RuleAction("KEEP", token, token, "no_change", 1.0)

        if lower in self.keep_words or lower in self.english_words:
            return RuleAction("KEEP_EN", token, token, "preserve", 1.0)

        if lower in self.banglish_map:
            return RuleAction("R11_BANGLISH_LEX", token, self.banglish_map[lower], "banglish_lexicon", 1.0)

        if lower in self.dialect_map:
            return RuleAction("R12_DIALECT", token, self.dialect_map[lower], "dialect_map", 1.0)

        for v in self.variants(token):
            if v in self.lookup_map:
                return RuleAction("R13_VARIANT", token, self.lookup_map[v], "variant_match", 0.95)

        simple = self.simplify_roman_token(lower)
        if simple in self.lookup_map:
            return RuleAction("R14_SIMPLIFIED", token, self.lookup_map[simple], "simplified_match", 0.93)

        # Confidence-controlled fuzzy fallback.
        if len(simple) >= 4 and self.looks_like_banglish(token):
            match = get_close_matches(simple, self.lookup_keys, n=1, cutoff=self.fuzzy_cutoff)
            if match:
                return RuleAction("R15_FUZZY", token, self.lookup_map[match[0]], "confidence_fallback", 0.87)

        return RuleAction("KEEP_LOWCONF", token, token, "low_confidence_no_change", 0.0)

    def normalize_hybrid(self, safe_text: str, language_type: str, flags: Dict) -> Tuple[str, List[RuleAction]]:
        actions: List[RuleAction] = []

        if language_type == "bangla":
            return safe_text, actions

        # Run hybrid on Banglish, mixed, and English rows that contain Banglish signals.
        if language_type == "english":
            tokens = [t for t in self.tokenize(safe_text) if not self.is_punct(t)]
            if not any(self.looks_like_banglish(t) for t in tokens):
                return safe_text, actions

        protected_text, protected_map = self.protect_tokens(safe_text, actions, flags)
        phrased = self.apply_phrase_map(protected_text, actions, flags)
        tokens = self.tokenize(phrased)

        out_tokens: List[str] = []
        changed = 0
        for tok in tokens:
            if re.fullmatch(r"__KEEP_\d+__", tok):
                out_tokens.append(tok)
                continue

            decision = self.normalize_token(tok)
            out_tokens.append(decision.target)
            if decision.source != decision.target:
                changed += 1
                actions.append(decision)

        flags["token_change_count"] = changed
        detok = self.detokenize(out_tokens)
        restored = self.restore_tokens(detok, protected_map)
        restored = re.sub(r"\s+", " ", restored).strip()
        return restored, actions

    def normalize_review(self, text: str) -> Dict:
        all_actions: List[RuleAction] = []
        flags: Dict = {
            "norm_version": NORM_VERSION,
            "digit_policy": self.digit_policy,
            "has_url": False,
            "has_email": False,
            "has_phone": False,
            "has_emoji": False,
            "has_elongation": False,
        }

        raw = "" if text is None else str(text)
        safe = self.safe_cleanup(raw, all_actions, flags)
        lang_type = self.detect_language_type(safe)
        stats = self.script_stats(safe)

        hybrid, hybrid_actions = self.normalize_hybrid(safe, lang_type, flags)
        all_actions.extend(hybrid_actions)

        effective_actions = [a for a in all_actions if a.source != a.target and not a.rule_id.startswith("KEEP")]
        avg_conf = round(
            sum(a.confidence for a in effective_actions) / max(len(effective_actions), 1),
            3
        ) if effective_actions else 1.0

        return {
            "Review_raw": raw,
            "Review_norm_safe": safe,
            "Review_norm_hybrid": hybrid,
            "norm_language": lang_type,
            "script_ratio_bn": stats["script_ratio_bn"],
            "script_ratio_latin": stats["script_ratio_latin"],
            "norm_flags": json.dumps(flags, ensure_ascii=False),
            "norm_actions": json.dumps([asdict(a) for a in effective_actions], ensure_ascii=False),
            "norm_confidence": avg_conf,
            "norm_version": NORM_VERSION,
            "raw_hash": self.md5_text(raw),
        }

    def normalize_dataframe(self, df: pd.DataFrame, text_col: str = "Review") -> pd.DataFrame:
        rows = []
        for text in df[text_col].astype(str).tolist():
            rows.append(self.normalize_review(text))
        norm_df = pd.DataFrame(rows)

        out = df.copy()
        for col in norm_df.columns:
            out[col] = norm_df[col]
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output normalized CSV path")
    parser.add_argument("--text-col", default="Review", help="Review text column")
    parser.add_argument("--resource-dir", default="resources", help="Resource folder path")
    parser.add_argument("--fuzzy-cutoff", default=0.91, type=float)
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8", low_memory=False)
    normalizer = ResearchBanglaEcomNormalizer(args.resource_dir, fuzzy_cutoff=args.fuzzy_cutoff)
    out = normalizer.normalize_dataframe(df, text_col=args.text_col)
    out.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved normalized CSV: {args.output}")
    print(f"Rows: {len(out)}")
    print("Main text columns: Review_raw, Review_norm_safe, Review_norm_hybrid")


if __name__ == "__main__":
    main()
