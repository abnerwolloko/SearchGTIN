import json
import logging
import math
import os
import re
import statistics
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
APP_TOKEN = os.getenv("APP_TOKEN", "").strip()
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
}

GTIN_KEYS = {"gtin", "gtin8", "gtin12", "gtin13", "gtin14", "ean", "isbn"}
KIT_PATTERNS = [
    r"\bkit\b", r"\bcombo\b", r"\bconjunto\b", r"\bbundle\b", r"\bpack\b", r"\bduo\b",
    r"\bdupla\b", r"\bpar\b", r"\bc\/2\b", r"\bc\/3\b", r"\b2x\b", r"\b3x\b",
    r"\b2 unidades\b", r"\b3 unidades\b", r"\b2 un\b", r"\b3 un\b",
]
STOPWORDS = {
    "de", "da", "do", "das", "dos", "para", "com", "sem", "e", "a", "o", "as", "os",
    "um", "uma", "the", "with", "by", "in", "on", "new", "novo", "nova",
}
ACCEPTED_STATUSES = {"EXATO", "PROVAVEL"}
TEXT_FALLBACK_ACCEPTED_STATUSES = {"EXATO", "PROVAVEL"}
MAX_QUERY_ATTEMPTS = int(os.getenv("MAX_QUERY_ATTEMPTS", "5"))
MAX_SHOPPING_CANDIDATES = int(os.getenv("MAX_SHOPPING_CANDIDATES", "24"))

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Google Shopping GTIN Analyzer", version="4.3.0")


class ExternalAPIError(Exception):
    pass


class EntryInput(BaseModel):
    ean: str = Field(..., description="GTIN/EAN do produto")
    base_url: str = Field(..., description="URL de referencia do produto")

    @field_validator("ean")
    @classmethod
    def validate_ean(cls, v):
        d = digits_only(v)
        if len(d) not in (8, 12, 13, 14):
            raise ValueError("GTIN/EAN invalido.")
        return d

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v):
        v = (v or "").strip()
        if not v.lower().startswith(("http://", "https://")):
            raise ValueError("URL precisa comecar com http:// ou https://")
        return v


class AnalyzeRequest(BaseModel):
    entries: List[EntryInput]
    gl: str = Field(default="br")
    hl: str = Field(default="pt-BR")
    location: str = Field(default="Sao Paulo, State of Sao Paulo, Brazil")
    max_products_per_entry: int = Field(default=20, ge=5, le=30)

    @field_validator("entries")
    @classmethod
    def validate_entries(cls, values):
        if not values:
            raise ValueError("Informe pelo menos 1 entrada.")
        if len(values) > 10:
            raise ValueError("Maximo de 10 entradas por execucao.")
        seen, unique = set(), []
        for item in values:
            k = (item.ean, item.base_url)
            if k not in seen:
                unique.append(item)
                seen.add(k)
        return unique


# --- helpers ---

def digits_only(v):
    return re.sub(r"\D+", "", v or "")


def normalize_text(v):
    v = v or ""
    v = unicodedata.normalize("NFKD", v)
    v = "".join(ch for ch in v if not unicodedata.combining(ch))
    v = v.lower()
    v = re.sub(r"[^a-z0-9]+", " ", v)
    return re.sub(r"\s+", " ", v).strip()


def tokenize(v):
    return {t for t in normalize_text(v).split() if len(t) > 1 and t not in STOPWORDS}


def ordered_keywords(v, min_len=2):
    out, seen = [], set()
    for token in normalize_text(v).split():
        if len(token) < min_len or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def token_similarity(a, b):
    sa, sb = tokenize(a), tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def looks_like_kit_or_combo(text):
    txt = normalize_text(text)
    return any(re.search(p, txt) for p in KIT_PATTERNS)


def parse_money(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    sl = s.lower()
    if "free" in sl or "gratis" in sl:
        return 0.0
    c = re.sub(r"[^0-9,.\-]", "", s)
    if not c:
        return None
    if "," in c and "." in c:
        if c.rfind(",") > c.rfind("."):
            c = c.replace(".", "").replace(",", ".")
        else:
            c = c.replace(",", "")
    elif "," in c:
        c = c.replace(".", "").replace(",", ".")
    else:
        if c.count(".") > 1:
            c = c.replace(".", "")
    try:
        return float(c)
    except ValueError:
        return None


def money_br(v):
    if v is None:
        return "nao visivel"
    if v == 0.0:
        return "Gratis"
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def host_from_url(url):
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def canonicalize_url(url):
    if not url:
        return ""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower()
        path = re.sub(r"/+", "/", p.path or "")
        query_pairs = []
        if p.query:
            for piece in p.query.split("&"):
                key = piece.split("=", 1)[0].lower()
                if key.startswith("utm_") or key in {
                    "srsltid", "gclid", "fbclid", "ref", "source", "campaign", "clkid"
                }:
                    continue
                query_pairs.append(piece)
        query = "&".join(query_pairs)
        return f"{host}{path}?{query}" if query else f"{host}{path}"
    except Exception:
        return url


def marketplace_from_url(url, source=None):
    host = host_from_url(url)
    sn = normalize_text(source)
    if "mercadolivre" in host or "mercadolibre" in host or "mercado livre" in sn:
        return "Mercado Livre"
    if "amazon" in host or "amazon" in sn:
        return "Amazon"
    if "shopee" in host or "shopee" in sn:
        return "Shopee"
    if "magazineluiza" in host or "magalu" in host or "magalu" in sn:
        return "Magalu"
    if "americanas" in host or "americanas" in sn:
        return "Americanas"
    if "casasbahia" in host or "casas bahia" in sn:
        return "Casas Bahia"
    if "carrefour" in host or "carrefour" in sn:
        return "Carrefour"
    if "submarino" in host or "submarino" in sn:
        return "Submarino"
    if "extra.com" in host or "extra" in sn:
        return "Extra"
    if "kabum" in host or "kabum" in sn:
        return "KaBuM"
    if source and str(source).strip():
        return str(source).strip()
    return host or "nao visivel"


def infer_seller(store_name, source, url):
    if store_name and str(store_name).strip():
        return str(store_name).strip()
    if source and str(source).strip():
        return str(source).strip()
    host = host_from_url(url)
    if host:
        n = re.sub(r"^www\.", "", host)
        n = re.sub(r"\.(com\.br|com|br|net|org)$", "", n)
        return n.capitalize() if n else "nao visivel"
    return "nao visivel"


def slug_title_from_url(url):
    try:
        path = urlparse(url).path or ""
        slug = path.strip("/").split("/")[-1]
        slug = re.sub(r"[-_]+", " ", slug)
        slug = re.sub(r"\b(jm|p|dp|mlb\d+)\b", " ", slug, flags=re.I)
        slug = re.sub(r"\s+", " ", slug).strip()
        return slug if len(slug) >= 8 else None
    except Exception:
        return None


# --- fetcher ---

def requests_get_text(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.warning("Falha HTML %s: %s", url, e)
        return None


# --- gtin extraction ---

def collect_gtins_from_json(obj, out):
    if isinstance(obj, dict):
        for k, v in obj.items():
            kn = str(k).strip().lower()
            if kn in GTIN_KEYS:
                items = v if isinstance(v, list) else [v]
                for i in items:
                    d = digits_only(str(i))
                    if d:
                        out.add(d)
            else:
                collect_gtins_from_json(v, out)
    elif isinstance(obj, list):
        for i in obj:
            collect_gtins_from_json(i, out)


def extract_gtins_from_html(html):
    found = set()
    soup = BeautifulSoup(html or "", "html.parser")
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            collect_gtins_from_json(json.loads(raw), found)
        except Exception:
            for m in re.finditer(r'"(?:gtin\d*|ean)"\s*:\s*"(\d{8,14})"', raw, re.I):
                found.add(m.group(1))
    for tag in soup.select("[itemprop]"):
        ip = (tag.get("itemprop") or "").strip().lower()
        if ip in GTIN_KEYS:
            d = digits_only(tag.get("content") or tag.get_text(" ", strip=True))
            if d:
                found.add(d)
    for m in re.finditer(r'(?:gtin\d*|ean)[^0-9]{0,25}(\d{8,14})', html or "", re.I):
        found.add(m.group(1))
    return found


# --- product meta extraction ---

def _extract_schema_product(obj, meta):
    if isinstance(obj, list):
        for i in obj:
            _extract_schema_product(i, meta)
        return
    if not isinstance(obj, dict):
        return
    typ = str(obj.get("@type", "")).lower()
    if "product" in typ:
        if not meta.get("title") and obj.get("name"):
            meta["title"] = str(obj["name"]).strip()
        if not meta.get("brand"):
            br = obj.get("brand")
            if isinstance(br, dict):
                b = br.get("name") or br.get("@name") or ""
            elif isinstance(br, str):
                b = br
            else:
                b = ""
            if b and b.strip():
                meta["brand"] = b.strip()
        if not meta.get("model") and obj.get("model"):
            meta["model"] = str(obj["model"]).strip()
        if not meta.get("mpn") and obj.get("mpn"):
            meta["mpn"] = str(obj["mpn"]).strip()
        if not meta.get("category"):
            cat = obj.get("category")
            if cat:
                meta["category"] = (cat[0] if isinstance(cat, list) else str(cat)).strip()
    skip = {"@context", "@type", "gtin", "gtin8", "gtin12", "gtin13", "gtin14", "ean", "isbn"}
    for k, v in obj.items():
        if k in skip:
            continue
        if isinstance(v, (dict, list)):
            _extract_schema_product(v, meta)


def _meta_content(soup, *attrs):
    for a in attrs:
        tag = soup.find("meta", property=a) or soup.find("meta", attrs={"name": a})
        if tag:
            v = (tag.get("content") or "").strip()
            if v:
                return v
    return None


def extract_product_meta(html):
    meta = {"title": None, "brand": None, "model": None, "mpn": None, "category": None}
    soup = BeautifulSoup(html or "", "html.parser")

    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            _extract_schema_product(json.loads(raw), meta)
        except Exception:
            for field, key in [("name", "title"), ("brand", "brand"), ("model", "model"), ("mpn", "mpn")]:
                if not meta[key]:
                    m = re.search(rf'"{field}"\s*:\s*"([^"{{}}]+)"', raw, re.I)
                    if m:
                        v = m.group(1).strip()
                        if v:
                            meta[key] = v

    if not meta["title"]:
        meta["title"] = _meta_content(soup, "og:title", "twitter:title")
    if not meta["brand"]:
        meta["brand"] = _meta_content(soup, "product:brand", "og:brand", "brand")

    for ip, mk in [("name", "title"), ("brand", "brand"), ("model", "model"), ("mpn", "mpn")]:
        if meta[mk]:
            continue
        tag = soup.find(attrs={"itemprop": ip})
        if not tag:
            continue
        if ip == "brand":
            inner = tag.find(attrs={"itemprop": "name"})
            if inner:
                tag = inner
        v = (tag.get("content") or tag.get_text(" ", strip=True) or "").strip()
        if v:
            meta[mk] = v

    if not meta["title"] and soup.title:
        meta["title"] = soup.title.get_text(" ", strip=True)

    return {k: (v.strip() if isinstance(v, str) and v.strip() else None) for k, v in meta.items()}


# --- base reference ---

def fetch_base_reference(entry):
    ean = entry.ean
    url = entry.base_url
    notes = []
    html = requests_get_text(url)
    html_gtins = extract_gtins_from_html(html or "") if html else set()
    meta = extract_product_meta(html) if html else {}
    title = meta.get("title") or slug_title_from_url(url)
    brand = meta.get("brand")
    model = meta.get("model")
    mpn = meta.get("mpn")
    category = meta.get("category")
    gtins_found = sorted(html_gtins)

    if gtins_found:
        if ean in gtins_found:
            notes.append("GTIN/EAN confirmado na URL de referencia")
            base_status = "BASE CONFIRMADA"
        else:
            notes.append(f"GTIN/EAN divergente: encontrados {', '.join(gtins_found[:5])}")
            base_status = "BASE COM DIVERGENCIA"
    else:
        notes.append("GTIN/EAN nao visivel na URL de referencia")
        base_status = "BASE SEM GTIN VISIVEL"

    if looks_like_kit_or_combo(title):
        notes.append("URL de referencia parece kit/combo")
    if brand:
        notes.append(f"Marca extraida: {brand}")
    if model:
        notes.append(f"Modelo extraido: {model}")
    if mpn:
        notes.append(f"MPN extraido: {mpn}")
    if title:
        notes.append(f"Titulo base: {title[:120]}")

    return {
        "ean_input": ean,
        "base_url": url,
        "base_title": title or "nao visivel",
        "base_brand": brand,
        "base_model": model,
        "base_mpn": mpn,
        "base_category": category,
        "base_status": base_status,
        "base_notes": " | ".join(notes),
        "_gtins_set": set(gtins_found),
    }


# --- validacao ---

def validate_offer(ref, title, url, strict_text=False):
    notes = []
    if looks_like_kit_or_combo(title):
        return {
            "status": "EXCLUIDO - KIT/COMBO",
            "kit_combo_detectado": True,
            "notes": ["kit/combo no titulo"],
            "combined_score": 0.0,
        }

    gtins = set()
    if url and url != "nao visivel":
        html = requests_get_text(url)
        if html:
            gtins = extract_gtins_from_html(html)

    base_brand = ref.get("base_brand")
    base_model = ref.get("base_model") or ref.get("base_mpn")
    title_norm = normalize_text(title or "")
    similarity = token_similarity(title, ref.get("base_title"))
    brand_match = bool(base_brand and normalize_text(base_brand) in title_norm)
    model_match = bool(base_model and normalize_text(base_model) in title_norm)
    combined = round(min(1.0, similarity + (0.10 if brand_match else 0) + (0.15 if model_match else 0)), 4)

    if gtins:
        if ref["ean_input"] in gtins:
            return {
                "status": "EXATO",
                "kit_combo_detectado": False,
                "notes": ["GTIN exato confirmado"],
                "combined_score": 1.0,
            }
        notes.append(f"GTIN divergente: {', '.join(sorted(gtins)[:3])}")
        return {
            "status": "DIVERGENTE",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": 0.0,
        }

    if base_brand and not brand_match and combined < 0.80:
        notes.append(f"Marca '{base_brand}' ausente no candidato (score={combined:.2f})")
        return {
            "status": "DIVERGENTE",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": 0.0,
        }

    probable_threshold = 0.85 if strict_text else 0.72
    tentative_threshold = 0.65 if strict_text else 0.45

    if strict_text and base_brand and not brand_match:
        notes.append(f"Fallback textual rejeitado sem marca confirmada (score={combined:.2f})")
        return {
            "status": "DIVERGENTE",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": 0.0,
        }

    if combined >= probable_threshold and (brand_match or similarity >= 0.80 or model_match):
        boosts = [
            f"marca '{base_brand}'" if brand_match else None,
            f"modelo '{base_model}'" if model_match else None,
        ]
        detail = " | confirmados: " + ", ".join(b for b in boosts if b) if any(boosts) else ""
        notes.append(f"Validacao titulo/marca/modelo (score={combined:.2f}){detail}")
        return {
            "status": "PROVAVEL",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": combined,
        }

    if combined >= tentative_threshold:
        notes.append(f"Similaridade insuficiente ({combined:.2f})")
        return {
            "status": "NAO CONFIRMADO",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": 0.0,
        }

    return {
        "status": "DIVERGENTE",
        "kit_combo_detectado": False,
        "notes": [f"baixa aderencia ({combined:.2f})"],
        "combined_score": 0.0,
    }


def relevance_score(position, reviews, rating, multiple_sources, validation_boost=0.0, query_bonus=0.0):
    score = 0.0
    if position is not None:
        score += max(0.0, 40.0 - (position * 3.0))
    if reviews is not None:
        score += min(30.0, math.log10(reviews + 1) * 10.0)
    if rating is not None:
        score += rating * 6.0
    if multiple_sources:
        score += 8.0
    score += validation_boost * 20
    score += query_bonus
    return round(score, 2)


def relevance_label(score, reviews, position):
    if score >= 55 or (position is not None and position <= 3) or (reviews is not None and reviews >= 100):
        return "ALTA"
    if score >= 35:
        return "MEDIA"
    return "BAIXA"


# --- serpapi ---

def serpapi_request(params):
    if not SERPAPI_KEY:
        raise ExternalAPIError("SERPAPI_KEY nao configurada.")
    pl = dict(params)
    pl["api_key"] = SERPAPI_KEY
    try:
        r = requests.get(SERPAPI_ENDPOINT, params=pl, headers=HEADERS, timeout=HTTP_TIMEOUT)
    except requests.RequestException as e:
        raise ExternalAPIError(f"Falha SerpApi: {e}") from e
    preview = r.text[:500] if r.text else ""
    if r.status_code >= 400:
        try:
            msg = r.json().get("error") or preview
        except Exception:
            msg = preview
        raise ExternalAPIError(f"SerpApi HTTP {r.status_code}: {msg}")
    try:
        data = r.json()
    except ValueError as e:
        raise ExternalAPIError(f"Resposta invalida: {preview}") from e
    if data.get("error"):
        raise ExternalAPIError(f"SerpApi erro: {data['error']}")
    if ((data.get("search_metadata") or {}).get("status") or "").lower() == "error":
        raise ExternalAPIError(f"SerpApi status=Error: {preview}")
    return data


def is_no_results_error(msg):
    m = (msg or "").lower()
    return "hasn't returned any results" in m or "returned any results for this query" in m


def allow_text_fallback(ref):
    if ref.get("base_status") == "BASE COM DIVERGENCIA":
        return False
    if looks_like_kit_or_combo(ref.get("base_title")):
        return False
    return bool(ref.get("base_brand") or ref.get("base_model") or ref.get("base_mpn"))


def build_search_queries(ref):
    ean = ref["ean_input"]
    title = ref["base_title"] if ref["base_title"] != "nao visivel" else ""
    brand = (ref.get("base_brand") or "").strip()
    model = (ref.get("base_model") or ref.get("base_mpn") or "").strip()

    brand_tok = set(ordered_keywords(brand, min_len=2))
    model_tok = set(ordered_keywords(model, min_len=2))
    title_tokens = [t for t in ordered_keywords(title, min_len=3) if t not in brand_tok and t not in model_tok]

    ctx_parts = []
    if brand:
        ctx_parts.append(brand)
    if model:
        ctx_parts.append(model)
    context = " ".join(ctx_parts).strip()
    extra = " ".join(title_tokens[:4])
    full_context = (context + " " + extra).strip() if extra else context
    short_title = " ".join(title_tokens[:6]) if title_tokens else ""

    candidates = [
        {"q": f'"{ean}" {full_context}'.strip() if full_context else f'"{ean}"', "kind": "ean_context", "priority": 100},
        {"q": f'"{ean}" {context}'.strip() if context and context != full_context else None, "kind": "ean_context", "priority": 95},
        {"q": f'"{ean}" {brand}'.strip() if brand else None, "kind": "ean_brand", "priority": 92},
        {"q": f'"{ean}"', "kind": "ean_only", "priority": 90},
    ]

    if allow_text_fallback(ref):
        textual = full_context if full_context else short_title
        if textual:
            candidates.append({"q": textual, "kind": "text_fallback", "priority": 30})

    out, seen = [], set()
    for item in candidates:
        q = item.get("q")
        if not q:
            continue
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            item["q"] = q
            out.append(item)
            seen.add(q)
    return out[:MAX_QUERY_ATTEMPTS] or [{"q": f'"{ean}"', "kind": "ean_only", "priority": 90}]


def shopping_search_collect(ref, gl, hl, location):
    attempts = []
    collected = []
    seen_products = set()
    last_error = None

    for query in build_search_queries(ref):
        q = query["q"]
        kind = query["kind"]
        try:
            data = serpapi_request({
                "engine": "google_shopping",
                "q": q,
                "gl": gl,
                "hl": hl,
                "location": location,
                "no_cache": "true",
            })
            items = data.get("shopping_results", []) or []
            attempts.append({"query": q, "kind": kind, "hits": len(items)})
            for item in items:
                key = item.get("product_id") or canonicalize_url(item.get("product_link")) or (item.get("title") or "")
                if key in seen_products:
                    continue
                seen_products.add(key)
                copy = dict(item)
                copy["_query_used"] = q
                copy["_query_kind"] = kind
                copy["_query_priority"] = query["priority"]
                collected.append(copy)
                if len(collected) >= MAX_SHOPPING_CANDIDATES:
                    break
            if len(collected) >= MAX_SHOPPING_CANDIDATES:
                break
        except ExternalAPIError as e:
            last_error = str(e)
            attempts.append({"query": q, "kind": kind, "hits": 0, "error": last_error})
            if is_no_results_error(last_error):
                continue
            raise

    return {
        "shopping_results": collected,
        "attempts": attempts,
        "last_error": last_error,
    }


def immersive_product(page_token, gl, hl, location, next_page_token=None):
    params = {
        "engine": "google_immersive_product",
        "gl": gl,
        "hl": hl,
        "location": location,
        "no_cache": "true",
        "more_stores": "true",
    }
    if next_page_token:
        params["next_page_token"] = next_page_token
    else:
        params["page_token"] = page_token
    return serpapi_request(params)


# --- build top10 ---

def build_google_top10(ref, gl, hl, location, max_products_per_entry):
    search_data = shopping_search_collect(ref, gl, hl, location)
    shopping_results = (search_data.get("shopping_results") or [])[:max_products_per_entry]
    all_rows = []
    seen = set()

    if not shopping_results:
        return [], search_data

    for search_rank, result in enumerate(shopping_results, start=1):
        ct = result.get("title")
        cp = result.get("position") or search_rank
        cs = result.get("source")
        cl = result.get("product_link")
        cms = bool(result.get("multiple_sources"))
        cpn = result.get("extracted_price") or parse_money(result.get("price"))
        cdn = parse_money(result.get("delivery"))
        ctn = result.get("extracted_total") or (cpn + cdn if cpn is not None and cdn is not None else cpn)
        pt = result.get("immersive_product_page_token")
        created = False
        strict_text = result.get("_query_kind") == "text_fallback"
        query_bonus = 10 if result.get("_query_kind") in {"ean_context", "ean_brand"} else (6 if result.get("_query_kind") == "ean_only" else 0)

        if pt:
            npt = None
            pages = 0
            while pages < 2:
                try:
                    pd = immersive_product(page_token=pt, gl=gl, hl=hl, location=location, next_page_token=npt)
                except ExternalAPIError as e:
                    logger.warning("Immersive fail %s: %s", ref["ean_input"], e)
                    break
                pr = pd.get("product_results", {}) or {}
                for store in (pr.get("stores", []) or []):
                    sl = store.get("link")
                    v = validate_offer(ref, store.get("title") or ct, sl, strict_text=strict_text)
                    accepted_statuses = TEXT_FALLBACK_ACCEPTED_STATUSES if strict_text else ACCEPTED_STATUSES
                    if v["kit_combo_detectado"] or v["status"] not in accepted_statuses:
                        logger.debug("Descartado status=%s ean=%s", v["status"], ref["ean_input"])
                        continue
                    rev = store.get("reviews")
                    rat = store.get("rating")
                    sc = relevance_score(cp, rev, rat, cms, float(v.get("combined_score") or 0), query_bonus=query_bonus)
                    pn = store.get("extracted_price") or parse_money(store.get("price"))
                    shn = store.get("shipping_extracted")
                    if shn is None:
                        shn = parse_money(store.get("shipping"))
                    tn = store.get("extracted_total") or (pn + shn if pn is not None and shn is not None else pn)
                    row = {
                        "ean": ref["ean_input"],
                        "ranking": 0,
                        "seller": infer_seller(store.get("name"), cs, sl),
                        "marketplace": marketplace_from_url(sl, cs),
                        "produto": store.get("title") or ct or "nao visivel",
                        "preco_produto": money_br(pn),
                        "frete_gratis": "Sim" if shn == 0.0 else "Nao",
                        "preco_total": money_br(tn),
                        "link": sl or cl or "nao visivel",
                        "status_validacao": v["status"],
                        "relevancia": relevance_label(sc, rev, cp),
                        "query_usada": result.get("_query_used") or "nao visivel",
                        "modo_busca": result.get("_query_kind") or "nao visivel",
                        "_score": sc,
                        "_price_num": tn,
                        "_query_priority": result.get("_query_priority", 0),
                    }
                    key = f'{row["ean"]}|{canonicalize_url(row["link"])}|{row["seller"]}'
                    if key not in seen:
                        seen.add(key)
                        all_rows.append(row)
                        created = True
                npt = pr.get("stores_next_page_token")
                pages += 1
                if not npt:
                    break

        if not created:
            v = validate_offer(ref, ct, cl, strict_text=strict_text)
            accepted_statuses = TEXT_FALLBACK_ACCEPTED_STATUSES if strict_text else ACCEPTED_STATUSES
            if v["kit_combo_detectado"] or v["status"] not in accepted_statuses:
                logger.debug("Descartado fallback status=%s ean=%s", v["status"], ref["ean_input"])
                continue
            rev = result.get("reviews")
            rat = result.get("rating")
            sc = relevance_score(cp, rev, rat, cms, float(v.get("combined_score") or 0), query_bonus=query_bonus)
            row = {
                "ean": ref["ean_input"],
                "ranking": 0,
                "seller": infer_seller(None, cs, cl),
                "marketplace": marketplace_from_url(cl, cs),
                "produto": ct or "nao visivel",
                "preco_produto": money_br(cpn),
                "frete_gratis": "Sim" if cdn == 0.0 else "Nao",
                "preco_total": money_br(ctn),
                "link": cl or "nao visivel",
                "status_validacao": v["status"],
                "relevancia": relevance_label(sc, rev, cp),
                "query_usada": result.get("_query_used") or "nao visivel",
                "modo_busca": result.get("_query_kind") or "nao visivel",
                "_score": sc,
                "_price_num": ctn,
                "_query_priority": result.get("_query_priority", 0),
            }
            key = f'{row["ean"]}|{canonicalize_url(row["link"])}|{row["seller"]}'
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    all_rows.sort(key=lambda x: (-x["_score"], -x.get("_query_priority", 0), x["_price_num"] or 1e9))
    top10 = all_rows[:10]
    for i, row in enumerate(top10, 1):
        row["ranking"] = i
        row.pop("_score", None)
        row.pop("_price_num", None)
        row.pop("_query_priority", None)
    return top10, search_data


# --- resumo ---

def percentile(values, p):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * (p / 100.0)
    lo, hi = math.floor(rank), math.ceil(rank)
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def summarize_entry(ref, top10, search_data):
    prices = []
    for r in top10:
        raw = r.get("preco_total", "")
        num = parse_money((raw.replace("R$", "").replace(".", "").replace(",", ".").strip()) if isinstance(raw, str) else raw)
        if num is not None and num > 0:
            prices.append(num)
    mn = min(prices) if prices else None
    mx = max(prices) if prices else None
    av = statistics.mean(prices) if prices else None
    p25 = percentile(prices, 25)
    p45 = percentile(prices, 45)
    lider = top10[0] if top10 else None
    attempts = search_data.get("attempts") or []
    query_used = "; ".join([a["query"] for a in attempts if a.get("hits")])[:500] if attempts else "nao visivel"
    query_debug = " | ".join([
        f"{a.get('kind')}={a.get('hits', 0)}" + (f" ({a.get('error')[:60]})" if a.get("error") else "")
        for a in attempts
    ])[:500] if attempts else "nao visivel"

    observacao = None
    if not top10:
        if attempts and all(a.get("hits", 0) == 0 for a in attempts):
            observacao = "Sem resultados no Google Shopping para as queries testadas"
        else:
            observacao = "Resultados encontrados, mas rejeitados na validacao cruzada"

    return {
        "ean": ref["ean_input"],
        "base_url": ref["base_url"],
        "base_title": ref["base_title"],
        "base_brand": ref.get("base_brand") or "nao visivel",
        "base_model": ref.get("base_model") or ref.get("base_mpn") or "nao visivel",
        "base_status": ref["base_status"],
        "base_notes": ref.get("base_notes") or "nao visivel",
        "query_usada": query_used or "nao visivel",
        "query_debug": query_debug or "nao visivel",
        "menor_preco": money_br(mn),
        "preco_medio": money_br(av),
        "maior_preco": money_br(mx),
        "faixa_ideal_sugerida": (f"{money_br(p25)} a {money_br(p45)}" if p25 is not None and p45 is not None else "nao visivel"),
        "vendedores_com_frete_gratis": sum(1 for r in top10 if r.get("frete_gratis") == "Sim"),
        "total_vendedores_top10": len(top10),
        "seller_lider": lider["seller"] if lider else "nao visivel",
        "marketplace_lider": lider["marketplace"] if lider else "nao visivel",
        "observacao": observacao or "",
    }


# --- processo paralelo ---

def process_entry(entry, gl, hl, location, max_products_per_entry):
    ref = fetch_base_reference(entry)
    top10, search_data = build_google_top10(ref, gl, hl, location, max_products_per_entry)
    return {
        "ean": entry.ean,
        "summary": summarize_entry(ref, top10, search_data),
        "top10": top10,
    }


# --- endpoints ---

@app.get("/health")
def health():
    return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest, x_app_token: Optional[str] = Header(default=None)):
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido.")
    n = len(payload.entries)
    results = {}
    with ThreadPoolExecutor(max_workers=min(n, 5)) as ex:
        fmap = {
            ex.submit(process_entry, e, payload.gl, payload.hl, payload.location, payload.max_products_per_entry): i
            for i, e in enumerate(payload.entries)
        }
        for f in as_completed(fmap):
            i = fmap[f]
            try:
                results[i] = f.result()
            except ExternalAPIError as e:
                results[i] = {"ean": payload.entries[i].ean, "summary": {}, "top10": [], "error": str(e)}
            except Exception as e:
                logger.exception("Erro EAN %s", payload.entries[i].ean)
                results[i] = {"ean": payload.entries[i].ean, "summary": {}, "top10": [], "error": f"erro interno: {e}"}
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "location": payload.location,
            "gl": payload.gl,
            "hl": payload.hl,
        },
        "products": [results[i] for i in range(n)],
    }
