
import json
import logging
import math
import os
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import quote_plus, urlparse

import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field, field_validator

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
MELI_API_BASE = "https://api.mercadolibre.com"
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
APP_TOKEN = os.getenv("APP_TOKEN", "").strip()
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "35"))
SITE_ID = os.getenv("MELI_SITE_ID", "MLB").strip() or "MLB"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 Safari/537.36"
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
ATTRIBUTE_CANDIDATES = ["BRAND", "MODEL", "LINE", "MPN", "GTIN", "EAN", "COLOR", "VERSION", "UNITS_PER_PACK", "UNITS_PER_PACKAGE"]
STOPWORDS = {
    "de", "da", "do", "das", "dos", "para", "com", "sem", "e", "a", "o", "as", "os",
    "um", "uma", "the", "with", "by", "in", "on", "new", "novo", "nova"
}

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Google Shopping GTIN Analyzer - Modo 2", version="2.0.0")


class ExternalAPIError(Exception):
    pass


class EntryInput(BaseModel):
    ean: str = Field(..., description="GTIN/EAN do produto")
    base_url: str = Field(..., description="URL base do anúncio do Mercado Livre")

    @field_validator("ean")
    @classmethod
    def validate_ean(cls, value: str) -> str:
        digits = digits_only(value)
        if len(digits) not in (8, 12, 13, 14):
            raise ValueError("GTIN/EAN inválido.")
        return digits

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        value = (value or "").strip()
        if not value.lower().startswith(("http://", "https://")):
            raise ValueError("A URL base precisa começar com http:// ou https://")
        host = urlparse(value).netloc.lower()
        if "mercadolivre" not in host and "mercadolibre" not in host:
            raise ValueError("A URL base precisa ser de um anúncio do Mercado Livre.")
        return value


class AnalyzeRequest(BaseModel):
    entries: List[EntryInput]
    gl: str = Field(default="br")
    hl: str = Field(default="pt-BR")
    location: str = Field(default="Brazil")
    max_products_per_entry: int = Field(default=4, ge=1, le=8)
    max_ml_results: int = Field(default=30, ge=10, le=50)

    @field_validator("entries")
    @classmethod
    def validate_entries(cls, values: List[EntryInput]) -> List[EntryInput]:
        if not values:
            raise ValueError("Informe pelo menos 1 entrada.")
        if len(values) > 10:
            raise ValueError("Máximo de 10 entradas por execução.")
        seen = set()
        unique = []
        for item in values:
            key = (item.ean, item.base_url)
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique


def digits_only(value: str) -> str:
    return re.sub(r"\D+", "", value or "")


def normalize_text(value: Optional[str]) -> str:
    value = value or ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def tokenize(value: Optional[str]) -> Set[str]:
    return {t for t in normalize_text(value).split() if len(t) > 1 and t not in STOPWORDS}


def token_similarity(a: Optional[str], b: Optional[str]) -> float:
    sa = tokenize(a)
    sb = tokenize(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def looks_like_kit_or_combo(text: Optional[str]) -> bool:
    txt = normalize_text(text)
    return any(re.search(pattern, txt) for pattern in KIT_PATTERNS)


def parse_money(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    s_low = s.lower()
    if "free" in s_low or "gratis" in s_low or "grátis" in s_low:
        return 0.0
    cleaned = re.sub(r"[^0-9,.\-]", "", s)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(".", "").replace(",", ".")
    else:
        if cleaned.count(".") > 1:
            cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def money_br(value: Optional[float]) -> str:
    if value is None:
        return "não visível"
    s = f"{value:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def host_from_url(url: Optional[str]) -> str:
    if not url:
        return ""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def marketplace_from_url(url: Optional[str], source: Optional[str] = None) -> str:
    host = host_from_url(url)
    source_norm = normalize_text(source)
    if "mercadolivre" in host or "mercadolibre" in host or "mercado livre" in source_norm:
        return "Mercado Livre"
    if "amazon" in host or "amazon" in source_norm:
        return "Amazon"
    if "shopee" in host or "shopee" in source_norm:
        return "Shopee"
    if "magazineluiza" in host or "magalu" in host or "magalu" in source_norm:
        return "Magalu"
    if "americanas" in host or "americanas" in source_norm:
        return "Americanas"
    if "casasbahia" in host or "casas bahia" in source_norm:
        return "Casas Bahia"
    if "carrefour" in host or "carrefour" in source_norm:
        return "Carrefour"
    return source or host or "não visível"


def extract_item_id_from_url(url: str) -> Optional[str]:
    patterns = [
        r"\b(ML[A-Z]{1,3}\d{6,})\b",
        r"item_id[:=](ML[A-Z]{1,3}\d{6,})",
        r"/p/(ML[A-Z]{1,3}\d{6,})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url, flags=re.I)
        if match:
            return match.group(1).upper()
    return None


def requests_get_json(url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        logger.warning("Falha ao obter JSON %s: %s", url, exc)
        return None


def requests_get_text(url: str) -> Optional[str]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
        return response.text
    except Exception as exc:
        logger.warning("Falha ao obter HTML %s: %s", url, exc)
        return None


def collect_gtins_from_json(obj: Any, out: Set[str]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).strip().lower()
            if key_norm in GTIN_KEYS:
                if isinstance(value, list):
                    for item in value:
                        d = digits_only(str(item))
                        if d:
                            out.add(d)
                else:
                    d = digits_only(str(value))
                    if d:
                        out.add(d)
            else:
                collect_gtins_from_json(value, out)
    elif isinstance(obj, list):
        for item in obj:
            collect_gtins_from_json(item, out)


def extract_gtins_from_html(html: str) -> Set[str]:
    found: Set[str] = set()
    soup = BeautifulSoup(html or "", "html.parser")
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            collect_gtins_from_json(data, found)
        except Exception:
            for match in re.finditer(r'"(?:gtin|gtin8|gtin12|gtin13|gtin14|ean)"\s*:\s*"(\d{8,14})"', raw, flags=re.I):
                found.add(match.group(1))
    for tag in soup.select("[itemprop]"):
        itemprop = (tag.get("itemprop") or "").strip().lower()
        if itemprop in GTIN_KEYS:
            value = tag.get("content") or tag.get_text(" ", strip=True)
            d = digits_only(value)
            if d:
                found.add(d)
    for match in re.finditer(r'(?:gtin|gtin8|gtin12|gtin13|gtin14|ean)[^0-9]{0,25}(\d{8,14})', html, flags=re.I):
        found.add(match.group(1))
    return found


def get_attribute_value(attributes_map: Dict[str, Any], key: str) -> Optional[str]:
    value = attributes_map.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def attributes_to_map(attributes: List[Dict[str, Any]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for attr in attributes or []:
        attr_id = str(attr.get("id") or "").strip().upper()
        if not attr_id:
            continue
        value = attr.get("value_name")
        if value is None:
            value = attr.get("value_id")
        if value is None:
            value_struct = attr.get("value_struct") or {}
            value = value_struct.get("number") if isinstance(value_struct, dict) else None
        if value is not None and str(value).strip():
            out[attr_id] = str(value).strip()
    return out


def fetch_base_reference(entry: EntryInput) -> Dict[str, Any]:
    ean = entry.ean
    url = entry.base_url
    item_id = extract_item_id_from_url(url)
    notes: List[str] = []
    item_data = requests_get_json(f"{MELI_API_BASE}/items/{item_id}") if item_id else None
    html = requests_get_text(url)
    html_gtins = extract_gtins_from_html(html or "") if html else set()

    title = None
    permalink = url
    category_id = None
    seller_id = None
    seller_nickname = None
    attributes_map: Dict[str, str] = {}
    api_gtins: Set[str] = set()

    if item_data:
        title = item_data.get("title")
        permalink = item_data.get("permalink") or url
        category_id = item_data.get("category_id")
        seller_id = item_data.get("seller_id")
        attributes_map = attributes_to_map(item_data.get("attributes") or [])
        for key in ["GTIN", "EAN"]:
            val = get_attribute_value(attributes_map, key)
            d = digits_only(val or "")
            if d:
                api_gtins.add(d)

        seller_data = requests_get_json(f"{MELI_API_BASE}/users/{seller_id}") if seller_id else None
        if seller_data:
            seller_nickname = seller_data.get("nickname")

    if not title and html:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)

    gtins_found = sorted(api_gtins | html_gtins)
    brand = get_attribute_value(attributes_map, "BRAND")
    model = get_attribute_value(attributes_map, "MODEL")
    line = get_attribute_value(attributes_map, "LINE")
    mpn = get_attribute_value(attributes_map, "MPN")
    color = get_attribute_value(attributes_map, "COLOR")
    version = get_attribute_value(attributes_map, "VERSION")
    units_pack = get_attribute_value(attributes_map, "UNITS_PER_PACK")
    units_package = get_attribute_value(attributes_map, "UNITS_PER_PACKAGE")

    if gtins_found:
        if ean in gtins_found:
            notes.append("GTIN/EAN confirmado na URL base")
            base_status = "BASE CONFIRMADA"
        else:
            notes.append(f"GTIN/EAN divergente na URL base: encontrados {', '.join(gtins_found[:5])}")
            base_status = "BASE COM DIVERGÊNCIA"
    else:
        notes.append("GTIN/EAN não visível na URL base")
        base_status = "BASE SEM GTIN VISÍVEL"

    if looks_like_kit_or_combo(title):
        notes.append("URL base parece kit/combo; atenção na validação")

    return {
        "ean_input": ean,
        "base_url": url,
        "base_item_id": item_id or "não visível",
        "base_title": title or "não visível",
        "base_permalink": permalink,
        "base_category_id": category_id or "não visível",
        "base_seller_id": seller_id or "não visível",
        "base_seller_nickname": seller_nickname or "não visível",
        "brand": brand or "não visível",
        "model": model or "não visível",
        "line": line or "não visível",
        "mpn": mpn or "não visível",
        "color": color or "não visível",
        "version": version or "não visível",
        "units_per_pack": units_pack or "não visível",
        "units_per_package": units_package or "não visível",
        "gtins_found": ", ".join(gtins_found) if gtins_found else "não visível",
        "base_status": base_status,
        "base_notes": " | ".join(notes),
        "_attributes_map": attributes_map,
        "_gtins_set": set(gtins_found),
    }


def build_search_queries(ref: Dict[str, Any]) -> List[str]:
    ean = ref["ean_input"]
    title = ref["base_title"] if ref["base_title"] != "não visível" else ""
    brand = ref["brand"] if ref["brand"] != "não visível" else ""
    model = ref["model"] if ref["model"] != "não visível" else ""
    line = ref["line"] if ref["line"] != "não visível" else ""
    mpn = ref["mpn"] if ref["mpn"] != "não visível" else ""

    title_tokens = [t for t in tokenize(title) if len(t) > 2]
    short_title = " ".join(list(title_tokens)[:8])

    candidates = [
        f'"{title}" "{ean}"' if title else "",
        f'{brand} {model} {ean}'.strip(),
        f'{brand} {line} {model} {ean}'.strip(),
        f'{brand} {mpn} {ean}'.strip(),
        f'ean {ean} {short_title}'.strip(),
    ]
    out: List[str] = []
    seen = set()
    for item in candidates:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    if not out:
        out = [ean]
    return out


def serpapi_request(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERPAPI_KEY:
        raise ExternalAPIError("SERPAPI_KEY não configurada no Render.")
    payload = dict(params)
    payload["api_key"] = SERPAPI_KEY
    try:
        response = requests.get(SERPAPI_ENDPOINT, params=payload, headers=HEADERS, timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise ExternalAPIError(f"Falha de conexão com a SerpApi: {exc}") from exc

    preview = response.text[:500] if response.text else ""
    if response.status_code >= 400:
        try:
            data = response.json()
            message = data.get("error") or data.get("message") or preview
        except Exception:
            message = preview or f"HTTP {response.status_code}"
        raise ExternalAPIError(f"SerpApi HTTP {response.status_code}: {message}")

    try:
        data = response.json()
    except ValueError as exc:
        raise ExternalAPIError(f"Resposta inválida da SerpApi: {preview}") from exc

    status = ((data.get("search_metadata") or {}).get("status") or "").lower()
    if data.get("error"):
        raise ExternalAPIError(f"SerpApi retornou erro: {data['error']}")
    if status == "error":
        raise ExternalAPIError(f"SerpApi search_metadata.status=Error: {preview}")
    return data


def is_no_results_error(message: str) -> bool:
    msg = (message or "").lower()
    return "hasn't returned any results" in msg or "returned any results for this query" in msg


def shopping_search(ref: Dict[str, Any], gl: str, hl: str, location: str) -> Dict[str, Any]:
    last_error: Optional[str] = None
    queries = build_search_queries(ref)
    for query in queries:
        try:
            data = serpapi_request({
                "engine": "google_shopping",
                "q": query,
                "gl": gl,
                "hl": hl,
                "location": location,
                "no_cache": "true",
            })
            if data.get("shopping_results"):
                data["_query_used"] = query
                return data
            last_error = "Google Shopping não retornou resultados."
        except ExternalAPIError as exc:
            last_error = str(exc)
            if is_no_results_error(last_error):
                continue
            raise
    return {"shopping_results": [], "_query_used": None, "_notes": [last_error or "Sem resultados."]}


def immersive_product(page_token: str, gl: str, hl: str, location: str, next_page_token: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {
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


def infer_seller(store_name: Optional[str], source: Optional[str], url: Optional[str]) -> str:
    if store_name and str(store_name).strip():
        return str(store_name).strip()
    if source and str(source).strip():
        return str(source).strip()
    return host_from_url(url) or "não visível"


def attribute_match_score(title: Optional[str], ref: Dict[str, Any]) -> float:
    score = 0.0
    title_norm = normalize_text(title)
    for key in ["brand", "model", "line", "mpn", "color", "version"]:
        value = ref.get(key)
        if value and value != "não visível":
            norm = normalize_text(str(value))
            if norm and norm in title_norm:
                score += 0.12
    for key in ["units_per_pack", "units_per_package"]:
        value = ref.get(key)
        if value and value != "não visível":
            raw = digits_only(str(value))
            if raw and raw in digits_only(title_norm):
                score += 0.08
    return min(score, 0.5)


def validate_offer(ref: Dict[str, Any], title: Optional[str], url: Optional[str]) -> Dict[str, Any]:
    notes: List[str] = []
    if looks_like_kit_or_combo(title):
        return {"status": "EXCLUÍDO - KIT/COMBO", "kit_combo_detectado": True, "notes": ["kit/combo detectado no título"]}

    gtins = set()
    if url and url != "não visível":
        html = requests_get_text(url)
        if html:
            gtins = extract_gtins_from_html(html)

    ref_gtins = set(ref.get("_gtins_set") or [])
    if ref["ean_input"]:
        ref_gtins.add(ref["ean_input"])

    similarity = token_similarity(title, ref.get("base_title"))
    attr_score = attribute_match_score(title, ref)
    combined = round(min(1.0, similarity + attr_score), 4)

    if gtins:
        if ref["ean_input"] in gtins:
            notes.append("GTIN exato confirmado na página")
            return {"status": "EXATO", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}
        notes.append(f"GTIN divergente encontrado na página: {', '.join(sorted(gtins)[:5])}")
        return {"status": "DIVERGENTE", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}

    if combined >= 0.72:
        notes.append(f"GTIN não visível; validação forte por título/atributos ({combined:.2f})")
        return {"status": "PROVÁVEL", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}
    if combined >= 0.45:
        notes.append(f"GTIN não visível; sem confirmação plena ({combined:.2f})")
        return {"status": "NÃO CONFIRMADO", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}

    return {"status": "DIVERGENTE", "kit_combo_detectado": False, "notes": [f"baixa aderência ao anúncio base ({combined:.2f})"], "combined_score": combined}


def build_proxy(position: Optional[int], reviews: Optional[int], rating: Optional[float], multiple_sources: bool) -> str:
    parts: List[str] = []
    if position is not None:
        parts.append(f"posição {position}")
    if reviews is not None:
        parts.append(f"{reviews} avaliações")
    if rating is not None:
        parts.append(f"nota {rating}")
    parts.append("multi-ofertas" if multiple_sources else "oferta única")
    return " | ".join(parts) if parts else "não visível"


def relevance_score(position: Optional[int], reviews: Optional[int], rating: Optional[float], multiple_sources: bool, validation_boost: float = 0.0) -> float:
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
    return round(score, 2)


def relevance_label(score: float, reviews: Optional[int], position: Optional[int]) -> str:
    if score >= 55 or (position is not None and position <= 3) or (reviews is not None and reviews >= 100):
        return "ALTA RELEVÂNCIA"
    if score >= 35:
        return "MÉDIA RELEVÂNCIA"
    return "BAIXA RELEVÂNCIA"


def build_google_rows(ref: Dict[str, Any], gl: str, hl: str, location: str, max_products_per_entry: int) -> List[Dict[str, Any]]:
    search_data = shopping_search(ref, gl, hl, location)
    shopping_results = search_data.get("shopping_results", []) or []
    query_used = search_data.get("_query_used")
    search_notes = search_data.get("_notes", []) or []

    rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    if not shopping_results:
        return [{
            "ean": ref["ean_input"],
            "base_url": ref["base_url"],
            "seller": "não visível",
            "marketplace": "não visível",
            "produto_google": "não visível",
            "produto_loja": "não visível",
            "preco_atual": "não visível",
            "frete": "não visível",
            "preco_total_estimado": "não visível",
            "quantidade_vendida_ou_proxy": "não visível",
            "avaliacoes": "não visível",
            "nota": "não visível",
            "url_completo": "não visível",
            "observacoes": "SEM RESULTADOS | " + " | ".join(search_notes),
            "alta_relevancia": "não visível",
            "relevancia_score": 0.0,
            "status_validacao": "SEM RESULTADOS",
            "origem_google_posicao": None,
        }]

    for search_rank, result in enumerate(shopping_results[:max_products_per_entry], start=1):
        candidate_title = result.get("title")
        candidate_position = result.get("position") or search_rank
        candidate_source = result.get("source")
        candidate_product_link = result.get("product_link")
        candidate_multiple_sources = bool(result.get("multiple_sources"))
        candidate_price_num = result.get("extracted_price") or parse_money(result.get("price"))
        candidate_delivery_num = parse_money(result.get("delivery"))
        candidate_total_num = (
            candidate_price_num + candidate_delivery_num
            if candidate_price_num is not None and candidate_delivery_num is not None
            else candidate_price_num
        )
        page_token = result.get("immersive_product_page_token")
        product_rows_created = False

        if page_token:
            next_page_token = None
            pages_collected = 0
            while pages_collected < 2:
                try:
                    product_data = immersive_product(page_token=page_token, gl=gl, hl=hl, location=location, next_page_token=next_page_token)
                except ExternalAPIError as exc:
                    logger.warning("Falha no immersive product para %s: %s", ref["ean_input"], exc)
                    break

                product_results = product_data.get("product_results", {}) or {}
                stores = product_results.get("stores", []) or []

                for store in stores:
                    store_link = store.get("link")
                    validation = validate_offer(ref, store.get("title") or candidate_title, store_link)
                    reviews = store.get("reviews")
                    rating = store.get("rating")
                    combined_score = float(validation.get("combined_score") or 0.0)
                    score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources, combined_score)
                    notes = list(validation["notes"])
                    if query_used:
                        notes.append(f"consulta usada: {query_used}")
                    row = {
                        "ean": ref["ean_input"],
                        "base_url": ref["base_url"],
                        "seller": infer_seller(store.get("name"), candidate_source, store_link),
                        "marketplace": marketplace_from_url(store_link, candidate_source),
                        "produto_google": candidate_title or "não visível",
                        "produto_loja": store.get("title") or candidate_title or "não visível",
                        "preco_atual": money_br(store.get("extracted_price") or parse_money(store.get("price"))),
                        "frete": money_br(store.get("shipping_extracted") if store.get("shipping_extracted") is not None else parse_money(store.get("shipping"))),
                        "preco_total_estimado": money_br(store.get("extracted_total") if store.get("extracted_total") is not None else (
                            (store.get("extracted_price") or parse_money(store.get("price")) or 0) +
                            (store.get("shipping_extracted") if store.get("shipping_extracted") is not None else parse_money(store.get("shipping")) or 0)
                        )),
                        "quantidade_vendida_ou_proxy": build_proxy(candidate_position, reviews, rating, candidate_multiple_sources),
                        "avaliacoes": reviews if reviews is not None else "não visível",
                        "nota": rating if rating is not None else "não visível",
                        "url_completo": store_link or candidate_product_link or "não visível",
                        "observacoes": " | ".join(notes) if notes else "sem observações",
                        "alta_relevancia": relevance_label(score, reviews, candidate_position),
                        "relevancia_score": score,
                        "status_validacao": validation["status"],
                        "origem_google_posicao": candidate_position,
                    }
                    key = f'{row["ean"]}|{row["url_completo"]}|{row["seller"]}'
                    if key not in seen:
                        seen.add(key)
                        rows.append(row)
                        product_rows_created = True

                next_page_token = product_results.get("stores_next_page_token")
                pages_collected += 1
                if not next_page_token:
                    break

        if not product_rows_created:
            validation = validate_offer(ref, candidate_title, candidate_product_link)
            reviews = result.get("reviews")
            rating = result.get("rating")
            combined_score = float(validation.get("combined_score") or 0.0)
            score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources, combined_score)
            notes = list(validation["notes"])
            if query_used:
                notes.append(f"consulta usada: {query_used}")
            row = {
                "ean": ref["ean_input"],
                "base_url": ref["base_url"],
                "seller": infer_seller(None, candidate_source, candidate_product_link),
                "marketplace": marketplace_from_url(candidate_product_link, candidate_source),
                "produto_google": candidate_title or "não visível",
                "produto_loja": candidate_title or "não visível",
                "preco_atual": money_br(candidate_price_num),
                "frete": result.get("delivery") or "não visível",
                "preco_total_estimado": money_br(candidate_total_num),
                "quantidade_vendida_ou_proxy": build_proxy(candidate_position, reviews, rating, candidate_multiple_sources),
                "avaliacoes": reviews if reviews is not None else "não visível",
                "nota": rating if rating is not None else "não visível",
                "url_completo": candidate_product_link or "não visível",
                "observacoes": " | ".join(notes) if notes else "sem observações",
                "alta_relevancia": relevance_label(score, reviews, candidate_position),
                "relevancia_score": score,
                "status_validacao": validation["status"],
                "origem_google_posicao": candidate_position,
            }
            key = f'{row["ean"]}|{row["url_completo"]}|{row["seller"]}'
            if key not in seen:
                seen.add(key)
                rows.append(row)

    return rows


def infer_ml_search_query(ref: Dict[str, Any]) -> str:
    title = ref["base_title"] if ref["base_title"] != "não visível" else ""
    brand = ref["brand"] if ref["brand"] != "não visível" else ""
    model = ref["model"] if ref["model"] != "não visível" else ""
    line = ref["line"] if ref["line"] != "não visível" else ""

    if title:
        return title
    return " ".join([x for x in [brand, line, model, ref["ean_input"]] if x]).strip() or ref["ean_input"]


def build_ml_top10(ref: Dict[str, Any], max_ml_results: int) -> List[Dict[str, Any]]:
    q = infer_ml_search_query(ref)
    data = requests_get_json(f"{MELI_API_BASE}/sites/{SITE_ID}/search", params={"q": q, "limit": max_ml_results})
    results = (data or {}).get("results") or []
    rows: List[Dict[str, Any]] = []

    for item in results:
        permalink = item.get("permalink")
        title = item.get("title")
        validation = validate_offer(ref, title, permalink)
        if validation["status"] not in {"EXATO", "PROVÁVEL"} or validation["kit_combo_detectado"]:
            continue

        shipping_cost = None
        shipping = item.get("shipping") or {}
        if isinstance(shipping, dict):
            if shipping.get("free_shipping") is True:
                shipping_cost = 0.0
            elif shipping.get("cost") is not None:
                shipping_cost = parse_money(shipping.get("cost"))

        price_num = item.get("price")
        total_num = price_num + shipping_cost if price_num is not None and shipping_cost is not None else price_num

        seller = item.get("seller") or {}
        seller_name = seller.get("nickname") or seller.get("id") or "não visível"
        sold_qty = item.get("sold_quantity")
        reviews = None
        rating = None

        available_qty = item.get("available_quantity")
        if isinstance(available_qty, (int, float)):
            sold_proxy = f"vendidos {sold_qty}" if sold_qty is not None else f"estoque ref. {available_qty}"
        else:
            sold_proxy = f"vendidos {sold_qty}" if sold_qty is not None else "não visível"

        score = relevance_score(None, reviews, rating, False, float(validation.get("combined_score") or 0.0))
        rows.append({
            "ean": ref["ean_input"],
            "ranking": 0,
            "seller": str(seller_name),
            "marketplace": "Mercado Livre",
            "preco_atual": money_br(parse_money(price_num)),
            "frete": money_br(shipping_cost),
            "preco_total_estimado": money_br(total_num),
            "quantidade_vendida_ou_proxy": sold_proxy,
            "avaliacoes": "não visível",
            "nota": "não visível",
            "url_completo": permalink or "não visível",
            "observacoes": " | ".join(validation["notes"]),
            "alta_relevancia": relevance_label(score, None, None),
            "status_validacao": validation["status"],
            "_score": score,
        })

    rows.sort(key=lambda x: (-x["_score"], parse_money(x["preco_total_estimado"]) or 10**9))
    for idx, row in enumerate(rows[:10], start=1):
        row["ranking"] = idx
        row.pop("_score", None)
    return rows[:10]


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    values = sorted(values)
    rank = (len(values) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] + (values[high] - values[low]) * frac


def summarize_entry(ref: Dict[str, Any], google_rows: List[Dict[str, Any]], ml_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [r for r in google_rows if r["status_validacao"] in {"EXATO", "PROVÁVEL"}]
    prices: List[float] = []
    for r in eligible:
        parsed = parse_money(r.get("preco_total_estimado"))
        if parsed is not None:
            prices.append(parsed)

    min_price = min(prices) if prices else None
    max_price = max(prices) if prices else None
    avg_price = statistics.mean(prices) if prices else None
    p25 = percentile(prices, 25)
    p45 = percentile(prices, 45)

    strongest_google = max(eligible, key=lambda x: x["relevancia_score"], default=None)
    strongest_ml = ml_rows[0] if ml_rows else None

    if strongest_google:
        strongest_seller = strongest_google["seller"]
        strongest_marketplace = strongest_google["marketplace"]
    elif strongest_ml:
        strongest_seller = strongest_ml["seller"]
        strongest_marketplace = "Mercado Livre"
    else:
        strongest_seller = "não visível"
        strongest_marketplace = "não visível"

    if not eligible:
        resumo = (
            f"Não foram encontrados anúncios confirmados/prováveis no Google Shopping para o GTIN {ref['ean_input']} "
            f"usando a URL base informada como referência."
        )
        conclusao = (
            "Sem base competitiva suficiente para sugerir faixa ideal. "
            "Valide a URL base, o GTIN e considere ampliar a consulta."
        )
        faixa_ideal = "não visível"
    else:
        resumo = (
            f"Foram encontrados {len(eligible)} anúncios equivalentes confirmados/prováveis para o GTIN {ref['ean_input']}. "
            f"O seller mais forte é {strongest_seller} em {strongest_marketplace}. "
            f"Menor preço total {money_br(min_price)}, maior {money_br(max_price)} e médio {money_br(avg_price)}."
        )
        faixa_ideal = f"{money_br(p25)} a {money_br(p45)}" if p25 is not None and p45 is not None else "não visível"
        conclusao = (
            f"Faixa de preço ideal sugerida: {faixa_ideal}. "
            f"Seller mais forte: {strongest_seller} ({strongest_marketplace}). "
            f"Top 10 Mercado Livre encontrado: {len(ml_rows)} anúncios validados."
        )

    return {
        "ean": ref["ean_input"],
        "base_url": ref["base_url"],
        "base_status": ref["base_status"],
        "base_title": ref["base_title"],
        "seller_mais_forte_proxy": strongest_seller,
        "marketplace_lider": strongest_marketplace,
        "menor_preco_total": money_br(min_price),
        "maior_preco_total": money_br(max_price),
        "preco_medio_total": money_br(avg_price),
        "faixa_preco_ideal": faixa_ideal,
        "qtd_anuncios_google_validos": len(eligible),
        "qtd_anuncios_mercado_livre_top10": len(ml_rows),
        "resumo_executivo": resumo,
        "conclusao_pratica": conclusao,
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest, x_app_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

    all_google_rows: List[Dict[str, Any]] = []
    all_ml_rows: List[Dict[str, Any]] = []
    all_base_refs: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    try:
        for entry in payload.entries:
            ref = fetch_base_reference(entry)
            all_base_refs.append({
                k: v for k, v in ref.items() if not k.startswith("_")
            })

            google_rows = build_google_rows(ref, payload.gl, payload.hl, payload.location, payload.max_products_per_entry)
            ml_rows = build_ml_top10(ref, payload.max_ml_results)

            all_google_rows.extend(google_rows)
            all_ml_rows.extend(ml_rows)
            summaries.append(summarize_entry(ref, google_rows, ml_rows))
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except Exception as exc:
        logger.exception("Erro interno no worker")
        raise HTTPException(status_code=500, detail=f"Erro interno: {exc}")

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "location": payload.location,
            "gl": payload.gl,
            "hl": payload.hl,
            "mode": "GTIN/EAN + URL base Mercado Livre",
        },
        "base_references": all_base_refs,
        "summary": summaries,
        "results": all_google_rows,
        "mercado_livre_top10": all_ml_rows,
    }
