
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
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "").strip()
APP_TOKEN = os.getenv("APP_TOKEN", "").strip()
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "35"))

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
STOPWORDS = {
    "de", "da", "do", "das", "dos", "para", "com", "sem", "e", "a", "o", "as", "os",
    "um", "uma", "the", "with", "by", "in", "on", "new", "novo", "nova"
}

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Google Shopping GTIN Analyzer", version="3.0.0")


class ExternalAPIError(Exception):
    pass


class EntryInput(BaseModel):
    ean: str = Field(..., description="GTIN/EAN do produto")
    base_url: str = Field(..., description="URL de referência do produto (qualquer loja)")

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
            raise ValueError("A URL precisa começar com http:// ou https://")
        return value


class AnalyzeRequest(BaseModel):
    entries: List[EntryInput]
    gl: str = Field(default="br")
    hl: str = Field(default="pt-BR")
    location: str = Field(default="Brazil")
    max_products_per_entry: int = Field(default=6, ge=1, le=10)

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


# ── Helpers ──────────────────────────────────────────────────────────────────

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
    if value == 0.0:
        return "Grátis"
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


# ── Fetchers ──────────────────────────────────────────────────────────────────

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


# ── GTIN extraction ───────────────────────────────────────────────────────────

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


# ── Base reference (sem API Mercado Livre) ────────────────────────────────────

def fetch_base_reference(entry: EntryInput) -> Dict[str, Any]:
    ean = entry.ean
    url = entry.base_url
    notes: List[str] = []

    html = requests_get_text(url)
    html_gtins = extract_gtins_from_html(html or "") if html else set()

    title = None
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Tenta og:title primeiro, depois title normal
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            title = og["content"].strip()
        elif soup.title:
            title = soup.title.get_text(" ", strip=True)

    gtins_found = sorted(html_gtins)

    if gtins_found:
        if ean in gtins_found:
            notes.append("GTIN/EAN confirmado na URL de referência")
            base_status = "BASE CONFIRMADA"
        else:
            notes.append(f"GTIN/EAN divergente na URL de referência: encontrados {', '.join(gtins_found[:5])}")
            base_status = "BASE COM DIVERGÊNCIA"
    else:
        notes.append("GTIN/EAN não visível na URL de referência")
        base_status = "BASE SEM GTIN VISÍVEL"

    if looks_like_kit_or_combo(title):
        notes.append("URL de referência parece kit/combo; atenção na validação")

    return {
        "ean_input": ean,
        "base_url": url,
        "base_title": title or "não visível",
        "base_status": base_status,
        "base_notes": " | ".join(notes),
        "_gtins_set": set(gtins_found),
    }


# ── Validação e relevância ────────────────────────────────────────────────────

def validate_offer(ref: Dict[str, Any], title: Optional[str], url: Optional[str]) -> Dict[str, Any]:
    notes: List[str] = []
    if looks_like_kit_or_combo(title):
        return {"status": "EXCLUÍDO - KIT/COMBO", "kit_combo_detectado": True, "notes": ["kit/combo detectado no título"], "combined_score": 0.0}

    gtins: Set[str] = set()
    if url and url != "não visível":
        html = requests_get_text(url)
        if html:
            gtins = extract_gtins_from_html(html)

    ref_gtins = set(ref.get("_gtins_set") or [])
    if ref["ean_input"]:
        ref_gtins.add(ref["ean_input"])

    similarity = token_similarity(title, ref.get("base_title"))
    combined = round(min(1.0, similarity), 4)

    if gtins:
        if ref["ean_input"] in gtins:
            notes.append("GTIN exato confirmado na página")
            return {"status": "EXATO", "kit_combo_detectado": False, "notes": notes, "combined_score": 1.0}
        notes.append(f"GTIN divergente na página: {', '.join(sorted(gtins)[:3])}")
        return {"status": "DIVERGENTE", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}

    if combined >= 0.72:
        notes.append(f"Validação por título ({combined:.2f})")
        return {"status": "PROVÁVEL", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}
    if combined >= 0.45:
        notes.append(f"Validação parcial ({combined:.2f})")
        return {"status": "NÃO CONFIRMADO", "kit_combo_detectado": False, "notes": notes, "combined_score": combined}

    return {"status": "DIVERGENTE", "kit_combo_detectado": False, "notes": [f"baixa aderência ({combined:.2f})"], "combined_score": combined}


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
        return "ALTA"
    if score >= 35:
        return "MÉDIA"
    return "BAIXA"


def infer_seller(store_name: Optional[str], source: Optional[str], url: Optional[str]) -> str:
    if store_name and str(store_name).strip():
        return str(store_name).strip()
    if source and str(source).strip():
        return str(source).strip()
    return host_from_url(url) or "não visível"


# ── SerpApi ───────────────────────────────────────────────────────────────────

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


def build_search_queries(ref: Dict[str, Any]) -> List[str]:
    ean = ref["ean_input"]
    title = ref["base_title"] if ref["base_title"] != "não visível" else ""
    title_tokens = [t for t in tokenize(title) if len(t) > 2]
    short_title = " ".join(list(title_tokens)[:8])

    candidates = [
        f'"{ean}"' if not title else f'"{title}" "{ean}"',
        f'ean {ean} {short_title}'.strip(),
        ean,
    ]
    out: List[str] = []
    seen = set()
    for item in candidates:
        item = re.sub(r"\s+", " ", item).strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out or [ean]


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


# ── Build top 10 Google Shopping ─────────────────────────────────────────────

def build_google_top10(ref: Dict[str, Any], gl: str, hl: str, location: str, max_products_per_entry: int) -> List[Dict[str, Any]]:
    search_data = shopping_search(ref, gl, hl, location)
    shopping_results = search_data.get("shopping_results", []) or []

    all_rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    if not shopping_results:
        return []

    for search_rank, result in enumerate(shopping_results[:max_products_per_entry], start=1):
        candidate_title = result.get("title")
        candidate_position = result.get("position") or search_rank
        candidate_source = result.get("source")
        candidate_product_link = result.get("product_link")
        candidate_multiple_sources = bool(result.get("multiple_sources"))
        candidate_price_num = result.get("extracted_price") or parse_money(result.get("price"))
        candidate_delivery_raw = result.get("delivery")
        candidate_delivery_num = parse_money(candidate_delivery_raw)
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
                    if validation["kit_combo_detectado"]:
                        continue

                    reviews = store.get("reviews")
                    rating = store.get("rating")
                    combined_score = float(validation.get("combined_score") or 0.0)
                    score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources, combined_score)

                    price_num = store.get("extracted_price") or parse_money(store.get("price"))
                    ship_num = store.get("shipping_extracted")
                    if ship_num is None:
                        ship_num = parse_money(store.get("shipping"))
                    total_num = (price_num + ship_num) if price_num is not None and ship_num is not None else price_num

                    frete_gratis = (ship_num == 0.0)

                    row = {
                        "ean": ref["ean_input"],
                        "ranking": 0,
                        "seller": infer_seller(store.get("name"), candidate_source, store_link),
                        "marketplace": marketplace_from_url(store_link, candidate_source),
                        "produto": store.get("title") or candidate_title or "não visível",
                        "preco_produto": money_br(price_num),
                        "frete": money_br(ship_num),
                        "frete_gratis": "Sim" if frete_gratis else "Não",
                        "preco_total": money_br(total_num),
                        "avaliacoes": reviews if reviews is not None else "não visível",
                        "nota": rating if rating is not None else "não visível",
                        "link": store_link or candidate_product_link or "não visível",
                        "status_validacao": validation["status"],
                        "relevancia": relevance_label(score, reviews, candidate_position),
                        "_score": score,
                        "_price_num": total_num,
                    }
                    key = f'{row["ean"]}|{row["link"]}|{row["seller"]}'
                    if key not in seen:
                        seen.add(key)
                        all_rows.append(row)
                        product_rows_created = True

                next_page_token = product_results.get("stores_next_page_token")
                pages_collected += 1
                if not next_page_token:
                    break

        if not product_rows_created:
            validation = validate_offer(ref, candidate_title, candidate_product_link)
            if validation["kit_combo_detectado"]:
                continue

            reviews = result.get("reviews")
            rating = result.get("rating")
            combined_score = float(validation.get("combined_score") or 0.0)
            score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources, combined_score)
            frete_gratis = (candidate_delivery_num == 0.0)

            row = {
                "ean": ref["ean_input"],
                "ranking": 0,
                "seller": infer_seller(None, candidate_source, candidate_product_link),
                "marketplace": marketplace_from_url(candidate_product_link, candidate_source),
                "produto": candidate_title or "não visível",
                "preco_produto": money_br(candidate_price_num),
                "frete": money_br(candidate_delivery_num) if candidate_delivery_num is not None else (candidate_delivery_raw or "não visível"),
                "frete_gratis": "Sim" if frete_gratis else "Não",
                "preco_total": money_br(candidate_total_num),
                "avaliacoes": reviews if reviews is not None else "não visível",
                "nota": rating if rating is not None else "não visível",
                "link": candidate_product_link or "não visível",
                "status_validacao": validation["status"],
                "relevancia": relevance_label(score, reviews, candidate_position),
                "_score": score,
                "_price_num": candidate_total_num,
            }
            key = f'{row["ean"]}|{row["link"]}|{row["seller"]}'
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

    # Ordena por score desc, depois preço asc
    all_rows.sort(key=lambda x: (-x["_score"], x["_price_num"] or 10**9))

    top10 = all_rows[:10]
    for idx, row in enumerate(top10, start=1):
        row["ranking"] = idx
        row.pop("_score", None)
        row.pop("_price_num", None)

    return top10


# ── Resumo competitivo ────────────────────────────────────────────────────────

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


def summarize_entry(ref: Dict[str, Any], top10: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [r for r in top10 if r["status_validacao"] in {"EXATO", "PROVÁVEL"}]
    prices: List[float] = []
    for r in top10:
        # Tenta extrair o número do campo formatado
        raw = r.get("preco_total", "")
        if isinstance(raw, str):
            num = parse_money(raw.replace("R$", "").replace(".", "").replace(",", ".").strip())
        else:
            num = parse_money(raw)
        if num is not None:
            prices.append(num)

    min_price = min(prices) if prices else None
    max_price = max(prices) if prices else None
    avg_price = statistics.mean(prices) if prices else None
    p25 = percentile(prices, 25)
    p45 = percentile(prices, 45)

    com_frete_gratis = sum(1 for r in top10 if r.get("frete_gratis") == "Sim")
    lider = top10[0] if top10 else None

    faixa_ideal = f"{money_br(p25)} a {money_br(p45)}" if p25 is not None and p45 is not None else "não visível"

    return {
        "ean": ref["ean_input"],
        "base_url": ref["base_url"],
        "base_title": ref["base_title"],
        "base_status": ref["base_status"],
        "menor_preco": money_br(min_price),
        "preco_medio": money_br(avg_price),
        "maior_preco": money_br(max_price),
        "faixa_ideal_sugerida": faixa_ideal,
        "vendedores_com_frete_gratis": com_frete_gratis,
        "total_vendedores_top10": len(top10),
        "seller_lider": lider["seller"] if lider else "não visível",
        "marketplace_lider": lider["marketplace"] if lider else "não visível",
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest, x_app_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

    products: List[Dict[str, Any]] = []

    try:
        for entry in payload.entries:
            ref = fetch_base_reference(entry)
            top10 = build_google_top10(ref, payload.gl, payload.hl, payload.location, payload.max_products_per_entry)
            summary = summarize_entry(ref, top10)
            products.append({
                "ean": entry.ean,
                "summary": summary,
                "top10": top10,
            })
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
        },
        "products": products,
    }
