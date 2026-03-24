import json
import logging
import math
import os
import re
import statistics
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
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
    # ruído de títulos de loja
    "compre", "compra", "frete", "gratis", "entrega", "rapida", "rapido",
    "oferta", "promocao", "menor", "preco", "melhor",
}

# Apenas esses statuses entram no resultado final.
# PROVAVEL_TITULO é aceito somente quando o scraping não encontrou nenhum GTIN
# (página bloqueou bots, GTIN não publicado, etc.).
ACCEPTED_STATUSES = {"EXATO", "PROVAVEL_TITULO"}

logger = logging.getLogger("uvicorn.error")
app = FastAPI(title="Google Shopping GTIN Analyzer", version="4.2.0")


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
            raise ValueError("A URL precisa comecar com http:// ou https://")
        return value


class AnalyzeRequest(BaseModel):
    entries: List[EntryInput]
    gl: str = Field(default="br")
    hl: str = Field(default="pt-BR")
    location: str = Field(default="Brazil")
    max_products_per_entry: int = Field(default=8, ge=1, le=20)

    @field_validator("entries")
    @classmethod
    def validate_entries(cls, values: List[EntryInput]) -> List[EntryInput]:
        if not values:
            raise ValueError("Informe pelo menos 1 entrada.")
        if len(values) > 10:
            raise ValueError("Maximo de 10 entradas por execucao.")
        seen: set = set()
        unique = []
        for item in values:
            key = (item.ean, item.base_url)
            if key not in seen:
                unique.append(item)
                seen.add(key)
        return unique


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def clean_store_title(raw_title: Optional[str]) -> str:
    """
    Remove sufixos típicos de títulos de páginas de lojas que contaminam
    a comparação de similaridade.
    Ex: 'Produto X - Compre na Loja Y | Frete Grátis' → 'Produto X'
    """
    if not raw_title:
        return ""
    for sep in [" | ", " - ", " – ", " — "]:
        if sep in raw_title:
            raw_title = raw_title.split(sep)[0]
    return raw_title.strip()


def key_tokens(value: Optional[str]) -> Set[str]:
    """
    Extrai tokens discriminativos do produto: números (modelo, capacidade),
    siglas e palavras longas (marca, nome).
    Esses tokens têm peso maior na validação por título.
    """
    tokens = tokenize(value)
    key: Set[str] = set()
    for t in tokens:
        if re.search(r"\d", t):           # contém número → capacidade, modelo, versão
            key.add(t)
        elif len(t) <= 5 and t.upper() == normalize_text(t).upper():
            key.add(t)                    # sigla curta (USB, HDMI, etc.)
        elif len(t) > 4:                  # palavra longa → nome, marca
            key.add(t)
    return key


def title_match_score(candidate: Optional[str], reference: Optional[str]) -> Tuple[float, float]:
    """
    Retorna (jaccard_geral, key_overlap).

    jaccard_geral  : Jaccard sobre todos os tokens sem stopwords.
    key_overlap    : proporção dos tokens-chave da referência presentes no candidato.
                     É mais discriminativo — detecta diferenças de modelo/capacidade.
    """
    cand_clean = clean_store_title(candidate)
    ref_clean  = clean_store_title(reference)

    sc = tokenize(cand_clean)
    sr = tokenize(ref_clean)
    jaccard = len(sc & sr) / len(sc | sr) if (sc and sr) else 0.0

    kr = key_tokens(ref_clean)
    kc = key_tokens(cand_clean)
    key_overlap = len(kr & kc) / len(kr) if kr else 0.0

    return round(jaccard, 4), round(key_overlap, 4)


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
        return "nao visivel"
    if value == 0.0:
        return "Gratis"
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
    if "submarino" in host or "submarino" in source_norm:
        return "Submarino"
    if "extra.com" in host or "extra" in source_norm:
        return "Extra"
    if "kabum" in host or "kabum" in source_norm:
        return "KaBuM"
    if source and str(source).strip():
        return str(source).strip()
    return host or "nao visivel"


def infer_seller(store_name: Optional[str], source: Optional[str], url: Optional[str]) -> str:
    if store_name and str(store_name).strip():
        return str(store_name).strip()
    if source and str(source).strip():
        return str(source).strip()
    host = host_from_url(url)
    if host:
        name = re.sub(r"^www\.", "", host)
        name = re.sub(r"\.(com\.br|com|br|net|org)$", "", name)
        return name.capitalize() if name else "nao visivel"
    return "nao visivel"


# ── Fetchers ──────────────────────────────────────────────────────────────────

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

    # JSON-LD estruturado
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
            collect_gtins_from_json(data, found)
        except Exception:
            for match in re.finditer(
                r'"(?:gtin|gtin8|gtin12|gtin13|gtin14|ean)"\s*:\s*"(\d{8,14})"', raw, flags=re.I
            ):
                found.add(match.group(1))

    # Microdados / itemprop
    for tag in soup.select("[itemprop]"):
        itemprop = (tag.get("itemprop") or "").strip().lower()
        if itemprop in GTIN_KEYS:
            value = tag.get("content") or tag.get_text(" ", strip=True)
            d = digits_only(value)
            if d:
                found.add(d)

    # Regex genérico no HTML bruto
    for match in re.finditer(
        r'(?:gtin|gtin8|gtin12|gtin13|gtin14|ean)[^0-9]{0,25}(\d{8,14})', html, flags=re.I
    ):
        found.add(match.group(1))

    # Atributos data-ean / data-gtin (comum em lojas BR)
    for tag in soup.find_all(True):
        for attr, val in tag.attrs.items():
            if "ean" in attr.lower() or "gtin" in attr.lower():
                d = digits_only(str(val))
                if d and len(d) in (8, 12, 13, 14):
                    found.add(d)

    return found


# ── Base reference ────────────────────────────────────────────────────────────

def fetch_base_reference(entry: EntryInput) -> Dict[str, Any]:
    ean = entry.ean
    url = entry.base_url
    notes: List[str] = []

    html = requests_get_text(url)
    html_gtins = extract_gtins_from_html(html or "") if html else set()

    raw_title = None
    if html:
        soup = BeautifulSoup(html, "html.parser")
        # Prioridade: og:title → h1 → <title>
        og = soup.find("meta", property="og:title")
        if og and og.get("content"):
            raw_title = og["content"].strip()
        else:
            h1 = soup.find("h1")
            if h1:
                raw_title = h1.get_text(" ", strip=True)
            elif soup.title:
                raw_title = soup.title.get_text(" ", strip=True)

    # Remove ruído de sufixo de loja do título
    title = clean_store_title(raw_title) if raw_title else None

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

    return {
        "ean_input": ean,
        "base_url": url,
        "base_title": title or "nao visivel",
        "base_status": base_status,
        "base_notes": " | ".join(notes),
        "_gtins_set": set(gtins_found),
    }


# ── Validação de oferta ───────────────────────────────────────────────────────

def validate_offer(ref: Dict[str, Any], title: Optional[str], url: Optional[str]) -> Dict[str, Any]:
    """
    Hierarquia de decisão:

    1. Kit/combo no título                       → EXCLUIDO         ❌ descartado
    2. GTIN exato confirmado via scraping        → EXATO            ✅ aceito
    3. GTIN diferente encontrado via scraping    → DIVERGENTE       ❌ descartado
    4. Scraping sem resultado (bloqueado/sem GTIN publicado):
       a. key_overlap ≥ 0.60 E jaccard ≥ 0.35   → PROVAVEL_TITULO  ✅ aceito
       b. Qualquer outro caso                    → NAO_CONFIRMADO   ❌ descartado

    Lógica do critério 4a:
    - key_overlap mede presença de tokens discriminativos (número de modelo, capacidade).
      Ex: "Galaxy A55 128GB" vs "Galaxy A55 256GB" → key_overlap baixo (128 ≠ 256) → REJEITADO.
    - Exige AMBOS os critérios para minimizar falsos positivos.
    - Ser mais permissivo aqui traria de volta o problema original de produtos errados.
    """
    notes: List[str] = []

    # 1. Kit/combo
    if looks_like_kit_or_combo(title):
        return {
            "status": "EXCLUIDO - KIT/COMBO",
            "kit_combo_detectado": True,
            "notes": ["kit/combo detectado no titulo"],
            "combined_score": 0.0,
        }

    # 2 & 3. Tenta confirmar via GTIN na página do candidato
    gtins: Set[str] = set()
    scraping_attempted = False
    if url and url != "nao visivel":
        scraping_attempted = True
        html = requests_get_text(url)
        if html:
            gtins = extract_gtins_from_html(html)

    if gtins:
        if ref["ean_input"] in gtins:
            notes.append("GTIN exato confirmado na pagina")
            return {
                "status": "EXATO",
                "kit_combo_detectado": False,
                "notes": notes,
                "combined_score": 1.0,
            }
        # GTIN presente mas diferente → produto claramente errado
        notes.append(f"GTIN divergente na pagina: {', '.join(sorted(gtins)[:3])}")
        return {
            "status": "DIVERGENTE",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": 0.0,
        }

    # 4. Sem GTIN encontrado — valida por título
    jaccard, key_overlap = title_match_score(title, ref.get("base_title"))
    scraping_note = "scraping sem GTIN" if scraping_attempted else "sem URL para scraping"
    notes.append(f"{scraping_note} — jaccard={jaccard:.2f} key_overlap={key_overlap:.2f}")

    if key_overlap >= 0.60 and jaccard >= 0.35:
        combined = round(key_overlap * 0.7 + jaccard * 0.3, 4)
        return {
            "status": "PROVAVEL_TITULO",
            "kit_combo_detectado": False,
            "notes": notes,
            "combined_score": combined,
        }

    return {
        "status": "NAO_CONFIRMADO",
        "kit_combo_detectado": False,
        "notes": notes,
        "combined_score": 0.0,
    }


def relevance_score(
    position: Optional[int],
    reviews: Optional[int],
    rating: Optional[float],
    multiple_sources: bool,
    validation_boost: float = 0.0,
) -> float:
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
        return "MEDIA"
    return "BAIXA"


# ── SerpApi ───────────────────────────────────────────────────────────────────

def serpapi_request(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERPAPI_KEY:
        raise ExternalAPIError("SERPAPI_KEY nao configurada no Render.")
    payload = dict(params)
    payload["api_key"] = SERPAPI_KEY
    try:
        response = requests.get(SERPAPI_ENDPOINT, params=payload, headers=HEADERS, timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        raise ExternalAPIError(f"Falha de conexao com a SerpApi: {exc}") from exc

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
        raise ExternalAPIError(f"Resposta invalida da SerpApi: {preview}") from exc

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
    """
    Estratégia de busca em 3 camadas — da mais precisa à mais ampla:

    1. Título limpo + EAN  → melhor cobertura: Google Shopping BR indexa bem por título
    2. Só o EAN            → fallback quando título não retorna resultados
    3. Só o título         → último recurso; produtos validados depois pelo GTIN via scraping

    NÃO usamos EAN entre aspas duplas como query primária porque o Google Shopping BR
    frequentemente retorna zero resultados para EAN isolado com aspas.
    """
    ean = ref["ean_input"]
    title = ref["base_title"] if ref["base_title"] != "nao visivel" else ""
    title_tokens = [t for t in tokenize(title) if len(t) > 2]
    short_title = " ".join(list(title_tokens)[:7])

    candidates: List[Optional[str]] = []

    if short_title:
        candidates.append(f"{short_title} {ean}")   # primária: título + EAN
        candidates.append(ean)                       # fallback 1: só EAN
        candidates.append(short_title)               # fallback 2: só título
    else:
        candidates.append(ean)

    out: List[str] = []
    seen: set = set()
    for item in candidates:
        if not item:
            continue
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
                logger.info("EAN %s — query usada: %s", ref["ean_input"], query)
                return data
            last_error = "Google Shopping nao retornou resultados."
        except ExternalAPIError as exc:
            last_error = str(exc)
            if is_no_results_error(last_error):
                continue
            raise
    return {"shopping_results": [], "_query_used": None, "_notes": [last_error or "Sem resultados."]}


def immersive_product(
    page_token: str, gl: str, hl: str, location: str, next_page_token: Optional[str] = None
) -> Dict[str, Any]:
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

def build_google_top10(
    ref: Dict[str, Any], gl: str, hl: str, location: str, max_products_per_entry: int
) -> List[Dict[str, Any]]:
    search_data = shopping_search(ref, gl, hl, location)
    shopping_results = search_data.get("shopping_results", []) or []

    all_rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    if not shopping_results:
        return []

    for search_rank, result in enumerate(shopping_results[:max_products_per_entry], start=1):
        candidate_title    = result.get("title")
        candidate_position = result.get("position") or search_rank
        candidate_source   = result.get("source")
        candidate_product_link     = result.get("product_link")
        candidate_multiple_sources = bool(result.get("multiple_sources"))
        candidate_price_num    = result.get("extracted_price") or parse_money(result.get("price"))
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
                    product_data = immersive_product(
                        page_token=page_token, gl=gl, hl=hl, location=location,
                        next_page_token=next_page_token,
                    )
                except ExternalAPIError as exc:
                    logger.warning("Falha no immersive product para %s: %s", ref["ean_input"], exc)
                    break

                product_results = product_data.get("product_results", {}) or {}
                stores = product_results.get("stores", []) or []

                for store in stores:
                    store_link  = store.get("link")
                    store_title = store.get("title") or candidate_title
                    validation  = validate_offer(ref, store_title, store_link)

                    if validation["kit_combo_detectado"]:
                        continue
                    if validation["status"] not in ACCEPTED_STATUSES:
                        logger.debug(
                            "Descartado store status=%s ean=%s titulo=%s",
                            validation["status"], ref["ean_input"], store_title,
                        )
                        continue

                    reviews = store.get("reviews")
                    rating  = store.get("rating")
                    combined_score = float(validation.get("combined_score") or 0.0)
                    score = relevance_score(
                        candidate_position, reviews, rating, candidate_multiple_sources, combined_score
                    )

                    price_num = store.get("extracted_price") or parse_money(store.get("price"))
                    ship_num  = store.get("shipping_extracted")
                    if ship_num is None:
                        ship_num = parse_money(store.get("shipping"))
                    total_num = (
                        (price_num + ship_num)
                        if price_num is not None and ship_num is not None
                        else price_num
                    )
                    frete_gratis = ship_num == 0.0

                    row = {
                        "ean": ref["ean_input"],
                        "ranking": 0,
                        "seller": infer_seller(store.get("name"), candidate_source, store_link),
                        "marketplace": marketplace_from_url(store_link, candidate_source),
                        "produto": store_title or "nao visivel",
                        "preco_produto": money_br(price_num),
                        "frete_gratis": "Sim" if frete_gratis else "Nao",
                        "preco_total": money_br(total_num),
                        "link": store_link or candidate_product_link or "nao visivel",
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

        # Fallback: sem immersive, ou todas as stores do immersive foram descartadas
        if not product_rows_created:
            validation = validate_offer(ref, candidate_title, candidate_product_link)

            if validation["kit_combo_detectado"]:
                continue
            if validation["status"] not in ACCEPTED_STATUSES:
                logger.debug(
                    "Descartado fallback status=%s ean=%s titulo=%s",
                    validation["status"], ref["ean_input"], candidate_title,
                )
                continue

            reviews = result.get("reviews")
            rating  = result.get("rating")
            combined_score = float(validation.get("combined_score") or 0.0)
            score = relevance_score(
                candidate_position, reviews, rating, candidate_multiple_sources, combined_score
            )
            frete_gratis = candidate_delivery_num == 0.0

            row = {
                "ean": ref["ean_input"],
                "ranking": 0,
                "seller": infer_seller(None, candidate_source, candidate_product_link),
                "marketplace": marketplace_from_url(candidate_product_link, candidate_source),
                "produto": candidate_title or "nao visivel",
                "preco_produto": money_br(candidate_price_num),
                "frete_gratis": "Sim" if frete_gratis else "Nao",
                "preco_total": money_br(candidate_total_num),
                "link": candidate_product_link or "nao visivel",
                "status_validacao": validation["status"],
                "relevancia": relevance_label(score, reviews, candidate_position),
                "_score": score,
                "_price_num": candidate_total_num,
            }
            key = f'{row["ean"]}|{row["link"]}|{row["seller"]}'
            if key not in seen:
                seen.add(key)
                all_rows.append(row)

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
    low  = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] + (values[high] - values[low]) * frac


def summarize_entry(ref: Dict[str, Any], top10: List[Dict[str, Any]]) -> Dict[str, Any]:
    prices: List[float] = []
    for r in top10:
        raw = r.get("preco_total", "")
        if isinstance(raw, str):
            num = parse_money(raw.replace("R$", "").replace(".", "").replace(",", ".").strip())
        else:
            num = parse_money(raw)
        if num is not None and num > 0:
            prices.append(num)

    min_price = min(prices) if prices else None
    max_price = max(prices) if prices else None
    avg_price = statistics.mean(prices) if prices else None
    p25 = percentile(prices, 25)
    p45 = percentile(prices, 45)

    com_frete_gratis = sum(1 for r in top10 if r.get("frete_gratis") == "Sim")
    lider = top10[0] if top10 else None
    faixa_ideal = (
        f"{money_br(p25)} a {money_br(p45)}"
        if p25 is not None and p45 is not None
        else "nao visivel"
    )

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
        "seller_lider": lider["seller"] if lider else "nao visivel",
        "marketplace_lider": lider["marketplace"] if lider else "nao visivel",
    }


# ── Processamento de uma entrada ──────────────────────────────────────────────

def process_entry(
    entry: EntryInput, gl: str, hl: str, location: str, max_products_per_entry: int
) -> Dict[str, Any]:
    ref  = fetch_base_reference(entry)
    top10 = build_google_top10(ref, gl, hl, location, max_products_per_entry)
    summary = summarize_entry(ref, top10)
    return {
        "ean": entry.ean,
        "summary": summary,
        "top10": top10,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(
    payload: AnalyzeRequest, x_app_token: Optional[str] = Header(default=None)
) -> Dict[str, Any]:
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Token invalido.")

    n = len(payload.entries)
    max_workers = min(n, 5)

    results: Dict[int, Dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                process_entry,
                entry,
                payload.gl,
                payload.hl,
                payload.location,
                payload.max_products_per_entry,
            ): idx
            for idx, entry in enumerate(payload.entries)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except ExternalAPIError as exc:
                results[idx] = {
                    "ean": payload.entries[idx].ean,
                    "summary": {},
                    "top10": [],
                    "error": str(exc),
                }
            except Exception as exc:
                logger.exception("Erro ao processar EAN %s", payload.entries[idx].ean)
                results[idx] = {
                    "ean": payload.entries[idx].ean,
                    "summary": {},
                    "top10": [],
                    "error": f"erro interno: {exc}",
                }

    products = [results[i] for i in range(n)]

    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "location": payload.location,
            "gl": payload.gl,
            "hl": payload.hl,
        },
        "products": products,
    }
