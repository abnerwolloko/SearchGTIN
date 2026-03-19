import json
import logging
import math
import os
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

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
    r"\bkit\b",
    r"\bcombo\b",
    r"\bconjunto\b",
    r"\bbundle\b",
    r"\bpack\b",
    r"\bduo\b",
    r"\bdupla\b",
    r"\bpar\b",
    r"\bc\/2\b",
    r"\bc\/3\b",
    r"\b2x\b",
    r"\b3x\b",
    r"\b2 unidades\b",
    r"\b3 unidades\b",
    r"\b2 un\b",
    r"\b3 un\b",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-shopping-gtin")

app = FastAPI(title="Google Shopping GTIN Analyzer", version="1.0.3")


class ExternalAPIError(Exception):
    pass


def is_no_results_error(message: str) -> bool:
    msg = (message or "").lower()
    return (
        "google hasn't returned any results for this query" in msg
        or "hasn't returned any results" in msg
        or "no results" in msg
    )


class AnalyzeRequest(BaseModel):
    eans: List[str] = Field(..., description="Lista de 1 a 10 GTINs/EANs")
    gl: str = Field(default="br")
    hl: str = Field(default="pt-BR")
    location: str = Field(default="Brazil")
    max_products_per_ean: int = Field(default=4, ge=1, le=8)

    @field_validator("eans")
    @classmethod
    def validate_eans(cls, values: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen: Set[str] = set()
        for value in values:
            digits = digits_only(str(value))
            if not digits:
                continue
            if len(digits) not in (8, 12, 13, 14):
                continue
            if digits not in seen:
                cleaned.append(digits)
                seen.add(digits)

        if not cleaned:
            raise ValueError("Informe pelo menos 1 GTIN/EAN válido.")
        if len(cleaned) > 10:
            raise ValueError("Máximo de 10 GTINs por execução.")
        return cleaned


def digits_only(value: str) -> str:
    return re.sub(r"\D+", "", value or "")


def normalize_text(value: Optional[str]) -> str:
    value = value or ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def looks_like_kit_or_combo(text: Optional[str]) -> bool:
    txt = normalize_text(text)
    return any(re.search(pattern, txt) for pattern in KIT_PATTERNS)


def token_similarity(a: Optional[str], b: Optional[str]) -> float:
    sa = {t for t in normalize_text(a).split() if len(t) > 2}
    sb = {t for t in normalize_text(b).split() if len(t) > 2}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


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
        if cleaned.count(",") == 1:
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
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


def infer_marketplace(source: Optional[str], url: Optional[str]) -> str:
    source_norm = normalize_text(source)
    host = host_from_url(url)

    if "mercadolivre" in host or "mercado livre" in source_norm:
        return "Mercado Livre"
    if "amazon" in host or "amazon" in source_norm:
        return "Amazon"
    if "shopee" in host or "shopee" in source_norm:
        return "Shopee"
    if "magazineluiza" in host or "magalu" in host or "magalu" in source_norm:
        return "Magalu"
    if "americanas" in host or "americanas" in source_norm:
        return "Americanas"
    if "casasbahia" in host or "casas bahia" in host or "casasbahia" in source_norm:
        return "Casas Bahia"
    if "carrefour" in host or "carrefour" in source_norm:
        return "Carrefour"
    if "submarino" in host or "submarino" in source_norm:
        return "Submarino"

    if source:
        return source.strip()
    return host or "não visível"


def infer_seller(store_name: Optional[str], source: Optional[str], url: Optional[str]) -> str:
    if store_name and str(store_name).strip():
        return str(store_name).strip()
    if source and str(source).strip():
        return str(source).strip()
    host = host_from_url(url)
    return host or "não visível"


def serpapi_request(params: Dict[str, Any]) -> Dict[str, Any]:
    if not SERPAPI_KEY:
        raise ExternalAPIError("SERPAPI_KEY não configurada no Render.")

    payload = dict(params)
    payload["api_key"] = SERPAPI_KEY

    try:
        response = requests.get(
            SERPAPI_ENDPOINT,
            params=payload,
            headers=HEADERS,
            timeout=HTTP_TIMEOUT,
        )
    except requests.RequestException as exc:
        raise ExternalAPIError(f"Falha de conexão com a SerpApi: {exc}") from exc

    try:
        data = response.json()
    except ValueError:
        data = None

    if response.status_code >= 400:
        message = None
        if isinstance(data, dict):
            message = data.get("error") or data.get("message")
        raise ExternalAPIError(f"SerpApi HTTP {response.status_code}: {message or response.text[:300]}")

    if isinstance(data, dict) and data.get("error"):
        raise ExternalAPIError(f"SerpApi retornou erro: {data['error']}")

    if not isinstance(data, dict):
        raise ExternalAPIError("SerpApi retornou resposta inválida.")

    return data


def shopping_search(gtin: str, gl: str, hl: str, location: str) -> Dict[str, Any]:
    tried_queries = [
        gtin,
        f'"{gtin}"',
        f"gtin {gtin}",
        f"ean {gtin}",
    ]

    last_error: Optional[str] = None
    for query in tried_queries:
        try:
            data = serpapi_request(
                {
                    "engine": "google_shopping",
                    "q": query,
                    "gl": gl,
                    "hl": hl,
                    "location": location,
                    "no_cache": "true",
                }
            )
            data["_query_used"] = query
            return data
        except ExternalAPIError as exc:
            last_error = str(exc)
            if is_no_results_error(last_error):
                logger.info("Sem resultados no Google Shopping para consulta %s", query)
                continue
            raise

    return {
        "shopping_results": [],
        "_query_used": None,
        "_notes": [last_error or "Google Shopping não retornou resultados para este GTIN/EAN."],
    }


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


def fetch_page_gtins(url: str) -> Set[str]:
    found: Set[str] = set()
    if not url or url == "não visível":
        return found

    try:
        response = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT, allow_redirects=True)
        response.raise_for_status()
    except Exception:
        return found

    html = response.text or ""
    soup = BeautifulSoup(html, "html.parser")

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

    for match in re.finditer(r'(?:gtin|gtin8|gtin12|gtin13|gtin14|ean)[^0-9]{0,20}(\d{8,14})', html, flags=re.I):
        found.add(match.group(1))

    return found


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


def relevance_score(position: Optional[int], reviews: Optional[int], rating: Optional[float], multiple_sources: bool) -> float:
    score = 0.0
    if position is not None:
        score += max(0.0, 40.0 - (position * 3.0))
    if reviews is not None:
        score += min(30.0, math.log10(reviews + 1) * 10.0)
    if rating is not None:
        score += rating * 6.0
    if multiple_sources:
        score += 8.0
    return round(score, 2)


def relevance_label(score: float, reviews: Optional[int], position: Optional[int]) -> str:
    if score >= 55 or (position is not None and position <= 3) or (reviews is not None and reviews >= 100):
        return "ALTA RELEVÂNCIA"
    if score >= 35:
        return "MÉDIA RELEVÂNCIA"
    return "BAIXA RELEVÂNCIA"


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


def validate_store_offer(
    ean: str,
    store_link: Optional[str],
    store_title: Optional[str],
    reference_title: Optional[str],
) -> Dict[str, Any]:
    notes: List[str] = []

    if looks_like_kit_or_combo(store_title):
        return {
            "status": "EXCLUÍDO - KIT/COMBO",
            "kit_combo_detectado": True,
            "notes": ["kit/combo detectado no título"],
        }

    gtins = fetch_page_gtins(store_link or "")
    if gtins:
        if ean in gtins:
            notes.append("GTIN exato confirmado na página")
            return {
                "status": "EXATO",
                "kit_combo_detectado": False,
                "notes": notes,
            }
        notes.append(f"GTINs encontrados na página: {', '.join(sorted(gtins)[:5])}")
        return {
            "status": "DIVERGENTE",
            "kit_combo_detectado": False,
            "notes": notes,
        }

    sim = token_similarity(store_title, reference_title)
    if sim >= 0.6:
        notes.append("GTIN não visível; validação por similaridade de título")
        return {
            "status": "PROVÁVEL",
            "kit_combo_detectado": False,
            "notes": notes,
        }

    return {
        "status": "NÃO CONFIRMADO",
        "kit_combo_detectado": False,
        "notes": ["GTIN não visível na página"],
    }


def build_rows_for_ean(ean: str, gl: str, hl: str, location: str, max_products_per_ean: int) -> List[Dict[str, Any]]:
    search_data = shopping_search(ean, gl, hl, location)
    shopping_results = search_data.get("shopping_results", []) or []
    query_used = search_data.get("_query_used")
    search_notes = search_data.get("_notes", []) or []

    rows: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()

    if not shopping_results:
        note = " | ".join(search_notes) if search_notes else "Google Shopping não retornou resultados para este GTIN/EAN."
        return [
            {
                "ean": ean,
                "seller": "não visível",
                "marketplace": "não visível",
                "produto_google": "não visível",
                "produto_loja": "não visível",
                "preco_atual": "não visível",
                "frete": "não visível",
                "preco_total_estimado": "não visível",
                "preco_atual_num": None,
                "frete_num": None,
                "preco_total_estimado_num": None,
                "quantidade_vendida_ou_proxy": "não visível",
                "avaliacoes": "não visível",
                "nota": "não visível",
                "url_completo": "não visível",
                "observacoes": f"SEM RESULTADOS | consulta usada: {query_used or 'nenhuma'} | {note}",
                "alta_relevancia": "não visível",
                "relevancia_score": 0.0,
                "status_validacao": "SEM RESULTADOS",
                "kit_combo_detectado": False,
                "origem_google_posicao": None,
            }
        ]

    for search_rank, result in enumerate(shopping_results[:max_products_per_ean], start=1):
        candidate_title = result.get("title")
        candidate_position = result.get("position")
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
                    product_data = immersive_product(
                        page_token=page_token,
                        gl=gl,
                        hl=hl,
                        location=location,
                        next_page_token=next_page_token,
                    )
                except ExternalAPIError as exc:
                    logger.warning("Falha no google_immersive_product para %s: %s", ean, exc)
                    break

                product_results = product_data.get("product_results", {}) or {}
                stores = product_results.get("stores", []) or []

                for store in stores:
                    store_link = store.get("link")
                    seller = infer_seller(store.get("name"), candidate_source, store_link)
                    marketplace = infer_marketplace(candidate_source, store_link)
                    price_num = store.get("extracted_price") or parse_money(store.get("price"))
                    shipping_num = store.get("shipping_extracted")
                    if shipping_num is None:
                        shipping_num = parse_money(store.get("shipping"))
                    total_num = store.get("extracted_total")
                    if total_num is None:
                        if price_num is not None and shipping_num is not None:
                            total_num = price_num + shipping_num
                        else:
                            total_num = price_num

                    validation = validate_store_offer(
                        ean=ean,
                        store_link=store_link,
                        store_title=store.get("title") or candidate_title,
                        reference_title=product_results.get("title") or candidate_title,
                    )

                    reviews = store.get("reviews")
                    rating = store.get("rating")
                    score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources)

                    notes = list(validation["notes"])
                    if query_used:
                        notes.append(f"consulta usada: {query_used}")

                    row = {
                        "ean": ean,
                        "seller": seller,
                        "marketplace": marketplace,
                        "produto_google": candidate_title or "não visível",
                        "produto_loja": store.get("title") or candidate_title or "não visível",
                        "preco_atual": money_br(price_num),
                        "frete": money_br(shipping_num),
                        "preco_total_estimado": money_br(total_num),
                        "preco_atual_num": price_num,
                        "frete_num": shipping_num,
                        "preco_total_estimado_num": total_num,
                        "quantidade_vendida_ou_proxy": build_proxy(candidate_position, reviews, rating, candidate_multiple_sources),
                        "avaliacoes": reviews if reviews is not None else "não visível",
                        "nota": rating if rating is not None else "não visível",
                        "url_completo": store_link or candidate_product_link or "não visível",
                        "observacoes": " | ".join(notes) or "sem observações",
                        "alta_relevancia": relevance_label(score, reviews, candidate_position),
                        "relevancia_score": score,
                        "status_validacao": validation["status"],
                        "kit_combo_detectado": validation["kit_combo_detectado"],
                        "origem_google_posicao": candidate_position if candidate_position is not None else search_rank,
                    }

                    dedupe_key = f"{ean}|{row['url_completo']}|{row['seller']}|{row['marketplace']}"
                    if dedupe_key not in seen_keys:
                        seen_keys.add(dedupe_key)
                        rows.append(row)
                        product_rows_created = True

                next_page_token = product_results.get("stores_next_page_token")
                pages_collected += 1
                if not next_page_token:
                    break

        if not product_rows_created:
            seller = infer_seller(None, candidate_source, candidate_product_link)
            marketplace = infer_marketplace(candidate_source, candidate_product_link)
            status = "NÃO CONFIRMADO"
            notes = ["sem detalhamento de stores; mantido como resultado de pesquisa"]
            if query_used:
                notes.append(f"consulta usada: {query_used}")
            if looks_like_kit_or_combo(candidate_title):
                status = "EXCLUÍDO - KIT/COMBO"
                notes = ["kit/combo detectado no título"]
                if query_used:
                    notes.append(f"consulta usada: {query_used}")

            reviews = result.get("reviews")
            rating = result.get("rating")
            score = relevance_score(candidate_position, reviews, rating, candidate_multiple_sources)
            row = {
                "ean": ean,
                "seller": seller,
                "marketplace": marketplace,
                "produto_google": candidate_title or "não visível",
                "produto_loja": candidate_title or "não visível",
                "preco_atual": money_br(candidate_price_num),
                "frete": result.get("delivery") or "não visível",
                "preco_total_estimado": money_br(candidate_total_num),
                "preco_atual_num": candidate_price_num,
                "frete_num": candidate_delivery_num,
                "preco_total_estimado_num": candidate_total_num,
                "quantidade_vendida_ou_proxy": build_proxy(candidate_position, reviews, rating, candidate_multiple_sources),
                "avaliacoes": reviews if reviews is not None else "não visível",
                "nota": rating if rating is not None else "não visível",
                "url_completo": candidate_product_link or "não visível",
                "observacoes": " | ".join(notes),
                "alta_relevancia": relevance_label(score, reviews, candidate_position),
                "relevancia_score": score,
                "status_validacao": status,
                "kit_combo_detectado": status == "EXCLUÍDO - KIT/COMBO",
                "origem_google_posicao": candidate_position if candidate_position is not None else search_rank,
            }

            dedupe_key = f"{ean}|{row['url_completo']}|{row['seller']}|{row['marketplace']}"
            if dedupe_key not in seen_keys:
                seen_keys.add(dedupe_key)
                rows.append(row)

    return rows


def summarize_ean(ean: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    eligible = [
        r for r in rows
        if r["status_validacao"] in ("EXATO", "PROVÁVEL") and not r["kit_combo_detectado"]
    ]
    ml_rows = [r for r in eligible if r["marketplace"] == "Mercado Livre"]
    totals = [r["preco_total_estimado_num"] for r in eligible if r["preco_total_estimado_num"] is not None]

    leader = max(
        eligible,
        key=lambda x: (x["relevancia_score"], -(x["preco_total_estimado_num"] or x["preco_atual_num"] or 10**9)),
        default=None,
    )
    cheapest = min(
        eligible,
        key=lambda x: (x["preco_total_estimado_num"] if x["preco_total_estimado_num"] is not None else 10**9),
        default=None,
    )
    priciest = max(
        eligible,
        key=lambda x: (x["preco_total_estimado_num"] if x["preco_total_estimado_num"] is not None else -1),
        default=None,
    )

    min_total = min(totals) if totals else None
    max_total = max(totals) if totals else None
    avg_total = statistics.mean(totals) if totals else None

    p25 = percentile(totals, 25)
    p45 = percentile(totals, 45)
    if p25 is None and min_total is not None:
        p25 = min_total
    if p45 is None and avg_total is not None:
        p45 = avg_total

    faixa_ideal = (
        f"{money_br(p25)} a {money_br(p45)}"
        if p25 is not None and p45 is not None
        else "não visível"
    )

    leader_name = leader["seller"] if leader else "não visível"
    leader_mkt = leader["marketplace"] if leader else "não visível"

    if not eligible:
        resumo = (
            f"Não foram encontrados anúncios confirmados/prováveis para o GTIN {ean} no Google Shopping com a localização configurada. "
            f"Os identificadores GTIN ajudam o Google a classificar produtos, mas a busca ainda depende do que está indexado e disponível na Shopping tab."
        )
        conclusao = (
            "Sem base competitiva suficiente para sugerir faixa de preço ideal. "
            "Tente nova coleta com outra localização, confira se o GTIN está correto e valide se o produto está indexado no Google Shopping."
        )
    else:
        resumo = (
            f"Foram encontrados {len(eligible)} anúncios confirmados/prováveis para o GTIN {ean}. "
            f"O seller mais forte por proxy de relevância é {leader_name} em {leader_mkt}. "
            f"O menor preço total estimado é {money_br(min_total)}, o maior é {money_br(max_total)} "
            f"e o preço médio é {money_br(avg_total)}. "
            f"Foram encontrados {len(ml_rows)} anúncios elegíveis de Mercado Livre."
        )

        conclusao = (
            f"Faixa de preço ideal sugerida: {faixa_ideal}. "
            f"Seller mais forte: {leader_name} ({leader_mkt}). "
            f"Menor preço observado: {cheapest['preco_total_estimado'] if cheapest else 'não visível'}. "
            f"Maior preço observado: {priciest['preco_total_estimado'] if priciest else 'não visível'}."
        )

    return {
        "ean": ean,
        "anuncios_confirmados_ou_provaveis": len(eligible),
        "seller_mais_forte_proxy": leader_name,
        "marketplace_lider": leader_mkt,
        "menor_preco_total": money_br(min_total),
        "maior_preco_total": money_br(max_total),
        "preco_medio_total": money_br(avg_total),
        "faixa_preco_ideal": faixa_ideal,
        "qtd_anuncios_mercado_livre": len(ml_rows),
        "resumo_executivo": resumo,
        "conclusao_pratica": conclusao,
    }


def build_ml_top10(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_ean: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_ean.setdefault(row["ean"], []).append(row)

    final_rows: List[Dict[str, Any]] = []
    for ean, ean_rows in by_ean.items():
        eligible_ml = [
            r for r in ean_rows
            if r["marketplace"] == "Mercado Livre"
            and r["status_validacao"] in ("EXATO", "PROVÁVEL")
            and not r["kit_combo_detectado"]
        ]
        eligible_ml.sort(
            key=lambda x: (
                -x["relevancia_score"],
                x["preco_total_estimado_num"] if x["preco_total_estimado_num"] is not None else 10**9,
                x["origem_google_posicao"] if x["origem_google_posicao"] is not None else 10**9,
            )
        )

        for idx, row in enumerate(eligible_ml[:10], start=1):
            final_rows.append(
                {
                    "ean": ean,
                    "ranking": idx,
                    "seller": row["seller"],
                    "marketplace": row["marketplace"],
                    "preco_atual": row["preco_atual"],
                    "frete": row["frete"],
                    "preco_total_estimado": row["preco_total_estimado"],
                    "quantidade_vendida_ou_proxy": row["quantidade_vendida_ou_proxy"],
                    "avaliacoes": row["avaliacoes"],
                    "nota": row["nota"],
                    "url_completo": row["url_completo"],
                    "observacoes": row["observacoes"],
                    "alta_relevancia": row["alta_relevancia"],
                    "status_validacao": row["status_validacao"],
                }
            )

    return final_rows


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/analyze")
def analyze(payload: AnalyzeRequest, x_app_token: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if APP_TOKEN and x_app_token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Token inválido.")

    try:
        all_rows: List[Dict[str, Any]] = []
        summaries: List[Dict[str, Any]] = []

        for ean in payload.eans:
            rows = build_rows_for_ean(
                ean=ean,
                gl=payload.gl,
                hl=payload.hl,
                location=payload.location,
                max_products_per_ean=payload.max_products_per_ean,
            )
            all_rows.extend(rows)
            summaries.append(summarize_ean(ean, rows))

        ml_top10 = build_ml_top10(all_rows)

        return {
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "location": payload.location,
                "gl": payload.gl,
                "hl": payload.hl,
                "max_products_per_ean": payload.max_products_per_ean,
            },
            "summary": summaries,
            "results": all_rows,
            "mercado_livre_top10": ml_top10,
        }
    except ExternalAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Erro interno no /analyze")
        raise HTTPException(status_code=500, detail=f"Erro interno no worker: {exc}") from exc
