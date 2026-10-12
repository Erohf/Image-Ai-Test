#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experimento de TCC:
Impacto da Compressão de Imagens na Detecção de Proveniência (C2PA e SynthID)
via OpenAI Content Provenance Checks API (POST /v1/content_provenance_checks).

Autor: Lucas (TCC)
Requisitos: openai>=2.52.0, Pillow, pandas, tqdm, python-dotenv
"""

import os
import re
import sys
import time
import json
import random
import logging
import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env (caso exista)
load_dotenv()

# Configuração de logging para terminal e arquivo
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TCC_Experimento")

# ==============================================================================
# 1. FUNÇÕES AUXILIARES E PREPARAÇÃO
# ==============================================================================

def mapear_mime_type(extensao: str) -> str:
    """Retorna o MIME type apropriado com base na extensão do arquivo."""
    ext = extensao.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext == "webp":
        return "image/webp"
    return "application/octet-stream"


def converter_para_rgb_se_necessario(img: Image.Image) -> Image.Image:
    """
    Converte imagens com canal alfa (RGBA, LA, P) para RGB preenchendo o fundo
    com branco. Isso evita erros ao salvar formatos que não suportam alfa (ex: JPEG).
    """
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img_rgba = img.convert("RGBA")
        fundo_branco = Image.new("RGB", img_rgba.size, (255, 255, 255))
        fundo_branco.paste(img_rgba, mask=img_rgba.split()[3])
        return fundo_branco
    elif img.mode != "RGB":
        return img.convert("RGB")
    return img


def identificar_categoria_e_id(caminho_imagem: Path) -> Tuple[str, str]:
    """
    Identifica a categoria ('IA' ou 'Real') e um identificador único para a imagem.
    Prioriza a pasta onde a imagem está (/ia/ ou /real/) ou termos no nome do arquivo.
    """
    nome_completo = caminho_imagem.stem.lower()
    pasta_pai = caminho_imagem.parent.name.lower()
    caminho_str = str(caminho_imagem).lower()

    # Prioridade 1: Nome da pasta imediata
    if pasta_pai == "real":
        categoria = "Real"
    elif pasta_pai in ("ia", "ai"):
        categoria = "IA"
    # Prioridade 2: Termos explícitos no nome do arquivo
    elif "real" in nome_completo or "foto" in nome_completo:
        categoria = "Real"
    elif "ia" in nome_completo or "ai" in nome_completo or "synth" in nome_completo or "dalle" in nome_completo:
        categoria = "IA"
    # Prioridade 3: Caminho completo
    elif "real" in caminho_str:
        categoria = "Real"
    else:
        categoria = "IA"

    id_imagem = caminho_imagem.stem
    return id_imagem, categoria


# ==============================================================================
# 2. GERAÇÃO DE VARIAÇÕES (COMPRESSÃO & FORMATOS)
# ==============================================================================

def gerar_variacoes_imagem(
    caminho_origem: Path,
    id_imagem: str,
    categoria: str,
    output_dir: Path
) -> List[Dict[str, Any]]:
    """
    Gera todas as variações de compressão e formato especificadas no TCC:
      - JPEG (Lossy): qualidade de 0% a 100% de 10 em 10 (0, 10, ..., 100)
      - PNG (Lossless): preservação total de dados (baseline)
      - WebP (Lossless): preservação total de dados (baseline)
      - WebP (Lossy): qualidade de 0% a 100% de 10 em 10 (0, 10, ..., 100)
      - Imagem Original: incluída como referência de baseline

    Nomenclatura:
      [ID]_[tipo_IA_ou_REAL]_[formato]_[qualidade].[ext]
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    variacoes_geradas: List[Dict[str, Any]] = []

    try:
        with Image.open(caminho_origem) as img_original:
            # 1. Baseline: Salvar cópia da imagem original com nomenclatura padronizada
            ext_original = caminho_origem.suffix.lower().lstrip(".")
            nome_original = f"{id_imagem}_{categoria}_original_baseline.{ext_original}"
            caminho_copia_orig = output_dir / nome_original
            img_original.save(caminho_copia_orig)

            variacoes_geradas.append({
                "id_imagem": id_imagem,
                "categoria": categoria,
                "formato": ext_original.upper(),
                "tipo_compressao": "Lossless" if ext_original == "png" else "Original",
                "nivel_qualidade": "Original",
                "caminho_arquivo": caminho_copia_orig,
            })

            # Prepara versão RGB para formatos que exigem 3 canais (ex: JPEG)
            img_rgb = converter_para_rgb_se_necessario(img_original)

            # 2. Variações JPEG (Lossy): 0% a 100% de 10 em 10
            for q in range(0, 101, 10):
                nome_arq = f"{id_imagem}_{categoria}_jpeg_q{q}.jpg"
                caminho_saida = output_dir / nome_arq
                img_rgb.save(caminho_saida, format="JPEG", quality=q)

                variacoes_geradas.append({
                    "id_imagem": id_imagem,
                    "categoria": categoria,
                    "formato": "JPEG",
                    "tipo_compressao": "Lossy",
                    "nivel_qualidade": q,
                    "caminho_arquivo": caminho_saida,
                })

            # 3. Variação PNG (Lossless baseline)
            nome_png = f"{id_imagem}_{categoria}_png_lossless.png"
            caminho_png = output_dir / nome_png
            img_original.save(caminho_png, format="PNG")

            variacoes_geradas.append({
                "id_imagem": id_imagem,
                "categoria": categoria,
                "formato": "PNG",
                "tipo_compressao": "Lossless",
                "nivel_qualidade": "N/A",
                "caminho_arquivo": caminho_png,
            })

            # 4. Variação WebP (Lossless baseline)
            nome_webp_ll = f"{id_imagem}_{categoria}_webp_lossless.webp"
            caminho_webp_ll = output_dir / nome_webp_ll
            img_original.save(caminho_webp_ll, format="WEBP", lossless=True)

            variacoes_geradas.append({
                "id_imagem": id_imagem,
                "categoria": categoria,
                "formato": "WEBP",
                "tipo_compressao": "Lossless",
                "nivel_qualidade": "N/A",
                "caminho_arquivo": caminho_webp_ll,
            })

            # 5. Variações WebP (Lossy): 0% a 100% de 10 em 10
            for q in range(0, 101, 10):
                nome_webp_lossy = f"{id_imagem}_{categoria}_webp_lossy_q{q}.webp"
                caminho_webp_lossy = output_dir / nome_webp_lossy
                img_original.save(caminho_webp_lossy, format="WEBP", quality=q, lossless=False)

                variacoes_geradas.append({
                    "id_imagem": id_imagem,
                    "categoria": categoria,
                    "formato": "WEBP",
                    "tipo_compressao": "Lossy",
                    "nivel_qualidade": q,
                    "caminho_arquivo": caminho_webp_lossy,
                })

    except Exception as e:
        logger.error(f"Erro ao processar imagem {caminho_origem}: {e}", exc_info=True)

    return variacoes_geradas


# ==============================================================================
# 3. INTEGRAÇÃO COM A API DE PROVENIÊNCIA DA OPENAI
# ==============================================================================

def verificar_proveniencia_openai(
    client: Any,
    caminho_arquivo: Path,
    max_retries: int = 5,
    base_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Realiza a chamada oficial para a API de Content Provenance da OpenAI:
      client.content_provenance_checks.create(file=(filename, image_bytes, media_type))

    Implementa retry robusto com exponential backoff para HTTP 429 e falhas transitórias.
    Extrai c2pa_outcome, c2pa_validation_state, c2pa_issuer, synthid_outcome e JSON bruto.
    """
    nome_arquivo = caminho_arquivo.name
    media_type = mapear_mime_type(caminho_arquivo.suffix)

    with open(caminho_arquivo, "rb") as f:
        image_bytes = f.read()

    file_tuple = (nome_arquivo, image_bytes, media_type)

    for tentativa in range(1, max_retries + 1):
        try:
            # Chamada oficial da API
            resposta = client.content_provenance_checks.create(file=file_tuple)

            # Converter resposta para dicionário estruturado
            if hasattr(resposta, "model_dump"):
                resp_dict = resposta.model_dump()
            elif hasattr(resposta, "to_dict"):
                resp_dict = resposta.to_dict()
            elif isinstance(resposta, dict):
                resp_dict = resposta
            else:
                resp_dict = json.loads(json.dumps(resposta, default=str))

            resposta_bruta_json = json.dumps(resp_dict, ensure_ascii=False)

            # Valores padrão de saída
            c2pa_outcome = "not_detected"
            c2pa_validation_state = "not_present"
            c2pa_issuer = "N/A"
            synthid_outcome = "not_detected"

            # Parse dos blocos em 'results'
            itens_results = resp_dict.get("results", [])

            for item in itens_results:
                if not isinstance(item, dict):
                    continue

                tipo = item.get("type", "").lower()

                if tipo == "c2pa":
                    c2pa_outcome = item.get("outcome", "detected" if item.get("detected") else "not_detected")
                    c2pa_validation_state = item.get("validation_state", "valid")
                    # Tenta obter o emissor de diferentes possíveis estruturas
                    c2pa_issuer = item.get("issuer") or item.get("manifest", {}).get("issuer") or "N/A"

                elif tipo == "synthid":
                    synthid_outcome = item.get("outcome", "detected" if item.get("detected") else "not_detected")

            # Fallback se a API retornar resultado consolidado direto no nível raiz
            if not itens_results and "result" in resp_dict:
                resultado_global = resp_dict.get("result")
                if resultado_global == "not_detected":
                    c2pa_outcome = "not_detected"
                    synthid_outcome = "not_detected"

            return {
                "c2pa_outcome": c2pa_outcome,
                "c2pa_validation_state": c2pa_validation_state,
                "c2pa_issuer": c2pa_issuer,
                "synthid_outcome": synthid_outcome,
                "resposta_bruta_json": resposta_bruta_json,
                "status_api": "sucesso",
            }

        except Exception as e:
            # Tratamento de erro com detecção inteligente de Rate Limit e backoff
            erro_str = str(e)
            is_rate_limit = "429" in erro_str or "rate_limit" in erro_str.lower()
            is_server_error = any(code in erro_str for code in ("500", "502", "503", "504"))

            # Tenta extrair o tempo exato indicado pela OpenAI (ex: "try again after 3279.897 seconds")
            match_tempo = re.search(r"try again after ([\d\.]+) seconds", erro_str)
            segundos_espera = float(match_tempo.group(1)) if match_tempo else None

            # Se a OpenAI pediu para esperar mais de 60 segundos (limite de cota por hora)
            if is_rate_limit and segundos_espera and segundos_espera > 60:
                logger.warning(
                    f"Cota horária da OpenAI atingida para {nome_arquivo}. "
                    f"Tempo exigido pela API: {segundos_espera/60:.1f} minutos ({segundos_espera:.0f}s)."
                )
                return {
                    "c2pa_outcome": "rate_limit",
                    "c2pa_validation_state": "rate_limit",
                    "c2pa_issuer": "N/A",
                    "synthid_outcome": "rate_limit",
                    "resposta_bruta_json": json.dumps({"erro": erro_str, "retry_after_seconds": segundos_espera}),
                    "status_api": "rate_limit_longo",
                    "segundos_espera": segundos_espera,
                }

            if tentativa < max_retries and (is_rate_limit or is_server_error or "connection" in erro_str.lower()):
                # Exponential backoff + jitter para erros de rajada curta
                espera = (base_delay * (2 ** (tentativa - 1))) + random.uniform(0.1, 1.0)
                logger.warning(
                    f"[Tentativa {tentativa}/{max_retries}] Erro transitório na API ({e}). "
                    f"Aguardando {espera:.2f}s antes de tentar novamente..."
                )
                time.sleep(espera)
            else:
                logger.error(f"Falha definitiva ao consultar API para {nome_arquivo}: {e}")
                return {
                    "c2pa_outcome": "erro",
                    "c2pa_validation_state": "erro",
                    "c2pa_issuer": "N/A",
                    "synthid_outcome": "erro",
                    "resposta_bruta_json": json.dumps({"erro": erro_str}),
                    "status_api": f"erro: {erro_str}",
                }

    return {
        "c2pa_outcome": "erro",
        "c2pa_validation_state": "erro",
        "c2pa_issuer": "N/A",
        "synthid_outcome": "erro",
        "resposta_bruta_json": json.dumps({"erro": "Max retries excedido"}),
        "status_api": "erro_max_retries",
    }


# ==============================================================================
# 4. SALVAMENTO E PERSISTÊNCIA DOS DADOS
# ==============================================================================

def obter_chave_amostra(id_imagem: str, categoria: str, formato: str, tipo_compressao: str, nivel_qualidade: Any) -> str:
    """Gera chave única padronizada para identificar a variação da amostra."""
    return f"{id_imagem}_{categoria}_{formato}_{tipo_compressao}_{nivel_qualidade}"


def salvar_resultados(df: pd.DataFrame, results_dir: Path) -> Tuple[Path, Path]:
    """
    Exporta o DataFrame com os resultados do experimento para:
      1. Arquivo CSV: /results/experimento_tcc.csv
      2. Banco SQLite: /results/experimento_tcc.db (tabela: experimento_tcc)
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    caminho_csv = results_dir / "experimento_tcc.csv"
    caminho_db = results_dir / "experimento_tcc.db"

    # 1. Salva CSV em UTF-8 com formatação limpa
    df.to_csv(caminho_csv, index=False, encoding="utf-8-sig")

    # 2. Salva em banco SQLite
    try:
        conn = sqlite3.connect(caminho_db)
        df.to_sql("experimento_tcc", conn, if_exists="replace", index=False)
        conn.close()
    except Exception as e:
        logger.error(f"Erro ao salvar dados no SQLite ({caminho_db}): {e}", exc_info=True)

    return caminho_csv, caminho_db


def carregar_dados_existentes(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Carrega todos os dados já salvos em results/experimento_tcc.csv.
    Se existirem dados antigos sem data_coleta, preenche com a data de hoje.
    Garante persistência integral contra quedas de luz e interrupções.
    """
    caminho_csv = results_dir / "experimento_tcc.csv"
    dados_mestre: Dict[str, Dict[str, Any]] = {}
    data_hoje_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if caminho_csv.exists():
        try:
            df_existente = pd.read_csv(caminho_csv)
            precisa_reescrever = False

            # Se a coluna data_coleta ainda não existir no CSV prévio
            if "data_coleta" not in df_existente.columns:
                df_existente["data_coleta"] = data_hoje_str
                precisa_reescrever = True

            for _, row in df_existente.iterrows():
                row_dict = row.to_dict()
                # Se data_coleta for nula ou vazia, preenche com a data de hoje
                data_val = str(row_dict.get("data_coleta", "")).strip()
                if not data_val or data_val.lower() in ("nan", "none", ""):
                    row_dict["data_coleta"] = data_hoje_str
                    precisa_reescrever = True

                chave = obter_chave_amostra(
                    str(row_dict["id_imagem"]),
                    str(row_dict["categoria"]),
                    str(row_dict["formato"]),
                    str(row_dict["tipo_compressao"]),
                    str(row_dict["nivel_qualidade"]),
                )
                dados_mestre[chave] = row_dict

            # Se atualizamos datas vazias para hoje, já sincroniza o arquivo no disco
            if precisa_reescrever and dados_mestre:
                salvar_resultados(pd.DataFrame(list(dados_mestre.values())), results_dir)
                logger.info(
                    f"Dados anteriores preservados: {len(dados_mestre)} amostras atualizadas com a data de hoje ({data_hoje_str[:10]})."
                )

        except Exception as e:
            logger.warning(f"Não foi possível ler dados existentes de {caminho_csv}: {e}")

    return dados_mestre


def amostra_esta_recente(registro: Dict[str, Any], max_dias: int = 10, permitir_simulado: bool = False) -> bool:
    """
    Verifica se uma amostra foi coletada com sucesso há menos de `max_dias` dias.
    Retorna True se os dados são recentes e devem ser preservados (não reenviar para a API).
    Retorna False se não tem data, deu erro, tem mais de `max_dias` dias ou é simulação.
    """
    outcome_c2pa = str(registro.get("c2pa_outcome", "")).lower()
    outcome_synth = str(registro.get("synthid_outcome", "")).lower()

    # Se a tentativa anterior terminou em erro ou rate limit, não é válida
    if outcome_c2pa in ("erro", "rate_limit", "nan", "") or outcome_synth in ("erro", "rate_limit", "nan", ""):
        return False

    # No modo oficial com API real, nunca considera dados simulados de dry-run como válidos
    resp_json = str(registro.get("resposta_bruta_json", "")).lower()
    if not permitir_simulado and '"simulado": true' in resp_json:
        return False

    data_str = str(registro.get("data_coleta", "")).strip()
    if not data_str or data_str.lower() in ("nan", "none"):
        return False

    try:
        if len(data_str) >= 19:
            data_dt = datetime.strptime(data_str[:19], "%Y-%m-%d %H:%M:%S")
        else:
            data_dt = datetime.strptime(data_str[:10], "%Y-%m-%d")

        dias_decorridos = (datetime.now() - data_dt).total_seconds() / 86400.0
        return dias_decorridos <= max_dias
    except Exception:
        return False


# ==============================================================================
# 5. EXECUÇÃO DO PIPELINE COMPLETO
# ==============================================================================

def executar_pipeline(
    input_dir: Path,
    processed_dir: Path,
    results_dir: Path,
    apenas_processar: bool = False,
    dry_run: bool = False,
    testar_ia: bool = False,
    apenas_ia: bool = False,
    delay: float = 1.5,
    esperar_rate_limit: bool = False,
) -> pd.DataFrame:
    """
    Executa o fluxo completo do experimento de TCC:
      1. Localiza e valida imagens de entrada (5 IA e 5 Reais)
      2. Gera todas as variações de compressão e formato com Pillow
      3. Consulta a API de Content Provenance da OpenAI com tqdm, retry e checkpointing de 10 dias
      4. Consolida métricas em DataFrame e exporta CSV + SQLite com data_coleta
    """
    logger.info("=" * 70)
    if testar_ia:
        logger.info("INICIANDO TESTE PILOTO: APENAS IMAGENS ORIGINAIS DE IA (INPUT_IMAGES)")
    elif apenas_ia:
        logger.info("INICIANDO PIPELINE: APENAS CATEGORIA IA (COM COMPRESSÕES)")
    else:
        logger.info("INICIANDO PIPELINE DO EXPERIMENTO DE TCC (IA + REAIS)")
    logger.info("=" * 70)

    # Coleta arquivos de imagem de entrada
    extensoes_validas = {".jpg", ".jpeg", ".png", ".webp"}
    arquivos_entrada: List[Path] = []

    if input_dir.exists():
        for arq in input_dir.rglob("*"):
            if arq.is_file() and arq.suffix.lower() in extensoes_validas:
                arquivos_entrada.append(arq)

    if not arquivos_entrada:
        logger.warning(
            f"Nenhuma imagem encontrada em '{input_dir}'!\n"
            f"Coloque 5 imagens geradas por IA na pasta '{input_dir}/ia' "
            f"e 5 imagens reais na pasta '{input_dir}/real'."
        )
        return pd.DataFrame()

    # Filtra por IA se solicitado
    if testar_ia or apenas_ia:
        arquivos_entrada = [
            arq for arq in arquivos_entrada
            if identificar_categoria_e_id(arq)[1] == "IA"
        ]
        logger.info(f"Filtro ativado: {len(arquivos_entrada)} imagens de IA selecionadas.")

    logger.info(f"Total de {len(arquivos_entrada)} imagens originais para análise.")

    # 1. GERAÇÃO DE VARIAÇÕES OU TESTE DIRETO DAS ORIGINAIS
    todas_variacoes: List[Dict[str, Any]] = []

    if testar_ia:
        logger.info("\n[ETAPA 1/3] Modo '--testar-ia': Pulando compressões e usando imagens originais diretamente...")
        for caminho_img in arquivos_entrada:
            id_imagem, categoria = identificar_categoria_e_id(caminho_img)
            ext_original = caminho_img.suffix.lower().lstrip(".")
            todas_variacoes.append({
                "id_imagem": id_imagem,
                "categoria": categoria,
                "formato": ext_original.upper(),
                "tipo_compressao": "Original",
                "nivel_qualidade": "Original",
                "caminho_arquivo": caminho_img,
            })
    else:
        logger.info("\n[ETAPA 1/3] Gerando variações de compressão (JPEG, WebP, PNG)...")
        for caminho_img in tqdm(arquivos_entrada, desc="Processando compressões", unit="imagem"):
            id_imagem, categoria = identificar_categoria_e_id(caminho_img)
            variacoes = gerar_variacoes_imagem(
                caminho_origem=caminho_img,
                id_imagem=id_imagem,
                categoria=categoria,
                output_dir=processed_dir
            )
            todas_variacoes.extend(variacoes)

        logger.info(f"Total de {len(todas_variacoes)} variações geradas na pasta '{processed_dir}'.")

    if apenas_processar:
        logger.info("Opção '--apenas-processar' ativada. Encerrando sem chamar a API.")
        return pd.DataFrame(todas_variacoes)

    # 2. CHAMADAS PARA A API DA OPENAI (COM CONTROLE DE 10 DIAS E CHECKPOINTING MESTRE)
    logger.info(f"\n[ETAPA 2/3] Integrando com a API de Proveniência da OpenAI ({len(todas_variacoes)} amostras)...")

    # Carrega dados mestre existentes garantindo integridade e preenchendo datas anteriores
    dados_mestre = carregar_dados_existentes(results_dir) if not dry_run else {}

    # Avalia quantas amostras já estão coletadas e válidas há menos de 10 dias
    amostras_recentes_poupadas = 0
    amostras_para_processar = []

    for item in todas_variacoes:
        chave = obter_chave_amostra(
            item["id_imagem"],
            item["categoria"],
            item["formato"],
            item["tipo_compressao"],
            item["nivel_qualidade"],
        )
        if chave in dados_mestre and amostra_esta_recente(dados_mestre[chave], max_dias=10, permitir_simulado=dry_run):
            amostras_recentes_poupadas += 1
        else:
            amostras_para_processar.append(item)

    logger.info(
        f"📊 Status do Dataset:\n"
        f"   - Total de amostras no experimento: {len(todas_variacoes)}\n"
        f"   - Coletadas recentemente (< 10 dias) mantidas: {amostras_recentes_poupadas}\n"
        f"   - Pendentes para consulta na API (novas ou > 10 dias): {len(amostras_para_processar)}"
    )

    if not amostras_para_processar:
        logger.info("🎉 Todas as amostras já possuem dados coletados há menos de 10 dias! Nenhum envio à API necessário.")
        # Retorna o DataFrame ordenado
        lista_ordenada = [
            dados_mestre[obter_chave_amostra(it["id_imagem"], it["categoria"], it["formato"], it["tipo_compressao"], it["nivel_qualidade"])]
            for it in todas_variacoes
            if obter_chave_amostra(it["id_imagem"], it["categoria"], it["formato"], it["tipo_compressao"], it["nivel_qualidade"]) in dados_mestre
        ]
        df_completo = pd.DataFrame(lista_ordenada)
        salvar_resultados(df_completo, results_dir)
        return df_completo

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    client = None

    if not dry_run:
        if not api_key or api_key.startswith("sua_chave") or "placeholder" in api_key:
            logger.warning(
                "OPENAI_API_KEY não encontrada ou inválida no arquivo .env!\n"
                "Para realizar as chamadas reais à API, configure sua chave no arquivo .env.\n"
                "Ativando modo simulado (--dry-run) para demonstração da estrutura."
            )
            dry_run = True
        else:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
            except Exception as e:
                logger.error(f"Erro ao instanciar cliente da OpenAI: {e}")
                dry_run = True

    pbar = tqdm(amostras_para_processar, desc="Consultando API OpenAI", unit="amostra")
    for item in pbar:
        chave_amostra = obter_chave_amostra(
            item["id_imagem"],
            item["categoria"],
            item["formato"],
            item["tipo_compressao"],
            item["nivel_qualidade"],
        )
        caminho_arq = item["caminho_arquivo"]

        if dry_run or client is None:
            # Modo Simulado (Dry-Run para testes e validação da estrutura)
            is_ia = item["categoria"] == "IA"
            qualidade = item["nivel_qualidade"]
            if is_ia:
                c2pa_status = "detected" if qualidade in ("Original", "N/A") or (isinstance(qualidade, int) and qualidade >= 40) else "not_detected"
                synthid_status = "detected" if qualidade in ("Original", "N/A") or (isinstance(qualidade, int) and qualidade >= 20) else "not_detected"
            else:
                c2pa_status = "not_detected"
                synthid_status = "not_detected"

            resultado_api = {
                "c2pa_outcome": c2pa_status,
                "c2pa_validation_state": "valid" if c2pa_status == "detected" else "not_present",
                "c2pa_issuer": "OpenAI" if c2pa_status == "detected" else "N/A",
                "synthid_outcome": synthid_status,
                "resposta_bruta_json": json.dumps({
                    "simulado": True,
                    "results": [
                        {"type": "c2pa", "outcome": c2pa_status, "validation_state": "valid" if c2pa_status == "detected" else "not_present"},
                        {"type": "synthid", "outcome": synthid_status}
                    ]
                }),
                "status_api": "simulado_dry_run"
            }
            data_devolucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            # Chamada real à API oficial
            resultado_api = verificar_proveniencia_openai(client, caminho_arq)
            data_devolucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Detecta bloqueio por janela horária (429 Rate Limit de longo prazo)
            if resultado_api.get("status_api") == "rate_limit_longo":
                segundos_espera = resultado_api.get("segundos_espera", 3600)
                hora_liberacao = datetime.now() + timedelta(seconds=segundos_espera)
                hora_str = hora_liberacao.strftime("%H:%M:%S")

                # Salva imediatamente TODO o dados_mestre (mantém 100% dos dados anteriores e atuais)
                if dados_mestre:
                    salvar_resultados(pd.DataFrame(list(dados_mestre.values())), results_dir)

                logger.warning("=" * 70)
                logger.warning("⚠️  LIMITE DE REQUISIÇÕES POR HORA DA OPENAI ATINGIDO!")
                logger.warning(
                    f"A OpenAI exige uma pausa de {segundos_espera/60:.1f} minutos "
                    f"({segundos_espera:.0f}s). Liberação estimada às: {hora_str}."
                )
                logger.warning(
                    f"Progresso 100% preservado no disco: {len(dados_mestre)} amostras salvas em 'results/'."
                )

                if esperar_rate_limit:
                    logger.info(f"Argumento '--esperar-rate-limit' ativo. Aguardando contagem regressiva até {hora_str}...")
                    tempo_restante = int(segundos_espera) + 10
                    while tempo_restante > 0:
                        minutos_rest = tempo_restante // 60
                        segundos_rest = tempo_restante % 60
                        pbar.set_description(f"Aguardando Rate Limit: {minutos_rest:02d}m{segundos_rest:02d}s")
                        time.sleep(min(10, tempo_restante))
                        tempo_restante -= 10
                    pbar.set_description("Consultando API OpenAI")
                    # Tenta novamente a mesma amostra após a espera
                    resultado_api = verificar_proveniencia_openai(client, caminho_arq)
                    data_devolucao = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                else:
                    logger.warning(
                        "O script foi pausado com segurança para não sobrecarregar sua conta.\n"
                        f"Basta executar novamente 'python main.py --esperar-rate-limit' por volta das {hora_str} "
                        "e o experimento continuará exatamente de onde parou sem repetir nada!"
                    )
                    logger.warning("=" * 70)
                    break

            # Delay suave entre requisições reais para evitar disparos em rajada
            if delay > 0:
                time.sleep(delay)

        # Monta linha com campos obrigatórios + data_coleta
        registro = {
            "id_imagem": item["id_imagem"],
            "categoria": item["categoria"],
            "formato": item["formato"],
            "tipo_compressao": item["tipo_compressao"],
            "nivel_qualidade": item["nivel_qualidade"],
            "c2pa_outcome": resultado_api["c2pa_outcome"],
            "c2pa_validation_state": resultado_api["c2pa_validation_state"],
            "c2pa_issuer": resultado_api["c2pa_issuer"],
            "synthid_outcome": resultado_api["synthid_outcome"],
            "resposta_bruta_json": resultado_api["resposta_bruta_json"],
            "data_coleta": data_devolucao,
        }

        # Atualiza a amostra no dicionário mestre (não perde nenhuma amostra prévia!)
        dados_mestre[chave_amostra] = registro

        # SALVAMENTO IMEDIATO A CADA NOVA REQUISIÇÃO: Proteção total contra quedas de luz!
        salvar_resultados(pd.DataFrame(list(dados_mestre.values())), results_dir)

    # 3. CONSOLIDAÇÃO E SALVAMENTO FINAL (ORDENADO CONFORME O DATASET)
    logger.info("\n[ETAPA 3/3] Consolidando e ordenando resultados finais em CSV e SQLite...")
    lista_final = []
    for it in todas_variacoes:
        ch = obter_chave_amostra(it["id_imagem"], it["categoria"], it["formato"], it["tipo_compressao"], it["nivel_qualidade"])
        if ch in dados_mestre:
            lista_final.append(dados_mestre[ch])

    df_resultados = pd.DataFrame(lista_final if lista_final else list(dados_mestre.values()))
    salvar_resultados(df_resultados, results_dir)

    logger.info("=" * 70)
    logger.info(f"Progresso consolidado: {len(df_resultados)} de {len(todas_variacoes)} amostras salvas.")
    logger.info(f"Relatórios atualizados em: {results_dir / 'experimento_tcc.csv'}")
    logger.info("=" * 70)

    return df_resultados


# ==============================================================================
# 6. PARSER DE LINHA DE COMANDO (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Script Automatizado para Experimento de TCC: "
                    "Resiliência de C2PA e SynthID sob Compressão de Imagens."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("./input_images"),
        help="Diretório com as imagens de entrada (5 IA e 5 Reais). Padrão: ./input_images",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("./processed_images"),
        help="Diretório onde serão salvas as variações geradas. Padrão: ./processed_images",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./results"),
        help="Diretório onde serão exportados o CSV e o SQLite. Padrão: ./results",
    )
    parser.add_argument(
        "--apenas-processar",
        action="store_true",
        help="Apenas gera as variações de imagem e encerra, sem consultar a API da OpenAI.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Executa em modo simulado (sem gastar chamadas ou créditos da API da OpenAI).",
    )
    parser.add_argument(
        "--testar-ia",
        action="store_true",
        help="Executa o teste piloto chamando a API somente para as imagens originais de IA em input_images (sem compressões), para avaliar detecção e custos.",
    )
    parser.add_argument(
        "--apenas-ia",
        action="store_true",
        help="Executa o pipeline completo (com variações de compressão) apenas para as imagens de IA, ignorando as reais.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Intervalo em segundos entre cada requisição para evitar disparos em rajada. Padrão: 1.5s",
    )
    parser.add_argument(
        "--esperar-rate-limit",
        action="store_true",
        help="Caso atinja o limite por hora da OpenAI, aguarda a contagem regressiva automaticamente em vez de encerrar.",
    )

    args = parser.parse_args()

    executar_pipeline(
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        apenas_processar=args.apenas_processar,
        dry_run=args.dry_run,
        testar_ia=getattr(args, "testar_ia", False),
        apenas_ia=getattr(args, "apenas_ia", False),
        delay=getattr(args, "delay", 1.5),
        esperar_rate_limit=getattr(args, "esperar_rate_limit", False),
    )


if __name__ == "__main__":
    main()
