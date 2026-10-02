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
import sys
import time
import json
import random
import logging
import sqlite3
import argparse
from pathlib import Path
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
    Aceita organização por pastas (ex: /ia/ e /real/) ou nome de arquivo (ex: ia_01.png, real_01.jpg).
    """
    nome_completo = caminho_imagem.stem.lower()
    caminho_str = str(caminho_imagem).lower()

    if "real" in nome_completo or "real" in caminho_str or "foto" in nome_completo:
        categoria = "Real"
    elif "ia" in nome_completo or "ai" in nome_completo or "synth" in nome_completo or "dalle" in nome_completo:
        categoria = "IA"
    else:
        # Se não puder inferir, assume IA por padrão
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
            # Tratamento de erro com exponential backoff
            erro_str = str(e)
            is_rate_limit = "429" in erro_str or "rate_limit" in erro_str.lower()
            is_server_error = any(code in erro_str for code in ("500", "502", "503", "504"))

            if tentativa < max_retries and (is_rate_limit or is_server_error or "connection" in erro_str.lower()):
                # Exponential backoff + jitter
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
    logger.info(f"Resultados exportados para CSV com sucesso: {caminho_csv}")

    # 2. Salva em banco SQLite
    try:
        conn = sqlite3.connect(caminho_db)
        df.to_sql("experimento_tcc", conn, if_exists="replace", index=False)
        conn.close()
        logger.info(f"Resultados gravados no banco SQLite com sucesso: {caminho_db}")
    except Exception as e:
        logger.error(f"Erro ao salvar dados no SQLite ({caminho_db}): {e}", exc_info=True)

    return caminho_csv, caminho_db


# ==============================================================================
# 5. GERADOR DE IMAGENS DE TESTE (DEMO / AMBIENTE INICIAL)
# ==============================================================================

def criar_imagens_exemplo(input_dir: Path) -> None:
    """
    Cria imagens sintéticas de exemplo (5 IA e 5 Reais) na pasta input_images
    caso o usuário queira validar o pipeline antes de colocar as imagens definitivas.
    """
    from PIL import ImageDraw

    pasta_ia = input_dir / "ia"
    pasta_real = input_dir / "real"
    pasta_ia.mkdir(parents=True, exist_ok=True)
    pasta_real.mkdir(parents=True, exist_ok=True)

    cores_ia = [(70, 130, 180), (147, 112, 219), (60, 179, 113), (255, 105, 180), (255, 165, 0)]
    cores_real = [(100, 100, 100), (139, 69, 19), (47, 79, 79), (72, 61, 139), (112, 128, 144)]

    for idx, cor in enumerate(cores_ia, 1):
        img = Image.new("RGB", (512, 512), color=cor)
        draw = ImageDraw.Draw(img)
        draw.text((20, 240), f"Amostra IA #{idx}", fill=(255, 255, 255))
        img.save(pasta_ia / f"ia_exemplo_{idx:02d}.png", format="PNG")

    for idx, cor in enumerate(cores_real, 1):
        img = Image.new("RGB", (512, 512), color=cor)
        draw = ImageDraw.Draw(img)
        draw.text((20, 240), f"Amostra Real #{idx}", fill=(255, 255, 255))
        img.save(pasta_real / f"real_exemplo_{idx:02d}.jpg", format="JPEG", quality=95)

    logger.info(f"10 imagens de exemplo geradas com sucesso em: {input_dir}")


# ==============================================================================
# 6. EXECUÇÃO DO PIPELINE COMPLETO
# ==============================================================================

def executar_pipeline(
    input_dir: Path,
    processed_dir: Path,
    results_dir: Path,
    apenas_processar: bool = False,
    dry_run: bool = False,
    testar_ia: bool = False,
    apenas_ia: bool = False,
) -> pd.DataFrame:
    """
    Executa o fluxo completo do experimento de TCC:
      1. Localiza e valida imagens de entrada (5 IA e 5 Reais)
      2. Gera todas as variações de compressão e formato com Pillow
      3. Consulta a API de Content Provenance da OpenAI com tqdm e retry
      4. Consolida métricas em DataFrame e exporta CSV + SQLite
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
            f"Coloque 5 imagens geradas por IA e 5 imagens reais na pasta '{input_dir}'.\n"
            f"Você também pode usar o argumento '--criar-amostras' para gerar dados sintéticos de teste."
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

    # 2. CHAMADAS PARA A API DA OPENAI
    logger.info(f"\n[ETAPA 2/3] Integrando com a API de Proveniência da OpenAI ({len(todas_variacoes)} chamadas programadas)...")

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

    registros_finais: List[Dict[str, Any]] = []

    for item in tqdm(todas_variacoes, desc="Consultando API OpenAI", unit="amostra"):
        caminho_arq = item["caminho_arquivo"]

        if dry_run or client is None:
            # Modo Simulado (Dry-Run para testes e validação da estrutura)
            is_ia = item["categoria"] == "IA"
            qualidade = item["nivel_qualidade"]
            # Exemplo heurístico de degradação em dry-run:
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
        else:
            # Chamada real à API oficial
            resultado_api = verificar_proveniencia_openai(client, caminho_arq)

        # Monta linha com campos estritamente obrigatórios no TCC.md
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
        }
        registros_finais.append(registro)

    # 3. CONSOLIDAÇÃO E SALVAMENTO
    logger.info("\n[ETAPA 3/3] Consolidando e salvando resultados em CSV e SQLite...")
    df_resultados = pd.DataFrame(registros_finais)

    salvar_resultados(df_resultados, results_dir)

    logger.info("=" * 70)
    logger.info("EXPERIMENTO CONCLUÍDO COM SUCESSO!")
    logger.info(f"Total de registros analisados: {len(df_resultados)}")
    logger.info("=" * 70)

    return df_resultados


# ==============================================================================
# 7. PARSER DE LINHA DE COMANDO (CLI)
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

    args = parser.parse_args()

    if getattr(args, "criar_amostras", False):
        criar_imagens_exemplo(args.input_dir)
        if len(sys.argv) == 2 and "--criar-amostras" in sys.argv:
            return

    executar_pipeline(
        input_dir=args.input_dir,
        processed_dir=args.processed_dir,
        results_dir=args.results_dir,
        apenas_processar=args.apenas_processar,
        dry_run=args.dry_run,
        testar_ia=getattr(args, "testar_ia", False),
        apenas_ia=getattr(args, "apenas_ia", False),
    )


if __name__ == "__main__":
    main()
