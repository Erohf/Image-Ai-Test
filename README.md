# Experimento de TCC: Resiliência de Marcas d'Água (C2PA e SynthID) sob Compressão de Imagens

Este projeto implementa o pipeline automatizado para o Trabalho de Conclusão de Curso (TCC) que investiga como diferentes modelos e níveis de compressão de imagem (lossy e lossless) afetam a detecção de proveniência (**C2PA** e **SynthID**) em imagens geradas por Inteligência Artificial versus imagens reais.

O experimento utiliza o endpoint oficial da OpenAI:
`POST /v1/content_provenance_checks` (documentado na biblioteca oficial `openai>=2.52.0`).

---

## 📁 Estrutura do Projeto

```text
Image Ai Test/
├── input_images/             # Imagens originais de entrada
│   ├── ia/                   # 5 imagens geradas por IA (DALL-E 3, etc.)
│   └── real/                 # 5 fotografias reais
├── processed_images/         # Variações geradas automaticamente com Pillow
├── results/                  # Saídas consolidadas do experimento
│   ├── experimento_tcc.csv   # Relatório tabular em CSV (UTF-8)
│   └── experimento_tcc.db    # Banco de dados SQLite com a tabela 'experimento_tcc'
├── .env                      # Variáveis de ambiente (sua OPENAI_API_KEY)
├── .env.example              # Modelo para configuração da chave
├── requirements.txt          # Dependências do Python
├── experimento_tcc.py        # Código modular do experimento
├── main.py                   # Ponto de entrada rápido
├── TCC.md                    # Especificação de requisitos do experimento
└── README.md                 # Este guia de instruções
```

---

## 🛠️ Requisitos e Instalação

### 1. Criar e ativar o ambiente virtual (Recomendado)

No terminal (PowerShell no Windows):

```powershell
# Cria o ambiente virtual
python -m venv venv

# Ativa o ambiente virtual
.\venv\Scripts\Activate.ps1
```

> **Dica**: Se o PowerShell bloquear a execução de scripts, execute antes:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. Instalar as dependências

```powershell
pip install -r requirements.txt
```

As principais bibliotecas são:
- `openai>=2.52.0`: SDK oficial da OpenAI com suporte ao endpoint `content_provenance_checks`.
- `pillow>=10.0.0`: Manipulação de imagens e compressões lossy/lossless.
- `pandas>=2.0.0`: Estruturação dos dados e persistência.
- `tqdm>=4.65.0`: Barra de progresso interativa no terminal.
- `python-dotenv>=1.0.0`: Carregamento seguro da chave da API via `.env`.

---

## 🔑 Configuração da Chave da OpenAI

1. Copie o arquivo `.env.example` para `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```
2. Abra o arquivo `.env` e adicione sua chave de API:
   ```env
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

---

## 📸 Dataset de Entrada

Coloque suas imagens dentro de `input_images/`:
- **5 imagens geradas por IA**: coloque dentro de `input_images/ia/` (ex: geradas com DALL-E 3 ou ChatGPT contendo marcas C2PA/SynthID).
- **5 imagens reais**: coloque dentro de `input_images/real/` (fotografias reais de câmera/smartphone).

Formatos aceitos: `.png`, `.jpg`, `.jpeg`, `.webp`.

---

## 🚀 Como Executar o Script

### A. Execução Completa (Oficial)
Após adicionar suas 10 imagens e a chave no `.env`:
```powershell
python main.py
```
O script irá:
1. Ler as 10 imagens originais.
2. Gerar **todas as variações de compressão** em `processed_images/`:
   - **JPEG (Lossy)**: 11 níveis de qualidade (0%, 10%, 20%, ..., 100%).
   - **WebP (Lossy)**: 11 níveis de qualidade (0%, 10%, 20%, ..., 100%).
   - **PNG (Lossless)**: preservação integral dos dados.
   - **WebP (Lossless)**: preservação integral dos dados.
   - **Original Baseline**: cópia fiel de referência.
   *(Total: 24 variações por imagem = 240 testes no total)*
3. Enviar cada imagem para `client.content_provenance_checks.create(...)`.
4. Tratar erros de taxa com retry e exponential backoff automático.
5. Exportar os resultados para CSV e SQLite em `results/`.

---

### B. Opções Úteis de Linha de Comando

| Comando | Descrição |
| :--- | :--- |
| `python main.py --dry-run` | Executa o pipeline simulando as chamadas da API (sem gastar créditos). Ideal para validar o fluxo. |
| `python main.py --apenas-processar` | Gera apenas as variações e compressões de imagem sem consultar a API da OpenAI. |
| `python main.py --criar-amostras` | Cria automaticamente 10 imagens sintéticas de exemplo (5 IA e 5 Reais) para testar o ambiente. |
| `python main.py --input-dir ./pasta` | Permite customizar a pasta das imagens de entrada. |
| `python main.py --results-dir ./saida` | Permite customizar a pasta dos relatórios finais. |

---

## 📊 Estrutura dos Dados Exportados

O arquivo `results/experimento_tcc.csv` e a tabela `experimento_tcc` no `results/experimento_tcc.db` possuem as seguintes colunas:

| Coluna | Descrição |
| :--- | :--- |
| `id_imagem` | Identificador da imagem de origem |
| `categoria` | "IA" ou "Real" |
| `formato` | Formato do arquivo (JPEG, PNG, WEBP) |
| `tipo_compressao` | "Lossy" ou "Lossless" |
| `nivel_qualidade` | Nível de qualidade (0 a 100 ou "N/A" para lossless) |
| `c2pa_outcome` | Resultado C2PA (`detected`, `not_detected`, `erro`) |
| `c2pa_validation_state` | Estado de validação C2PA (`trusted`, `valid`, `invalid`, `not_present`) |
| `c2pa_issuer` | Emissor do manifesto C2PA (ex: OpenAI, quando disponível) |
| `synthid_outcome` | Resultado SynthID (`detected`, `not_detected`, `erro`) |
| `resposta_bruta_json` | JSON completo retornado pela API para auditoria científica |
