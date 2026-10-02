Atue como um Especialista em Processamento Digital de Imagens, Inteligência Artificial e Desenvolvimento em Python.

Estou desenvolvendo o meu Trabalho de Conclusão de Curso (TCC), que investiga como diferentes modelos e níveis de compressão de imagem (lossy e lossless) afetam a capacidade de detecção de marcas d'água e proveniência (C2PA e SynthID) em imagens geradas por IA versus imagens reais.

Para a verificação das marcas d'água e proveniência, o experimento deve obrigatoriamente utilizar o endpoint oficial da OpenAI para verificação de proveniência de conteúdo (`POST /v1/content_provenance_checks`), conforme documentado em: https://developers.openai.com/api/docs/guides/content-provenance

Preciso que você escreva um script automatizado e completo em Python para executar este experimento.

### REQUISITOS DO EXPERIMENTO:

1. **Dataset de Entrada:**
   - 5 imagens geradas por IA.
   - 5 imagens reais (fotografias).
   - O algoritmo deve ler essas imagens de um diretório de entrada (`/input_images`).

2. **Geração de Variações (Compressão & Formato):**
   - **Formato Lossy (JPEG):** Gerar variações aplicando níveis de qualidade de 0% a 100%, variando de 10% em 10% (0%, 10%, 20%, ..., 100%).
   - **Formato Lossless (PNG e WebP Lossless):** Gerar as variações mantendo a preservação total de dados como baseline de comparação.
   - **Outros Formatos de Compressão:** Incluir variações em WebP (lossy) para comparar contra o JPEG.
   - **Nomenclatura e Organização:** Salvar cada variação em uma pasta de saída (`/processed_images`) usando uma convenção clara (ex: `[ID]_[tipo_IA_ou_REAL]_[formato]_[qualidade].[ext]`).

3. **Integração com a API de Proveniência da OpenAI:**
   - Utilizar a biblioteca oficial da OpenAI para Python (`openai>=2.52.0`).
   - Para cada imagem gerada (originais e variações comprimidas), realizar a chamada:
     `client.content_provenance_checks.create(file=(filename, image_bytes, media_type))`
   - Extrair os resultados detalhados dos objetos do array `results` da resposta (especificamente os blocos de tipo `c2pa` e `synthid`).

4. **Coleta e Armazenamento de Dados:**
   - Estruturar e salvar os resultados de todas as chamadas em um DataFrame do `pandas` e exportar para um arquivo CSV (`/results/experimento_tcc.csv`) e banco SQLite (`/results/experimento_tcc.db`).
   - Não há necessidade de registrar a categoria de nicho ou estilo da imagem.
   - Campos obrigatórios no relatório final:
     - `id_imagem`: Identificador da imagem original.
     - `categoria`: "IA" ou "Real".
     - `formato`: Formato do arquivo (JPEG, PNG, WEBP, etc.).
     - `tipo_compressao`: "Lossy" ou "Lossless".
     - `nivel_qualidade`: Nível de qualidade (0 a 100, ou "N/A" para lossless).
     - `c2pa_outcome`: Resultado do C2PA ("detected", "not_detected", etc.).
     - `c2pa_validation_state`: Estado de validação do C2PA ("trusted", "valid", "invalid", "not_present").
     - `c2pa_issuer`: Emissor do manifesto C2PA (quando disponível).
     - `synthid_outcome`: Resultado do SynthID ("detected", "not_detected").
     - `resposta_bruta_json`: JSON string completo retornado pela API para auditoria.

### ESTRUTURA TÉCNICA DESEJADA NO CÓDIGO:
- Manipulação e compressão de imagens via biblioteca `Pillow` (PIL).
- Tratamento adequado de erros de HTTP e rate limits (HTTP 429), implementando uma lógica simples de retry (exponential backoff).
- Uso de `tqdm` para barra de progresso no terminal.
- Modularização do código em funções claras e bem documentadas: `gerar_variacoes_imagem()`, `verificar_proveniencia_openai()`, `executar_pipeline()`, `salvar_resultados()`.

Por favor, forneça o código Python completo, comentado passo a passo, acompanhado do arquivo `requirements.txt` e das instruções de execução.