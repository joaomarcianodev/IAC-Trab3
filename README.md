> Trabalho 3 de Inteligência Artificial e Computacional - João Augusto Marciano Silva

# 🎵 IA Análise de Músicas - GUIA DE EXECUÇÃO RÁPIDA

Sistema de detecção de ofensas em áudio utilizando OpenAI Whisper
(transcrição), BERT (análise de sentimento) e Llama 3.2 (análise
contextual via Ollama).

------------------------------------------------------------------------

📋 Pré-requisitos do Sistema

Antes de iniciar, certifique-se de ter instalado:

1.  Python 3.10+
2.  FFmpeg (Obrigatório para processamento de áudio)
    -   Windows: Baixe, extraia e adicione a pasta bin ao PATH do sistema.
    -   ou rode no PowerShell: winget install Gyan.FFmpeg
    -   Teste: Abra o terminal e digite ffmpeg -version.
3.  Ollama (Para rodar a IA Llama 3.2)
    -   Download em: ollama.com

------------------------------------------------------------------------

🚀 Instalação e Configuração

1. Configurar Ambiente Virtual

Recomendado para isolar as dependências do projeto.

    # Criar a venv
    python -m venv .venv

    # Ativar a venv (Windows PowerShell)
    .\.venv\Scripts\Activate

    # Ativar a venv (Linux/Mac)
    source .venv/bin/activate

2. Instalar Dependências

         pip install -r requirements.txt

3. Preparar a IA (Ollama)

Com o Ollama instalado, baixe o modelo Llama 3.2 (3B):

    ollama pull llama3.2

------------------------------------------------------------------------

▶️ Como Executar

Inicie o Ollama

Certifique-se de que o aplicativo Ollama está rodando em segundo plano
ou execute:

    ollama serve

Rode a Aplicação

    uvicorn main:app --reload

Acesse

http://127.0.0.1:8000

------------------------------------------------------------------------

🛠️ Troubleshooting (Problemas Comuns)

Tabela de Erros e Soluções

|Erro|Solução Provável|
|:---:|:---:|
|FileNotFoundError [WinError 2]|O FFmpeg não está instalado ou não está no PATH.|
|ConnectionRefusedError [Ollama]|O Ollama não está rodando. Abra o app ou execute ollama serve|
|ImportError [transforms]|A biblioteca não foi instalada. Rode pip install -r requirements.txt|

------------------------------------------------------------------------

📦 Stack Tecnológica

-   Backend: FastAPI, Uvicorn
-   IA/ML: PyTorch, OpenAI Whisper, HuggingFace Transformers
-   LLM: Llama 3.2 (via Ollama)
-   Frontend: HTML5, Bootstrap 5, Jinja2
