# IAC-Trab3

Trabalho 3 de Inteligência Artificial e Computacional

========================================================================
IA Análise de Músicas - GUIA DE EXECUÇÃO RÁPIDA
========================================================================

1. PRÉ-REQUISITOS DO SISTEMA

---

[ ] Python 3.10 ou superior instalado.
[ ] FFmpeg instalado e adicionado ao PATH do Windows (Obrigatório para o Whisper).
-> Teste no terminal: ffmpeg -version
[ ] Ollama instalado (https://ollama.com).

2. CONFIGURAÇÃO DE AMBIENTE

---

1. Crie e ative um ambiente virtual (Recomendado):
   python -m venv .venv
   .\.venv\Scripts\Activate (Windows PowerShell)

2. Instale as dependências Python:
   pip install -r requirements.txt

3. Prepare o Modelo Llama (Ollama):
   Abra o terminal e execute:
   ollama pull llama3.2

4. EXECUÇÃO

---

Passo 1: Garanta que o Ollama esteja rodando em segundo plano.
(Geralmente ele inicia com o Windows, ou rode `ollama serve`).

Passo 2: Inicie o servidor da aplicação:
uvicorn main:app --reload

Passo 3: Acesse no navegador:
http://127.0.0.1:8000

========================================================================
NOTAS DE TROUBLESHOOTING:

- Erro "FileNotFoundError" no Whisper: Você não instalou o FFmpeg.
- Erro de Conexão Llama: O aplicativo Ollama não está rodando.
- Erro de Memória (CUDA): Se tiver GPU NVIDIA, instale o PyTorch com suporte CUDA.
  Caso contrário, o sistema rodará na CPU (mais lento, mas funcional).
  ========================================================================
