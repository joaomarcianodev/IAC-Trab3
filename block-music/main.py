from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ia_service import processar_audio_stream
import shutil
import os

app = FastAPI()

# --- Configurações Iniciais ---

# Serve arquivos estáticos (CSS, JS, Imagens) a partir da rota /static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configura o diretório de templates para renderização HTML (Jinja2)
templates = Jinja2Templates(directory="templates")

# Controle de Concorrência Global
# Impede que múltiplas análises pesadas rodem simultaneamente e travem o servidor
ESTADO_SERVIDOR = {"ocupado": False}

# --- Rotas de Navegação (Frontend) ---

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """Renderiza a página principal da aplicação."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc", response_class=HTMLResponse)
def documentacao(request: Request):
    """Renderiza a página de documentação técnica."""
    return templates.TemplateResponse("doc.html", {"request": request})

# --- Rotas da API (Backend) ---

@app.get("/status")
def get_status():
    """
    Retorna o estado atual do servidor.
    Utilizado pelo frontend (Polling) para saber quando o servidor foi liberado após um cancelamento.
    """
    return {"ocupado": ESTADO_SERVIDOR["ocupado"]}

@app.post("/analisar")
async def analisar(
    arquivo: UploadFile = File(...),
    modelo: str = Form("base"),
    palavras_proibidas: str = Form(""),
    motor: str = Form("bert"),      
    estrategia: str = Form("completa") 
):
    """
    Endpoint principal de processamento.
    Recebe o áudio e parâmetros, e retorna um StreamingResponse (SSE) com logs em tempo real.
    """
    
    # 1. Verificação de Disponibilidade
    if ESTADO_SERVIDOR["ocupado"]:
        return JSONResponse(status_code=503, content={"sucesso": False, "erro": "Servidor ocupado. Aguarde finalizar a tarefa atual."})

    # 2. Salvamento Temporário do Arquivo
    temp_filename = f"temp_{arquivo.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"sucesso": False, "erro": f"Erro ao salvar arquivo: {e}"})

    # 3. Definição do Iterador Assíncrono
    # Wrapper para garantir o bloqueio e desbloqueio seguro do servidor
    async def iterador_seguro():
        try:
            ESTADO_SERVIDOR["ocupado"] = True # Bloqueia novos acessos
            
            # Delega o processamento pesado para o serviço de IA
            for chunk in processar_audio_stream(temp_filename, modelo, palavras_proibidas, motor, estrategia):
                yield chunk
                
        finally:
            # Bloco Finally: Executa sempre, mesmo se houver erro ou cancelamento pelo cliente
            ESTADO_SERVIDOR["ocupado"] = False # Libera o servidor
            
            # Limpeza do arquivo temporário
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass 

    # Retorna o fluxo de dados contínuo (Server-Sent Events)
    return StreamingResponse(iterador_seguro(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Inicia o servidor Uvicorn na porta 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)