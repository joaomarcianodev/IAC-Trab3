from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from ia_service import processar_audio_stream
import shutil
import os

app = FastAPI()

# 1. Configura a pasta de arquivos estáticos (CSS, JS, Imagens)
app.mount("/static", StaticFiles(directory="static"), name="static")

# 2. Configura a pasta de templates (HTML)
templates = Jinja2Templates(directory="templates")

# Variável global para controle de estado do servidor
# Isso impede que duas análises pesadas rodem ao mesmo tempo
ESTADO_SERVIDOR = {"ocupado": False}

# --- ROTAS DE PÁGINAS ---

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    """
    Renderiza a página inicial.
    O Jinja2 injeta o 'request' para que url_for funcione se necessário.
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/doc", response_class=HTMLResponse)
def documentacao(request: Request):
    """
    Página de documentação standalone (acessível via URL direta).
    """
    return templates.TemplateResponse("doc.html", {"request": request})

# A rota /about foi removida pois agora é um modal na base.html

# --- ROTAS DE API ---

@app.get("/status")
def get_status():
    """
    Endpoint para o Frontend verificar se o servidor está livre.
    Usado pelo polling do botão 'Cancelar'.
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
    Recebe o arquivo e inicia o processamento em Streaming.
    """
    # 1. Bloqueio de concorrência
    if ESTADO_SERVIDOR["ocupado"]:
        return JSONResponse(status_code=503, content={"sucesso": False, "erro": "Servidor ocupado com outra análise."})

    # 2. Salva o arquivo temporariamente no disco
    temp_filename = f"temp_{arquivo.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
    except Exception as e:
        return JSONResponse(status_code=500, content={"sucesso": False, "erro": f"Erro ao salvar arquivo: {e}"})

    # 3. Gerador assíncrono para Streaming
    async def iterador_seguro():
        try:
            ESTADO_SERVIDOR["ocupado"] = True # Bloqueia
            
            # Chama a função geradora do ia_service
            for chunk in processar_audio_stream(temp_filename, modelo, palavras_proibidas, motor, estrategia):
                yield chunk
                
        finally:
            # Garante liberação e limpeza mesmo em caso de erro ou cancelamento
            ESTADO_SERVIDOR["ocupado"] = False # Libera
            
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except:
                    pass # Ignora erros de deleção

    # Retorna a resposta que envia dados aos poucos (Server-Sent Events style)
    return StreamingResponse(iterador_seguro(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    # Inicia o servidor na porta 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)