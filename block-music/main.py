from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from ia_service import processar_audio_com_ia
import shutil
import os
import signal
import time
import traceback

app = FastAPI()

# --- FUNÇÃO PARA MATAR O PROCESSO ---
def matar_servidor():
    print("🛑 Encerrando servidor via comando remoto...")
    time.sleep(1) # Espera 1 seg para garantir que a resposta chegue ao cliente
    
    # Envia sinal de interrupção (Simula Ctrl+C)
    # Funciona bem no Windows e Linux
    os.kill(os.getpid(), signal.SIGINT)

@app.get("/", response_class=HTMLResponse)
def home():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Erro: Arquivo index.html não encontrado.</h1>"

@app.post("/shutdown")
def desligar(background_tasks: BackgroundTasks):
    """
    Rota para desligar o servidor remotamente.
    Agenda a morte do processo para logo após o return.
    """
    background_tasks.add_task(matar_servidor)
    return JSONResponse(content={"sucesso": True, "mensagem": "Servidor desligando em 1 segundo..."})

@app.post("/analisar")
async def analisar(
    arquivo: UploadFile = File(...),
    modelo: str = Form("base"),
    palavras_proibidas: str = Form(""),
    estrategia: str = Form("completa")
):
    temp_filename = f"temp_{arquivo.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
        
        print(f"Pedido: Modelo={modelo} | Estratégia={estrategia}")
        
        resultado = processar_audio_com_ia(
            temp_filename, 
            modelo, 
            palavras_proibidas, 
            estrategia
        )
        return JSONResponse(content=resultado)

    except Exception as e:
        erro_msg = traceback.format_exc()
        return JSONResponse(
            status_code=200, 
            content={"sucesso": False, "erro": str(e), "logs": [erro_msg]}
        )
        
    finally:
        if os.path.exists(temp_filename):
            try: os.remove(temp_filename)
            except: pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)