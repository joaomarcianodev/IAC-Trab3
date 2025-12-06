from fastapi import FastAPI, UploadFile, File
from ia_service import processar_audio_com_ia
import shutil
import os

app = FastAPI(
    title="Detector de Ofensas (Whisper + BERT)",
    description="API que usa OpenAI Whisper para transcrever e BERT para analisar sentimento."
)

@app.get("/")
def home():
    return {"status": "IA Online. Acesse /docs para testar."}

@app.post("/analisar")
def analisar_audio(arquivo: UploadFile = File(...)):
    """
    Endpoint que recebe um arquivo de áudio (WAV, MP3, M4A...),
    salva temporariamente, processa na IA e retorna o resultado.
    """
    # Cria um nome temporário seguro
    nome_arquivo_temp = f"temp_{arquivo.filename}"
    
    try:
        # 1. Salva o arquivo recebido no disco
        with open(nome_arquivo_temp, "wb") as buffer:
            shutil.copyfileobj(arquivo.file, buffer)
            
        # 2. Chama o serviço de IA
        resultado = processar_audio_com_ia(nome_arquivo_temp)
        
        return resultado

    finally:
        # 3. Limpeza: Remove o arquivo temporário após o uso
        if os.path.exists(nome_arquivo_temp):
            os.remove(nome_arquivo_temp)

if __name__ == "__main__":
    import uvicorn
    # Roda o servidor na porta 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)