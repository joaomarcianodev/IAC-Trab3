import whisper
from transformers import pipeline
import warnings

# Ignora avisos irrelevantes do processador para limpar o terminal
warnings.filterwarnings("ignore")

print("--- INICIALIZANDO SISTEMAS DE IA ---")

# 1. Carrega o BERT (Análise de Sentimento - Multilíngue)
# Este modelo classifica textos de 1 estrela (muito ruim) a 5 estrelas (muito bom)
print("1/2 Carregando BERT (Análise de Sentimento)...")
try:
    analisador_sentimento = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
except Exception as e:
    print(f"ERRO ao carregar BERT: {e}")
    analisador_sentimento = None

# 2. Carrega o Whisper (Transcrição de Áudio)
# O modelo "base" é rápido e preciso. Se quiser mais velocidade e menos precisão, use "tiny".
print("2/2 Carregando Whisper (Reconhecimento de Fala)...")
try:
    modelo_transcricao = whisper.load_model("base")
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar Whisper (Verifique o FFmpeg): {e}")
    modelo_transcricao = None

print("--- SISTEMAS PRONTOS ---")

def processar_audio_com_ia(caminho_arquivo):
    """
    Recebe o caminho de um arquivo de áudio, transcreve e analisa se é ofensivo.
    """
    if not modelo_transcricao or not analisador_sentimento:
        return {"sucesso": False, "erro": "Modelos de IA não estão ativos."}

    try:
        # --- ETAPA 1: TRANSCRIÇÃO (Whisper) ---
        print(f"Processando áudio: {caminho_arquivo}")
        
        # fp16=False é importante para evitar erros em CPUs
        resultado_whisper = modelo_transcricao.transcribe(caminho_arquivo, fp16=False)
        texto_transcrito = resultado_whisper["text"]
        idioma_detectado = resultado_whisper["language"]

        print(f"Texto detectado: '{texto_transcrito}'")

        # --- ETAPA 2: ANÁLISE DE SENTIMENTO (BERT) ---
        # O BERT retorna algo como: [{'label': '1 star', 'score': 0.95}]
        resultado_bert = analisador_sentimento(texto_transcrito)[0]
        label = resultado_bert['label']
        confianca = resultado_bert['score']

        # Extrai o número de estrelas (o primeiro caractere da label)
        estrelas = int(label.split()[0])

        # Lógica de Classificação:
        # 1 ou 2 estrelas = Negativo/Ofensivo
        # 3, 4 ou 5 estrelas = Neutro/Positivo
        eh_ofensivo = estrelas <= 2

        return {
            "sucesso": True,
            "transcricao": texto_transcrito,
            "idioma": idioma_detectado,
            "analise_ia": {
                "classificacao_estrelas": label,
                "confianca": f"{confianca:.2f}",
                "veredito": "OFENSIVO/NEGATIVO" if eh_ofensivo else "POSITIVO/NEUTRO"
            },
            "eh_ofensivo": eh_ofensivo
        }

    except Exception as e:
        return {"sucesso": False, "erro": f"Falha no processamento: {str(e)}"}