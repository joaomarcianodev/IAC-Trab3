import whisper
from transformers import pipeline
import warnings

# Limpa avisos do terminal
warnings.filterwarnings("ignore")

print("--- INICIALIZANDO SISTEMAS DE IA ---")

# 1. Carrega o BERT 
print("1/2 Carregando BERT (Análise de Sentimento)...")
try:
    analisador_sentimento = pipeline(
        "sentiment-analysis", 
        model="nlptown/bert-base-multilingual-uncased-sentiment"
    )
except Exception as e:
    print(f"ERRO ao carregar BERT: {e}")
    analisador_sentimento = None

# 2. Carrega o Whisper (AQUI ESTÁ A MUDANÇA)
# Trocamos "base" por "small". Se ainda ficar ruim, tente "medium".
print("2/2 Carregando Whisper (Modelo 'small' - Melhor para música)...")
try:
    # Opções: "base" (rápido/ruim), "small" (bom), "medium" (excelente/lento)
    modelo_transcricao = whisper.load_model("small") 
except Exception as e:
    print(f"ERRO CRÍTICO ao carregar Whisper: {e}")
    modelo_transcricao = None

print("--- SISTEMAS PRONTOS ---")

def processar_audio_com_ia(caminho_arquivo):
    if not modelo_transcricao or not analisador_sentimento:
        return {"sucesso": False, "erro": "Modelos de IA não estão ativos."}

    try:
        # --- ETAPA 1: TRANSCRIÇÃO ---
        print(f"Processando áudio (Isso pode demorar um pouco mais agora)...")
        
        # O Whisper tenta "ignorar" o instrumental, mas ele precisa de poder de processamento
        resultado_whisper = modelo_transcricao.transcribe(caminho_arquivo, fp16=False)
        texto_transcrito = resultado_whisper["text"]
        idioma_detectado = resultado_whisper["language"]

        print(f"Texto detectado: '{texto_transcrito}'")

        # --- ETAPA 2: ANÁLISE ---
        resultado_bert = analisador_sentimento(texto_transcrito)[0]
        label = resultado_bert['label']
        confianca = resultado_bert['score']

        estrelas = int(label.split()[0])
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