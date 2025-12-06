import whisper
from transformers import pipeline
import warnings
import numpy as np
import time
import torch
import gc
import math
import re

warnings.filterwarnings("ignore")

print("--- SERVIÇO DE IA INICIADO ---")

# Variáveis Globais
modelo_atual_nome = None
modelo_whisper_carregado = None
analisador_sentimento = None

def get_logger():
    logs = []
    def log(msg):
        print(f"[IA] {msg}")
        logs.append(str(msg))
    return logs, log

def carregar_bert(log_func):
    global analisador_sentimento
    if analisador_sentimento is None:
        log_func("Carregando modelo BERT...")
        try:
            analisador_sentimento = pipeline(
                "sentiment-analysis", 
                model="nlptown/bert-base-multilingual-uncased-sentiment"
            )
        except Exception as e:
            log_func(f"ERRO BERT: {e}")
    return analisador_sentimento

def carregar_whisper(tamanho_escolhido, log_func):
    global modelo_atual_nome, modelo_whisper_carregado

    if modelo_whisper_carregado is not None and modelo_atual_nome == tamanho_escolhido:
        return modelo_whisper_carregado

    if modelo_whisper_carregado is not None:
        log_func(f"Limpando memória...")
        del modelo_whisper_carregado
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_func(f"Carregando Whisper '{tamanho_escolhido}'...")
    try:
        modelo_whisper_carregado = whisper.load_model(tamanho_escolhido)
        modelo_atual_nome = tamanho_escolhido
        return modelo_whisper_carregado
    except Exception as e:
        log_func(f"ERRO WHISPER: {e}")
        return None

# --- NOVA LÓGICA DE ANÁLISE ---

def verificar_palavras_proibidas(texto, lista_palavras):
    """Verifica se alguma palavra da lista está no texto."""
    if not lista_palavras or lista_palavras == [""]:
        return []
    
    texto_lower = texto.lower()
    encontradas = []
    
    for palavra in lista_palavras:
        palavra = palavra.strip().lower()
        if palavra and palavra in texto_lower:
            encontradas.append(palavra)
            
    return encontradas

def analisar_com_bert(texto_completo, estrategia="completa"):
    """
    Estratégia 'completa': Faz a média de tudo.
    Estratégia 'segmentada': Se encontrar UMA frase tóxica, condena.
    """
    if not texto_completo or len(str(texto_completo).strip()) < 2:
        return {"eh_ofensivo": False, "score": 0.0, "detalhes": "Texto vazio"}

    texto_completo = str(texto_completo)
    
    # Divide em frases (por ponto final, exclamação, interrogação ou quebra de linha)
    # Isso serve para a análise segmentada e para quebrar blocos para a completa
    frases = re.split(r'[.!?\n]+', texto_completo)
    frases = [f.strip() for f in frases if len(f.strip()) > 5] # Remove frases muito curtas

    scores_negativos = []
    frases_toxicas_detectadas = []

    for frase in frases:
        # Corta frase se for maior que 512 chars (limite BERT)
        frase_cut = frase[:512]
        
        if analisador_sentimento:
            try:
                resultado = analisador_sentimento(frase_cut)[0]
                estrelas = int(resultado['label'].split()[0])
                
                # Mapa: 1 estrela = 1.0 (Ruim), 5 estrelas = 0.0 (Bom)
                mapa = {1: 1.0, 2: 0.8, 3: 0.5, 4: 0.2, 5: 0.0}
                score = mapa.get(estrelas, 0.0)
                scores_negativos.append(score)

                # Se a estratégia for SEGMENTADA, verificamos se essa frase específica é muito ruim
                if estrategia == "segmentada":
                    # Se tiver score alto (>= 0.8) consideramos essa parte tóxica
                    if score >= 0.8:
                        frases_toxicas_detectadas.append(frase)

            except:
                continue

    if not scores_negativos:
        return {"eh_ofensivo": False, "score": 0.0, "detalhes": "Sem dados"}

    # --- DECISÃO BASEADA NA ESTRATÉGIA ---
    
    media_global = float(np.mean(scores_negativos))
    if math.isnan(media_global): media_global = 0.0

    if estrategia == "segmentada":
        # Na segmentada, se houver frases tóxicas, é ofensivo, não importa a média
        eh_ofensivo = len(frases_toxicas_detectadas) > 0
        score_final = media_global # Mantemos a média só para mostrar no gráfico
        detalhe = f"Encontradas {len(frases_toxicas_detectadas)} frases críticas."
    else:
        # Na completa (padrão), vale a média
        eh_ofensivo = media_global > 0.4
        score_final = media_global
        detalhe = "Média global de negatividade."

    return {
        "eh_ofensivo": bool(eh_ofensivo),
        "score": round(score_final * 100, 2),
        "detalhes": detalhe,
        "frases_toxicas": frases_toxicas_detectadas
    }

def processar_audio_com_ia(caminho_arquivo, modelo_escolhido, palavras_input, estrategia_bert):
    logs, log = get_logger()
    tempo_inicio_total = time.time()
    
    try:
        bert = carregar_bert(log)
        whisper_model = carregar_whisper(modelo_escolhido, log)

        if not whisper_model:
            return {"sucesso": False, "erro": "Falha no Whisper", "logs": logs}

        log("Transcrevendo áudio...")
        inicio_transcricao = time.time()
        
        resultado = whisper_model.transcribe(
            caminho_arquivo, 
            fp16=False, 
            language="pt",
            no_speech_threshold=0.6
        )
        
        fim_transcricao = time.time()
        texto_final = str(resultado["text"])
        duracao_transcricao = fim_transcricao - inicio_transcricao
        
        # --- ETAPA 1: PALAVRAS PROIBIDAS ---
        log(f"Verificando lista proibida...")
        lista_proibida = [p.strip() for p in palavras_input.split(',')]
        palavras_encontradas = verificar_palavras_proibidas(texto_final, lista_proibida)
        
        tem_palavra_proibida = len(palavras_encontradas) > 0

        # --- ETAPA 2: ANÁLISE BERT (Completa ou Segmentada) ---
        log(f"Analisando sentimento (Estratégia: {estrategia_bert})...")
        inicio_analise = time.time()
        
        resultado_bert = analisador_sentimento_global = analisart_bert = analisar_com_bert(texto_final, estrategia_bert)
        
        fim_analise = time.time()
        duracao_analise = fim_analise - inicio_analise

        # --- VEREDITO FINAL ---
        # É ofensivo se: Tiver palavra proibida OU O BERT disser que é ofensivo
        veredito_final_ofensivo = tem_palavra_proibida or resultado_bert["eh_ofensivo"]
        
        tempo_total = time.time() - tempo_inicio_total

        return {
            "sucesso": True,
            "transcricao": texto_final,
            "analise": {
                "eh_ofensivo": veredito_final_ofensivo,
                "motivo_palavras": palavras_encontradas,
                "motivo_bert": resultado_bert["detalhes"],
                "nivel_toxidade": resultado_bert["score"],
                "frases_criticas": resultado_bert.get("frases_toxicas", [])
            },
            "metricas": {
                "tempo_transcricao": f"{duracao_transcricao:.2f}s",
                "tempo_analise": f"{duracao_analise:.2f}s",
                "tempo_total": f"{tempo_total:.2f}s",
                "modelo_usado": modelo_escolhido
            },
            "logs": logs
        }

    except Exception as e:
        import traceback
        erro_detalhado = traceback.format_exc()
        log(f"ERRO FATAL: {erro_detalhado}")
        return {"sucesso": False, "erro": str(e), "logs": logs}