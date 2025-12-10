import whisper
import warnings
import time
import torch
import gc
import requests
import json
from transformers import pipeline
import numpy as np
import re
import math

# Ignora avisos não críticos das bibliotecas de IA para manter o log limpo
warnings.filterwarnings("ignore")

print("--- SERVIÇO DE IA INICIADO ---")

# --- Variáveis Globais de Cache ---
# Mantém os modelos na memória RAM/VRAM para evitar recarregamento a cada requisição
modelo_atual_nome = None
modelo_whisper_carregado = None
analisador_bert = None

# ==============================================================================
# GERENCIAMENTO DE MODELOS
# ==============================================================================

def carregar_whisper_dinamico(tamanho):
    """
    Gerencia o carregamento do modelo Whisper.
    Se o modelo solicitado já estiver em memória, reutiliza-o.
    Caso contrário, limpa a memória (GC/CUDA) e carrega o novo.
    """
    global modelo_atual_nome, modelo_whisper_carregado
    
    # Verifica se o modelo já está carregado
    if modelo_whisper_carregado and modelo_atual_nome == tamanho:
        return modelo_whisper_carregado, False # False indica que não houve novo carregamento

    # Limpeza de memória antes de carregar um novo modelo (Crítico para GPUs com pouca VRAM)
    if modelo_whisper_carregado:
        del modelo_whisper_carregado
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    try:
        model = whisper.load_model(tamanho)
        modelo_whisper_carregado = model
        modelo_atual_nome = tamanho
        return model, True # True indica que um novo modelo foi carregado
    except Exception as e:
        raise Exception(f"Erro ao carregar Whisper: {e}")

def carregar_bert_dinamico():
    """
    Carrega o pipeline do BERT (análise de sentimento) via HuggingFace.
    Utiliza o modelo multilingue 'nlptown' (escala de 1 a 5 estrelas).
    """
    global analisador_bert
    if analisador_bert: return analisador_bert
    try:
        # Define explicitamente a tarefa para evitar ambiguidade no Transformers
        analisador_bert = pipeline(
            task="sentiment-analysis", #type: ignore
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        ) # type: ignore
        return analisador_bert
    except Exception as e:
        raise Exception(f"Erro ao carregar BERT: {e}")

# ==============================================================================
# MOTORES DE ANÁLISE (INTELIGÊNCIA)
# ==============================================================================

import numpy as np

def analisar_com_bert(texto):
    """
    Realiza a análise usando BERT (Sentimento adaptado para Toxicidade).
    Configuração: REGUA NEUTRA (0.5)
    """
    bert = carregar_bert_dinamico()
    
    if not texto or len(texto) < 2:
        return {"eh_ofensivo": False, "score": 0, "detalhes": "Texto vazio"}

    # Mantemos a leitura em bloco (contexto) para evitar falsos positivos por falta de nexo,
    # mas limitamos o tamanho para não estourar o modelo.
    limite_chars = 2000
    texto_analise = texto[:limite_chars] 

    try:
        # Pega o resultado bruto do modelo
        res = bert(texto_analise, truncation=True, max_length=512)[0]
        
        # Ex: {'label': '1 star', 'score': 0.98}
        label = res['label']
        estrelas = int(label.split()[0])
        confianca = res['score'] 
        
        # --- CALIBRAGEM NEUTRA ---
        # 1 estrela = 1.0 (Ofensa/Negatividade Máxima)
        # 2 estrelas = 0.5 (Zona cinzenta/Reclamação)
        # 3, 4, 5 = 0.0 (Neutro ou Positivo - ignoramos)
        mapa_toxicidade = {
            1: 1.0, 
            2: 0.5, 
            3: 0.0, 
            4: 0.0, 
            5: 0.0  
        }
        
        score_calculado = mapa_toxicidade.get(estrelas, 0.0)
        
        # Ajuste de Incerteza:
        # Se o modelo não tem certeza (confiança < 70%), derrubamos o score pela metade
        # para evitar condenar o réu em caso de dúvida.
        if confianca < 0.7:
            score_calculado = score_calculado * 0.5

        # A "Régua": Se o score final for MAIOR que 0.5, é considerado ofensivo.
        # Isso significa que:
        # - "1 star" com alta confiança (1.0) -> BLOQUEIA
        # - "2 stars" com alta confiança (0.5) -> NÃO BLOQUEIA (fica no limite)
        # - "1 star" com baixa confiança (0.5) -> NÃO BLOQUEIA (benefício da dúvida)
        
        eh_ofensivo = score_calculado > 0.5

        return {
            "eh_ofensivo": eh_ofensivo,
            "score": round(score_calculado * 100, 2), # Retorna em porcentagem (0 a 100)
            "detalhes": f"Classificação: {label} (Confiança: {confianca:.2f})",
            "frases_criticas": [texto[:100] + "..."] if eh_ofensivo else []
        }

    except Exception as e:
        print(f"Erro na análise BERT: {e}")
        # Em caso de erro, na dúvida, libera (fail-open) ou bloqueia (fail-close)?
        # Geralmente fail-open (False) é melhor para não travar o sistema.
        return {"eh_ofensivo": False, "score": 0, "detalhes": "Erro interno"}

def analisar_com_ollama(texto):
    """
    Realiza a análise usando Llama 3.2 via API local do Ollama (Abordagem Generativa/Contextual).
    O prompt instrui o modelo a agir como um moderador e retornar JSON estrito.
    """
    url = "http://localhost:11434/api/generate"
    
    system_prompt = """
    Você é um classificador de conteúdo ofensivo brasileiro.
    Analise o texto e responda APENAS este JSON:
    {
        "eh_ofensivo": boolean, 
        "score_ofensivo": number (0 a 100), 
        "motivo_breve": "string", 
        "frases_criticas": ["lista", "com", "trechos", "exatos", "do", "texto", "que", "sao", "ofensivos"]
    }
    """
    
    try:
        payload = {
            "model": "llama3.2",
            "prompt": f"{system_prompt}\nTexto para analisar: {texto}",
            "stream": False,
            "format": "json" # Força o Ollama a estruturar a saída
        }
        resp = requests.post(url, json=payload)
        
        if resp.status_code == 200:
            dados = json.loads(resp.json()['response'])
            return {
                "eh_ofensivo": dados.get("eh_ofensivo", False),
                "score": dados.get("score_ofensivo", 0),
                "detalhes": dados.get("motivo_breve", "Análise Llama"),
                "frases_criticas": dados.get("frases_criticas", [])
            }
        else:
            # Fallback seguro em caso de erro na API
            return {
                "eh_ofensivo": False, 
                "score": 0, 
                "detalhes": f"Erro API Ollama: {resp.status_code}", 
                "frases_criticas": []
            }
            
    except Exception as e:
        return {
            "eh_ofensivo": False, 
            "score": 0, 
            "detalhes": f"Erro Conexão Ollama: {str(e)}", 
            "frases_criticas": []
        }

# ==============================================================================
# PIPELINE PRINCIPAL (STREAMING)
# ==============================================================================

def processar_audio_stream(caminho_arquivo, modelo_whisper_nome, palavras_input, motor_analise):
    """
    Pipeline principal executado como um Gerador (Generator).
    
    Funcionamento:
    1. Yields strings no formato "LOG: mensagem" para atualização em tempo real no frontend.
    2. Yield final no formato "RESULT: {json}" contendo todos os dados processados.
    
    Args:
        caminho_arquivo (str): Path do arquivo de áudio salvo temporariamente.
        modelo_whisper_nome (str): 'tiny', 'base', 'small', 'medium', 'large'.
        palavras_input (str): String com palavras proibidas separadas por vírgula.
        motor_analise (str): 'bert' ou 'ollama'.
    """
    logs_acumulados = []
    tempo_inicio = time.time()

    # Função auxiliar para formatar e guardar logs
    def enviar_log(msg):
        logs_acumulados.append(msg)
        return f"LOG:{msg}\n"

    try:
        # 1. Carregamento do Whisper
        yield enviar_log(f"Carregando Whisper '{modelo_whisper_nome}'...")
        whisper_model, carregou_novo = carregar_whisper_dinamico(modelo_whisper_nome)
        if carregou_novo: yield enviar_log("Modelo carregado na memória.")

        # 2. Transcrição (Speech-to-Text)
        yield enviar_log("Transcrevendo áudio...")
        start_trans = time.time()
        # fp16=False garante compatibilidade com CPUs
        res = whisper_model.transcribe(caminho_arquivo, fp16=False, language="pt", no_speech_threshold=0.6)
        texto = str(res["text"]).strip()
        tempo_trans = time.time() - start_trans
        yield enviar_log(f"Transcrição concluída em {tempo_trans:.2f}s.")

        if not texto:
            yield f"RESULT:{json.dumps({'sucesso': False, 'erro': 'Áudio vazio ou inaudível'})}\n"
            return

        # 3. Verificação de Palavras Proibidas (Regex Simples)
        yield enviar_log("Verificando palavras proibidas...")
        lista_bad = [p.strip().lower() for p in palavras_input.split(',') if p.strip()]
        encontradas = [p for p in lista_bad if p in texto.lower()]
        tem_bad = len(encontradas) > 0

        # 4. Análise de Inteligência (NLP)
        start_analise = time.time()
        
        # Inicialização padrão para garantir tipagem
        analise_res = {
            "eh_ofensivo": False, 
            "score": 0, 
            "detalhes": "Não analisado", 
            "frases_criticas": []
        }
        nome_modelo_final = ""
        
        if motor_analise == "ollama":
            yield enviar_log("Consultando Llama 3.2 (Análise Contextual)...")
            analise_res = analisar_com_ollama(texto)
            nome_modelo_final = "Llama 3.2 (Contextual)"
        else:
            yield enviar_log(f"Processando BERT...")
            analise_res = analisar_com_bert(texto)
            nome_modelo_final = f"BERT"

        tempo_analise = time.time() - start_analise
        yield enviar_log(f"Análise concluída em {tempo_analise:.2f}s.")

        # 5. Consolidação dos Resultados
        # O veredito é positivo se houver palavras proibidas OU se a IA detectar ofensa
        veredito = tem_bad or analise_res.get("eh_ofensivo", False)
        
        score_final = analise_res.get("score", 0)
        if tem_bad: score_final = 100 # Palavra proibida força toxicidade máxima

        tempo_total = time.time() - tempo_inicio
        yield enviar_log("Finalizando processo...")

        resultado_final = {
            "sucesso": True,
            "transcricao": texto,
            "analise": {
                "eh_ofensivo": veredito,
                "motivo_palavras": encontradas,
                "motivo_ia": analise_res.get("detalhes", "Sem detalhes"),
                "score_ofensivo": score_final,
                "frases_criticas": analise_res.get("frases_criticas", [])
            },
            "metricas": {
                "tempo_transcricao": f"{tempo_trans:.2f}s",
                "tempo_analise": f"{tempo_analise:.2f}s",
                "tempo_total": f"{tempo_total:.2f}s",
                "modelo_usado": f"Whisper {modelo_whisper_nome} + {nome_modelo_final}"
            },
            "logs": logs_acumulados
        }

        # Envia o payload final para o frontend encerrar a conexão
        yield f"RESULT:{json.dumps(resultado_final)}\n"

    except Exception as e:
        import traceback
        err = traceback.format_exc()
        yield enviar_log(f"ERRO CRÍTICO: {str(e)}")
        # Retorna o erro estruturado para o frontend tratar
        yield f"RESULT:{json.dumps({'sucesso': False, 'erro': str(e), 'logs': logs_acumulados})}\n"