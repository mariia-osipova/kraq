#!/usr/bin/env python3
"""
Script para analizar la tokenización de los campos 'rationale' y 'response' en JSON
y ayudar a solucionar problemas con el procesamiento de logprobs.
"""
import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
import tiktoken
import numpy as np
from pathlib import Path
from loguru import logger

# Cargar variables de entorno
load_dotenv()

# Configuración para vLLM local
VLLM_BASE_URL = "http://localhost:8000/v1"

class TokenInfo:
    def __init__(self, index, token, token_text):
        self.index = index
        self.token = token
        self.token_text = token_text
        
    def __str__(self):
        return f"[{self.index}] ID: {self.token}, Text: '{self.token_text}'"

async def analyze_tokens(model_name="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"):
    """
    Analiza cómo el tokenizador divide los campos 'rationale' y 'response' en el formato JSON.
    """
    # Crear cliente para vLLM
    client = AsyncOpenAI(
        base_url=VLLM_BASE_URL,
        api_key="not-needed"
    )
    
    # Diferentes variaciones a probar
    test_cases = [
        # Caso 1: JSON simple
        """{
  "rationale": "This is a test rationale.",
  "response": "This is a test response."
}""",
        # Caso 2: JSON con texto más largo
        """{
  "rationale": "Esto es un ejemplo de rationale más largo para ver cómo se tokeniza cuando hay más contenido.",
  "response": "Y esta es una respuesta de ejemplo más larga también."
}""",
        # Caso 3: JSON con orden invertido
        """{
  "response": "Response field first.",
  "rationale": "Rationale field second."
}""",
        # Caso 4: JSON con espacios variables
        """{"rationale":"Compact formatting.","response":"No extra spaces."}""",
        # Caso 5: JSON con campos adicionales
        """{
  "rationale": "With extra fields.",
  "extra": "Some extra field.",
  "response": "With response after extra field."
}"""
    ]
    
    # Analizar cada caso de prueba
    for i, test_json in enumerate(test_cases, 1):
        print(f"\n{'='*80}\nCASO DE PRUEBA #{i}:\n{test_json}\n{'='*80}")
        
        try:
            # Obtener la tokenización del modelo
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": f"Please tokenize this JSON: {test_json}"}],
                max_tokens=1,  # Solo necesitamos iniciar la generación para ver tokens
                logprobs=True,
                echo=True  # Para obtener la tokenización del prompt
            )
            
            # Extraer información de tokens
            tokens_info = []
            if hasattr(response.choices[0], 'logprobs') and hasattr(response.choices[0].logprobs, 'content'):
                for idx, token_info in enumerate(response.choices[0].logprobs.content):
                    if hasattr(token_info, 'token') and token_info.token is not None:
                        token_text = token_info.token
                        tokens_info.append(TokenInfo(idx, token_info.token_id, token_text))
            
            # Analizar tokens en busca de 'rationale' y 'response'
            rationale_found = False
            response_found = False
            print("\nTOKENS RELEVANTES:")
            
            # Buscar patrones que podrían indicar estos campos
            patterns = ["rational", "ration", "response", "rationale", "resp"]
            
            for token_info in tokens_info:
                token_text = token_info.token_text.lower()
                found_pattern = False
                
                for pattern in patterns:
                    if pattern in token_text:
                        print(f"{token_info} - MATCH FOR '{pattern}'")
                        found_pattern = True
                        
                        if "rational" in token_text or "ration" in token_text:
                            rationale_found = True
                        if "resp" in token_text:
                            response_found = True
                            
                # También imprimir tokens adyacentes para contexto
                if found_pattern and token_info.index > 0:
                    prev_idx = token_info.index - 1
                    if prev_idx < len(tokens_info):
                        print(f"  Prev token: {tokens_info[prev_idx]}")
                        
                if found_pattern and token_info.index < len(tokens_info) - 1:
                    next_idx = token_info.index + 1
                    if next_idx < len(tokens_info):
                        print(f"  Next token: {tokens_info[next_idx]}")
            
            if not rationale_found and not response_found:
                print("No se encontraron tokens para 'rationale' ni 'response'.")
                print("\nTODOS LOS TOKENS:")
                for token_info in tokens_info:
                    print(token_info)
            
            # Analizar JSON original para encontrar índices
            rationale_index = test_json.find('"rationale"')
            response_index = test_json.find('"response"')
            
            print(f"\nPosiciones en el texto original:")
            print(f"'rationale' comienza en el índice {rationale_index}")
            print(f"'response' comienza en el índice {response_index}")
            
            # Análisis de completitud
            print("\nANÁLISIS COMPLETO:")
            json_tokens = ''.join([t.token_text for t in tokens_info])
            print(f"Texto reconstruido: {json_tokens}")
            
            # Busqueda de substrings específicos
            key_strings = ['"rationale"', '"response"', 'rationale', 'response']
            for key in key_strings:
                print(f"Buscando '{key}' en tokens: {'✓' if key in json_tokens else '✗'}")
                
                substring_found = False
                substring_tokens = []
                for i, token_info in enumerate(tokens_info):
                    token_text = token_info.token_text
                    if key in token_text:
                        substring_found = True
                        substring_tokens.append(token_info)
                        
                if substring_found:
                    print(f"  Tokens que contienen '{key}':")
                    for token_info in substring_tokens:
                        print(f"  {token_info}")
                else:
                    print(f"  '{key}' no se encontró completo en ningún token, puede estar dividido")
                    
            print("\nRECOMENDACIÓN PARA LA BÚSQUEDA DE TOKENS:")
            recommendation = []
            
            # Analizar los tokens relevantes encontrados para generar recomendación
            if rationale_found:
                rational_tokens = [t for t in tokens_info if "rational" in t.token_text.lower() or "ration" in t.token_text.lower()]
                if rational_tokens:
                    recommendation.append(f"Para 'rationale', buscar: {[t.token_text for t in rational_tokens]}")
            
            if response_found:
                response_tokens = [t for t in tokens_info if "resp" in t.token_text.lower()]
                if response_tokens:
                    recommendation.append(f"Para 'response', buscar: {[t.token_text for t in response_tokens]}")
            
            if recommendation:
                for rec in recommendation:
                    print(rec)
            else:
                print("No se pudieron generar recomendaciones específicas. Utilizar un método basado en la posición relativa en el texto.")
            
        except Exception as e:
            print(f"Error en análisis: {e}")
    
    # Prueba adicional con tokenización directa usando tiktoken
    print("\n\nPRUEBA CON TIKTOKEN:")
    try:
        # Intentar con diferentes codificaciones de tokenización
        for encoding_name in ["cl100k_base", "p50k_base"]:
            try:
                enc = tiktoken.get_encoding(encoding_name)
                print(f"\nUsando codificación: {encoding_name}")
                
                for i, test_json in enumerate(test_cases, 1):
                    tokens = enc.encode(test_json)
                    token_texts = [enc.decode([token]) for token in tokens]
                    
                    print(f"\nCaso #{i} con tiktoken:")
                    
                    # Buscar tokens relevantes
                    for j, (token, text) in enumerate(zip(tokens, token_texts)):
                        for pattern in ["rational", "ration", "response", "resp"]:
                            if pattern in text.lower():
                                print(f"[{j}] ID: {token}, Text: '{text}' - MATCH FOR '{pattern}'")
                                
                                # Imprimir contexto
                                if j > 0:
                                    prev_token = tokens[j-1]
                                    prev_text = token_texts[j-1]
                                    print(f"  Prev: ID: {prev_token}, Text: '{prev_text}'")
                                
                                if j < len(tokens) - 1:
                                    next_token = tokens[j+1]
                                    next_text = token_texts[j+1]
                                    print(f"  Next: ID: {next_token}, Text: '{next_text}'")
            except Exception as e:
                print(f"Error con codificación {encoding_name}: {e}")
    except Exception as e:
        print(f"Error en prueba tiktoken: {e}")
    
    # Terminado
    print("\nAnálisis completo. Use esta información para ajustar la búsqueda de tokens.")

async def main():
    """Función principal"""
    try:
        await analyze_tokens()
    except Exception as e:
        logger.error(f"Error en análisis de tokens: {e}")

if __name__ == "__main__":
    asyncio.run(main())
