"""
Granite Engine for VAR Enforcer
Integrates IBM Granite via Ollama for VAR decision explanations
"""

import os
import json
import logging
import requests
from typing import List, Dict, Optional, Any
from enum import Enum
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class Language(str, Enum):
    """Supported languages for VAR explanations"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    PORTUGUESE = "pt"
    GERMAN = "de"
    ARABIC = "ar"
    ITALIAN = "it"
    DUTCH = "nl"
    JAPANESE = "ja"
    TURKISH = "tr"
    CROATIAN = "hr"
    NORWEGIAN = "no"


# Expanded language support dictionary
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "flag": "🇬🇧", "prompt_instruction": "Respond in English"},
    "es": {"name": "Español", "flag": "🇪🇸", "prompt_instruction": "Responde en español"},
    "fr": {"name": "Français", "flag": "🇫🇷", "prompt_instruction": "Réponds en français"},
    "pt": {"name": "Português", "flag": "🇧🇷", "prompt_instruction": "Responda em português"},
    "de": {"name": "Deutsch", "flag": "🇩🇪", "prompt_instruction": "Antworte auf Deutsch"},
    "ar": {"name": "العربية", "flag": "🇸🇦", "prompt_instruction": "أجب باللغة العربية"},
    "it": {"name": "Italiano", "flag": "🇮🇹", "prompt_instruction": "Rispondi in italiano"},
    "nl": {"name": "Nederlands", "flag": "🇳🇱", "prompt_instruction": "Antwoord in het Nederlands"},
    "ja": {"name": "日本語", "flag": "🇯🇵", "prompt_instruction": "日本語で答えてください"},
    "tr": {"name": "Türkçe", "flag": "🇹🇷", "prompt_instruction": "Türkçe yanıt ver"},
    "hr": {"name": "Hrvatski", "flag": "🇭🇷", "prompt_instruction": "Odgovori na hrvatskom"},
    "no": {"name": "Norsk", "flag": "🇳🇴", "prompt_instruction": "Svar på norsk"}
}


class VARExplanation:
    """Structured VAR explanation output"""
    
    def __init__(
        self,
        decision_explanation: str,
        rule_cited: List[str],
        controversy_score: int,
        consistency_note: str,
        plain_language_summary: str,
        language: str = "en",
        confidence_score: int = 75
    ):
        self.decision_explanation = decision_explanation
        self.rule_cited = rule_cited
        self.controversy_score = controversy_score
        self.consistency_note = consistency_note
        self.plain_language_summary = plain_language_summary
        self.language = language
        self.confidence_score = confidence_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_explanation": self.decision_explanation,
            "rule_cited": self.rule_cited,
            "controversy_score": self.controversy_score,
            "consistency_note": self.consistency_note,
            "confidence_score": self.confidence_score,
            "plain_language_summary": self.plain_language_summary,
            "language": self.language
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class GraniteEngine:
    """
    IBM Granite Engine for generating VAR decision explanations
    Uses Ollama for local model execution
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        ollama_url: Optional[str] = None,
        demo_mode: bool = False
    ):
        """
        Initialize Granite Engine
        
        Args:
            model_name: Ollama model name
            ollama_url: Ollama API endpoint
            demo_mode: If True, use mock responses without API calls
        """
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "granite3.3")
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.demo_mode = demo_mode or os.getenv("DEMO_MODE", "false").lower() == "true"
        
        # Ollama API endpoint
        self.api_url = f"{self.ollama_url}/api/generate"
        
        # Check if Ollama is available
        if not self.demo_mode:
            if not self._check_ollama_available():
                logger.warning("Ollama not available. Running in DEMO MODE.")
                logger.info("Install Ollama from: https://ollama.ai")
                logger.info(f"Then run: ollama pull {self.model_name}")
                self.demo_mode = True
        
        if self.demo_mode:
            logger.info("Granite Engine initialized in DEMO MODE")
        else:
            logger.info(f"Granite Engine initialized with Ollama")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Endpoint: {self.api_url}")
    
    def _check_ollama_available(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Ollama check failed: {str(e)}")
            return False
    
    def _build_prompt(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: Language = Language.ENGLISH
    ) -> str:
        """
        Build structured prompt for Granite
        
        Args:
            query: User's VAR incident query
            context_chunks: Retrieved FIFA rule chunks from RAG
            language: Target language for response
            
        Returns:
            Formatted prompt string
        """
        # Get language instruction from SUPPORTED_LANGUAGES dictionary
        lang_code = language.value if isinstance(language, Language) else language
        language_instruction = SUPPORTED_LANGUAGES.get(lang_code, SUPPORTED_LANGUAGES["en"])["prompt_instruction"]
        
        logger.info(f"Building prompt with language instruction: {language_instruction}")
        
        # Build context from chunks
        context_text = "\n\n".join([
            f"FIFA Rule Context {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks[:5])
        ])
        
        prompt = f"""LANGUAGE REQUIREMENT: {language_instruction}. ALL responses must be in this language.

You are an expert FIFA VAR (Video Assistant Referee) analyst. Your task is to explain VAR decisions based on official FIFA Laws of the Game.

USER QUERY:
{query}

RELEVANT FIFA RULES:
{context_text}

INSTRUCTIONS:
1. Analyze the VAR incident described in the user query
2. Reference the specific FIFA rules provided in the context
3. Provide a clear, authoritative explanation of the decision
4. Rate the controversy level (1-10 scale):
   - 1-3: Clear-cut decision, minimal controversy
   - 4-6: Moderate controversy, some debate expected
   - 7-10: Highly controversial, emotionally charged
5. Rate your confidence in this analysis (1-100 scale):
   - 75-100: High confidence, clear rule application
   - 50-74: Moderate confidence, some interpretation needed
   - Below 50: Low confidence, ambiguous situation
6. Assess historical consistency with similar past decisions
7. Provide a plain language summary accessible to casual fans

CRITICAL: Remember to {language_instruction}. Your entire response must be in the requested language.

OUTPUT FORMAT (JSON):
{{
  "decision_explanation": "Detailed technical explanation of the VAR decision with rule citations",
  "rule_cited": ["List of specific FIFA rule numbers/sections referenced"],
  "controversy_score": <integer 1-10>,
  "confidence_score": <integer 1-100>,
  "consistency_note": "Brief note on how this aligns with historical VAR decisions",
  "plain_language_summary": "Simple explanation a casual fan would understand"
}}

Generate the JSON response now:"""
        
        return prompt
    
    def _parse_response(self, response_text: str, language: str) -> VARExplanation:
        """
        Parse Granite's response into structured format with robust fallback strategies
        
        Args:
            response_text: Raw response from Granite
            language: Language code
            
        Returns:
            VARExplanation object
        """
        # Log raw response for debugging
        logger.info(f"Raw Granite response (first 500 chars): {response_text[:500]}")
        
        # Strategy 1: Try standard JSON parsing
        try:
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
                
                return VARExplanation(
                    decision_explanation=data.get("decision_explanation", ""),
                    rule_cited=data.get("rule_cited", []),
                    controversy_score=int(data.get("controversy_score", 5)),
                    consistency_note=data.get("consistency_note", ""),
                    plain_language_summary=data.get("plain_language_summary", ""),
                    language=language,
                    confidence_score=int(data.get("confidence_score", 75))
                )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"JSON parsing failed: {str(e)}, trying fallback strategies")
        
        # Strategy 2: Try to fix common JSON issues
        try:
            # Remove markdown code blocks if present
            cleaned = response_text.replace('```json', '').replace('```', '')
            
            # Try to find JSON with more lenient approach
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                # Fix common issues: trailing commas, single quotes
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
                json_str = json_str.replace("'", '"')  # Replace single quotes
                
                data = json.loads(json_str)
                
                return VARExplanation(
                    decision_explanation=data.get("decision_explanation", ""),
                    rule_cited=data.get("rule_cited", []),
                    controversy_score=int(data.get("controversy_score", 5)),
                    consistency_note=data.get("consistency_note", ""),
                    plain_language_summary=data.get("plain_language_summary", ""),
                    language=language,
                    confidence_score=int(data.get("confidence_score", 75))
                )
        except Exception as e:
            logger.warning(f"Lenient JSON parsing failed: {str(e)}, using text extraction fallback")
        
        # Strategy 3: Text extraction fallback - extract information from raw text
        logger.info("Using text extraction fallback to parse response")
        
        # Extract decision explanation (first substantial paragraph)
        decision_explanation = response_text.strip()
        if len(decision_explanation) > 500:
            # Take first 500 chars as explanation
            decision_explanation = decision_explanation[:500] + "..."
        
        # Extract rules cited - look for "Law", "Rule", "FIFA", "Article" mentions
        import re
        rule_cited = []
        
        # Pattern 1: "Law X" or "Rule X"
        law_matches = re.findall(r'(?:Law|Rule|Article)\s+\d+[A-Za-z]?(?:\s*[-–]\s*[^.,\n]{0,50})?', response_text, re.IGNORECASE)
        rule_cited.extend(law_matches[:5])  # Limit to 5 rules
        
        # Pattern 2: "FIFA" followed by relevant text
        fifa_matches = re.findall(r'FIFA[^.,\n]{0,50}', response_text, re.IGNORECASE)
        rule_cited.extend(fifa_matches[:2])
        
        # Pattern 3: "VAR Protocol" mentions
        var_matches = re.findall(r'VAR\s+Protocol[^.,\n]{0,50}', response_text, re.IGNORECASE)
        rule_cited.extend(var_matches[:2])
        
        # If no rules found, provide generic reference
        if not rule_cited:
            rule_cited = ["FIFA Laws of the Game", "VAR Protocol"]
        
        # Remove duplicates and limit to 5
        rule_cited = list(dict.fromkeys(rule_cited))[:5]
        
        # Extract controversy score - look for numbers 1-10
        controversy_score = 5  # Default
        controversy_patterns = [
            r'controversy[:\s]+(\d+)',
            r'controversy score[:\s]+(\d+)',
            r'rated?\s+(\d+)\s*(?:out of|/)\s*10',
            r'score[:\s]+(\d+)'
        ]
        for pattern in controversy_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 10:
                    controversy_score = score
                    break
        
        # Extract confidence score - look for percentage or number 1-100
        confidence_score = 75  # Default
        confidence_patterns = [
            r'confidence[:\s]+(\d+)%?',
            r'confidence score[:\s]+(\d+)%?',
            r'(\d+)%?\s+confidence',
            r'certainty[:\s]+(\d+)%?'
        ]
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 1 <= score <= 100:
                    confidence_score = score
                    break
        
        # Extract consistency note - look for sentences with consistency keywords
        consistency_note = "This decision follows established VAR protocols."  # Default
        consistency_keywords = ['consistent', 'consistency', 'similar', 'historical', 'previous', 'past', 'align', 'standard']
        
        sentences = re.split(r'[.!?]+', response_text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in consistency_keywords):
                consistency_note = sentence.strip()
                if len(consistency_note) > 200:
                    consistency_note = consistency_note[:200] + "..."
                break
        
        # Extract plain language summary - look for summary section or use last paragraph
        plain_language_summary = decision_explanation[:200]  # Default to first 200 chars
        
        summary_patterns = [
            r'(?:plain language|summary|in simple terms|simply put)[:\s]+([^.!?]+[.!?])',
            r'(?:casual fans?|general audience)[:\s]+([^.!?]+[.!?])'
        ]
        for pattern in summary_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                plain_language_summary = match.group(1).strip()
                break
        
        # If no summary found, use last substantial sentence
        if plain_language_summary == decision_explanation[:200]:
            sentences = [s.strip() for s in re.split(r'[.!?]+', response_text) if len(s.strip()) > 50]
            if sentences:
                plain_language_summary = sentences[-1]
                if len(plain_language_summary) > 200:
                    plain_language_summary = plain_language_summary[:200] + "..."
        
        logger.info(f"Extracted via fallback - Rules: {len(rule_cited)}, Controversy: {controversy_score}, Confidence: {confidence_score}")
        
        return VARExplanation(
            decision_explanation=decision_explanation,
            rule_cited=rule_cited,
            controversy_score=controversy_score,
            consistency_note=consistency_note,
            plain_language_summary=plain_language_summary,
            language=language,
            confidence_score=confidence_score
        )
    
    def _generate_with_ollama(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using Ollama API
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": 500,
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repeat_penalty": 1.1
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300
            )
            
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            return generated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    def _generate_demo_response(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: Language
    ) -> VARExplanation:
        """
        Generate mock response for demo mode
        
        Args:
            query: User query
            context_chunks: Context chunks
            language: Target language
            
        Returns:
            Mock VARExplanation
        """
        logger.info("Generating DEMO response (Ollama not available)")
        
        rule_context = context_chunks[0]['text'][:200] if context_chunks else "FIFA Laws of the Game"
        
        demo_responses = {
            Language.ENGLISH: {
                "decision_explanation": f"Based on the FIFA Laws of the Game, this VAR decision involves careful analysis of the incident. The referee's original decision was reviewed using video evidence. Context: {rule_context}... The VAR protocol requires clear and obvious errors to be corrected. In this case, the evidence supports the final decision made.",
                "rule_cited": ["Law 5 - The Referee", "Law 12 - Fouls and Misconduct", "VAR Protocol Section 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "This decision aligns with similar VAR interventions seen in recent major tournaments. The threshold for 'clear and obvious error' has been consistently applied.",
                "plain_language_summary": "The VAR checked the incident and found enough evidence to support the referee's decision. While some fans may disagree, the call follows the official rules and is consistent with how similar situations have been handled."
            },
            Language.SPANISH: {
                "decision_explanation": f"Según las Leyes del Juego de la FIFA, esta decisión del VAR implica un análisis cuidadoso del incidente. Contexto: {rule_context}... El protocolo VAR requiere errores claros y obvios para ser corregidos.",
                "rule_cited": ["Ley 5 - El Árbitro", "Ley 12 - Faltas y Conducta Incorrecta", "Protocolo VAR Sección 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Esta decisión se alinea con intervenciones VAR similares en torneos recientes.",
                "plain_language_summary": "El VAR revisó el incidente y encontró suficiente evidencia para apoyar la decisión del árbitro."
            },
            Language.FRENCH: {
                "decision_explanation": f"Selon les Lois du Jeu de la FIFA, cette décision VAR implique une analyse minutieuse de l'incident. Contexte: {rule_context}... Le protocole VAR nécessite des erreurs claires et évidentes pour être corrigées.",
                "rule_cited": ["Loi 5 - L'Arbitre", "Loi 12 - Fautes et Comportement Antisportif", "Protocole VAR Section 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Cette décision s'aligne avec des interventions VAR similaires dans les tournois récents.",
                "plain_language_summary": "Le VAR a vérifié l'incident et a trouvé suffisamment de preuves pour soutenir la décision de l'arbitre."
            },
            Language.PORTUGUESE: {
                "decision_explanation": f"De acordo com as Leis do Jogo da FIFA, esta decisão do VAR envolve análise cuidadosa do incidente. Contexto: {rule_context}... O protocolo VAR requer erros claros e óbvios para serem corrigidos.",
                "rule_cited": ["Lei 5 - O Árbitro", "Lei 12 - Faltas e Má Conduta", "Protocolo VAR Seção 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Esta decisão está alinhada com intervenções VAR similares em torneios recentes.",
                "plain_language_summary": "O VAR verificou o incidente e encontrou evidências suficientes para apoiar a decisão do árbitro."
            },
            Language.GERMAN: {
                "decision_explanation": f"Gemäß den FIFA-Spielregeln erfordert diese VAR-Entscheidung eine sorgfältige Analyse des Vorfalls. Kontext: {rule_context}... Das VAR-Protokoll erfordert klare und offensichtliche Fehler zur Korrektur.",
                "rule_cited": ["Regel 5 - Der Schiedsrichter", "Regel 12 - Fouls und unsportliches Verhalten", "VAR-Protokoll Abschnitt 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Diese Entscheidung entspricht ähnlichen VAR-Eingriffen bei jüngsten großen Turnieren.",
                "plain_language_summary": "Der VAR überprüfte den Vorfall und fand genügend Beweise zur Unterstützung der Schiedsrichterentscheidung."
            },
            Language.ARABIC: {
                "decision_explanation": f"بناءً على قوانين اللعبة من الفيفا، يتطلب قرار الفار هذا تحليلاً دقيقاً للحادثة. السياق: {rule_context}... يتطلب بروتوكول الفار أخطاء واضحة وجلية ليتم تصحيحها.",
                "rule_cited": ["القانون 5 - الحكم", "القانون 12 - الأخطاء وسوء السلوك", "بروتوكول الفار القسم 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "يتماشى هذا القرار مع تدخلات الفار المماثلة في البطولات الكبرى الأخيرة.",
                "plain_language_summary": "راجع الفار الحادثة ووجد أدلة كافية لدعم قرار الحكم."
            },
            Language.ITALIAN: {
                "decision_explanation": f"Secondo le Regole del Gioco FIFA, questa decisione VAR richiede un'analisi attenta dell'incidente. Contesto: {rule_context}... Il protocollo VAR richiede errori chiari ed evidenti da correggere.",
                "rule_cited": ["Regola 5 - L'Arbitro", "Regola 12 - Falli e Scorrettezze", "Protocollo VAR Sezione 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Questa decisione è in linea con interventi VAR simili visti nei recenti tornei importanti.",
                "plain_language_summary": "Il VAR ha controllato l'incidente e ha trovato prove sufficienti per supportare la decisione dell'arbitro."
            },
            Language.DUTCH: {
                "decision_explanation": f"Volgens de FIFA-spelregels vereist deze VAR-beslissing een zorgvuldige analyse van het incident. Context: {rule_context}... Het VAR-protocol vereist duidelijke en voor de hand liggende fouten om te corrigeren.",
                "rule_cited": ["Regel 5 - De Scheidsrechter", "Regel 12 - Overtredingen en Wangedrag", "VAR-protocol Sectie 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Deze beslissing sluit aan bij vergelijkbare VAR-interventies in recente grote toernooien.",
                "plain_language_summary": "De VAR controleerde het incident en vond voldoende bewijs om de beslissing van de scheidsrechter te ondersteunen."
            },
            Language.JAPANESE: {
                "decision_explanation": f"FIFAの競技規則に基づき、このVAR判定には事象の慎重な分析が必要です。コンテキスト: {rule_context}... VARプロトコルは、明白な誤りを修正する必要があります。",
                "rule_cited": ["規則5 - 主審", "規則12 - ファウルと不正行為", "VARプロトコル セクション3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "この判定は、最近の主要トーナメントで見られた同様のVAR介入と一致しています。",
                "plain_language_summary": "VARは事象を確認し、主審の判定を支持する十分な証拠を見つけました。"
            },
            Language.TURKISH: {
                "decision_explanation": f"FIFA Oyun Kurallarına göre, bu VAR kararı olayın dikkatli bir analizini gerektirir. Bağlam: {rule_context}... VAR protokolü, düzeltilmesi gereken açık ve belirgin hatalar gerektirir.",
                "rule_cited": ["Kural 5 - Hakem", "Kural 12 - Fauller ve Kötü Davranış", "VAR Protokolü Bölüm 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Bu karar, son büyük turnuvalarda görülen benzer VAR müdahaleleriyle uyumludur.",
                "plain_language_summary": "VAR olayı kontrol etti ve hakemin kararını desteklemek için yeterli kanıt buldu."
            },
            Language.CROATIAN: {
                "decision_explanation": f"Prema FIFA pravilima igre, ova VAR odluka zahtijeva pažljivu analizu incidenta. Kontekst: {rule_context}... VAR protokol zahtijeva jasne i očite pogreške za ispravak.",
                "rule_cited": ["Pravilo 5 - Sudac", "Pravilo 12 - Prekršaji i Neprimjereno Ponašanje", "VAR Protokol Odjeljak 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Ova odluka je u skladu sa sličnim VAR intervencijama viđenim na nedavnim velikim turnirima.",
                "plain_language_summary": "VAR je provjerio incident i pronašao dovoljno dokaza za podršku sudačkoj odluci."
            },
            Language.NORWEGIAN: {
                "decision_explanation": f"I henhold til FIFAs spilleregler krever denne VAR-avgjørelsen en nøye analyse av hendelsen. Kontekst: {rule_context}... VAR-protokollen krever klare og åpenbare feil for å korrigeres.",
                "rule_cited": ["Regel 5 - Dommeren", "Regel 12 - Overtredelser og Ufint Spill", "VAR-protokoll Seksjon 3"],
                "controversy_score": 6,
                "confidence_score": 78,
                "consistency_note": "Denne avgjørelsen er i tråd med lignende VAR-inngrep sett i nylige store turneringer.",
                "plain_language_summary": "VAR sjekket hendelsen og fant nok bevis til å støtte dommerens avgjørelse."
            }
        }
        
        response_data = demo_responses.get(language, demo_responses[Language.ENGLISH])
        
        return VARExplanation(
            decision_explanation=response_data["decision_explanation"],
            rule_cited=response_data["rule_cited"],
            controversy_score=response_data["controversy_score"],
            consistency_note=response_data["consistency_note"],
            plain_language_summary=response_data["plain_language_summary"],
            language=language.value,
            confidence_score=response_data["confidence_score"]
        )
    
    def explain_var_decision(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        language: Language = Language.ENGLISH,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> VARExplanation:
        """
        Generate VAR decision explanation
        
        Args:
            query: User's VAR incident query
            context_chunks: Retrieved FIFA rule chunks from RAG
            language: Target language for response
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            VARExplanation object with structured response
        """
        try:
            lang_code = language.value if isinstance(language, Language) else language
            logger.info(f"Generating VAR explanation in language: {lang_code}")
            logger.info(f"Query: {query[:100]}...")
            
            if self.demo_mode:
                return self._generate_demo_response(query, context_chunks, language)
            
            prompt = self._build_prompt(query, context_chunks, language)
            
            try:
                response_text = self._generate_with_ollama(prompt, max_tokens, temperature)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                logger.warning("Falling back to demo mode for this request")
                return self._generate_demo_response(query, context_chunks, language)
            
            explanation = self._parse_response(response_text, language.value)
            
            logger.info(f"Generated explanation with controversy score: {explanation.controversy_score}")
            return explanation
            
        except Exception as e:
            logger.error(f"Error in explain_var_decision: {str(e)}")
            raise
    
    def batch_explain(
        self,
        queries: List[str],
        context_chunks_list: List[List[Dict[str, Any]]],
        language: Language = Language.ENGLISH
    ) -> List[VARExplanation]:
        """
        Generate explanations for multiple queries
        
        Args:
            queries: List of user queries
            context_chunks_list: List of context chunks for each query
            language: Target language
            
        Returns:
            List of VARExplanation objects
        """
        explanations = []
        
        for query, context_chunks in zip(queries, context_chunks_list):
            try:
                explanation = self.explain_var_decision(query, context_chunks, language)
                explanations.append(explanation)
            except Exception as e:
                logger.error(f"Error processing query '{query[:50]}...': {str(e)}")
                explanations.append(VARExplanation(
                    decision_explanation=f"Error processing query: {str(e)}",
                    rule_cited=["Error"],
                    controversy_score=0,
                    consistency_note="Error occurred",
                    plain_language_summary="Unable to generate explanation",
                    language=language.value
                ))
        
        return explanations
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the engine configuration
        
        Returns:
            Dictionary with engine info
        """
        return {
            "model_name": self.model_name,
            "demo_mode": self.demo_mode,
            "ollama_url": self.ollama_url,
            "api_url": self.api_url,
            "ollama_available": self._check_ollama_available() if not self.demo_mode else False,
            "supported_languages": SUPPORTED_LANGUAGES
        }


def create_granite_engine(demo_mode: bool = False) -> GraniteEngine:
    """
    Create and return a Granite Engine instance
    
    Args:
        demo_mode: Force demo mode
        
    Returns:
        Initialized GraniteEngine instance
    """
    return GraniteEngine(demo_mode=demo_mode)


if __name__ == "__main__":
    engine = create_granite_engine()
    
    info = engine.get_engine_info()
    print(f"\nEngine Info: {json.dumps(info, indent=2)}")
    
    query = "Was the penalty decision correct when the defender touched the ball first but also made contact with the attacker's leg?"
    
    context_chunks = [
        {
            "text": "A direct free kick is awarded if a player commits any of the following offences: handles the ball deliberately, holds an opponent, impedes an opponent with contact, or tackles an opponent to gain possession of the ball, making contact with the opponent before touching the ball.",
            "metadata": {"chunk_id": 0}
        }
    ]
    
    explanation = engine.explain_var_decision(query, context_chunks, Language.ENGLISH)
    
    print(f"\nQuery: {query}")
    print(f"\nExplanation:\n{explanation.to_json()}")
    
    print("\n" + "="*50)
    print("Testing Spanish output:")
    explanation_es = engine.explain_var_decision(query, context_chunks, Language.SPANISH)
    print(explanation_es.to_json())

# Made with Bob
