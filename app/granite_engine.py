"""
Granite Engine for VAR Enforcer
Integrates IBM Granite via HuggingFace for VAR decision explanations
"""

import os
import json
import logging
from typing import List, Dict, Optional, Any
from enum import Enum
from dotenv import load_dotenv

# HuggingFace Transformers imports
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Using demo mode only.")

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


class VARExplanation:
    """Structured VAR explanation output"""
    
    def __init__(
        self,
        decision_explanation: str,
        rule_cited: List[str],
        controversy_score: int,
        consistency_note: str,
        plain_language_summary: str,
        language: str = "en"
    ):
        self.decision_explanation = decision_explanation
        self.rule_cited = rule_cited
        self.controversy_score = controversy_score
        self.consistency_note = consistency_note
        self.plain_language_summary = plain_language_summary
        self.language = language
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "decision_explanation": self.decision_explanation,
            "rule_cited": self.rule_cited,
            "controversy_score": self.controversy_score,
            "consistency_note": self.consistency_note,
            "plain_language_summary": self.plain_language_summary,
            "language": self.language
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class GraniteEngine:
    """
    IBM Granite Engine for generating VAR decision explanations
    Uses HuggingFace transformers for local model execution
    """
    
    def __init__(
        self,
        model_id: Optional[str] = None,
        demo_mode: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize Granite Engine
        
        Args:
            model_id: HuggingFace model ID
            demo_mode: If True, use mock responses without loading model
            device: Device to run model on ('cuda', 'cpu', or None for auto)
        """
        self.model_id = model_id or os.getenv("GRANITE_MODEL_ID", "ibm-granite/granite-3.0-8b-instruct")
        self.demo_mode = demo_mode or os.getenv("DEMO_MODE", "false").lower() == "true"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu") if TRANSFORMERS_AVAILABLE else "cpu"
        
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        
        # Check if we can use real model
        if not self.demo_mode and TRANSFORMERS_AVAILABLE:
            self._initialize_model()
        else:
            if not self.demo_mode:
                logger.warning("Transformers not available or model loading failed. Running in DEMO MODE.")
            self.demo_mode = True
            logger.info("Granite Engine initialized in DEMO MODE")
    
    def _initialize_model(self):
        """Initialize HuggingFace model and pipeline"""
        try:
            logger.info(f"Loading Granite model from HuggingFace: {self.model_id}")
            logger.info(f"Using device: {self.device}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            
            # Load model with appropriate settings
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
            
            logger.info(f"Granite model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading Granite model: {str(e)}")
            logger.warning("Falling back to DEMO MODE")
            self.demo_mode = True
            self.pipeline = None
            self.model = None
            self.tokenizer = None
    
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
        # Language-specific instructions
        language_instructions = {
            Language.ENGLISH: "Respond in English.",
            Language.SPANISH: "Responde en español.",
            Language.FRENCH: "Répondez en français.",
            Language.PORTUGUESE: "Responda em português."
        }
        
        # Build context from chunks
        context_text = "\n\n".join([
            f"FIFA Rule Context {i+1}:\n{chunk['text']}"
            for i, chunk in enumerate(context_chunks[:5])  # Limit to top 5
        ])
        
        prompt = f"""You are an expert FIFA VAR (Video Assistant Referee) analyst. Your task is to explain VAR decisions based on official FIFA Laws of the Game.

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
5. Assess historical consistency with similar past decisions
6. Provide a plain language summary accessible to casual fans

{language_instructions[language]}

OUTPUT FORMAT (JSON):
{{
  "decision_explanation": "Detailed technical explanation of the VAR decision with rule citations",
  "rule_cited": ["List of specific FIFA rule numbers/sections referenced"],
  "controversy_score": <integer 1-10>,
  "consistency_note": "Brief note on how this aligns with historical VAR decisions",
  "plain_language_summary": "Simple explanation a casual fan would understand"
}}

Generate the JSON response now:"""
        
        return prompt
    
    def _parse_response(self, response_text: str, language: str) -> VARExplanation:
        """
        Parse Granite's response into structured format
        
        Args:
            response_text: Raw response from Granite
            language: Language code
            
        Returns:
            VARExplanation object
        """
        try:
            # Try to extract JSON from response
            # Handle cases where model adds extra text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                # Fallback: try parsing entire response
                data = json.loads(response_text)
            
            return VARExplanation(
                decision_explanation=data.get("decision_explanation", ""),
                rule_cited=data.get("rule_cited", []),
                controversy_score=int(data.get("controversy_score", 5)),
                consistency_note=data.get("consistency_note", ""),
                plain_language_summary=data.get("plain_language_summary", ""),
                language=language
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            
            # Return a fallback explanation
            return VARExplanation(
                decision_explanation=response_text[:500],
                rule_cited=["Unable to parse specific rules"],
                controversy_score=5,
                consistency_note="Unable to assess consistency",
                plain_language_summary=response_text[:200],
                language=language
            )
    
    def _generate_with_transformers(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response using HuggingFace transformers pipeline
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        try:
            # Generate with pipeline
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                return_full_text=False
            )
            
            # Extract generated text
            generated_text = outputs[0]['generated_text']
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating with transformers: {str(e)}")
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
        logger.info("Generating DEMO response (no model loaded)")
        
        # Extract some context for realistic demo
        rule_context = context_chunks[0]['text'][:200] if context_chunks else "FIFA Laws of the Game"
        
        # Language-specific demo responses
        demo_responses = {
            Language.ENGLISH: {
                "decision_explanation": f"Based on the FIFA Laws of the Game, this VAR decision involves careful analysis of the incident. The referee's original decision was reviewed using video evidence. Context: {rule_context}... The VAR protocol requires clear and obvious errors to be corrected. In this case, the evidence supports the final decision made.",
                "rule_cited": ["Law 5 - The Referee", "Law 12 - Fouls and Misconduct", "VAR Protocol Section 3"],
                "controversy_score": 6,
                "consistency_note": "This decision aligns with similar VAR interventions seen in recent major tournaments. The threshold for 'clear and obvious error' has been consistently applied.",
                "plain_language_summary": "The VAR checked the incident and found enough evidence to support the referee's decision. While some fans may disagree, the call follows the official rules and is consistent with how similar situations have been handled."
            },
            Language.SPANISH: {
                "decision_explanation": f"Según las Leyes del Juego de la FIFA, esta decisión del VAR implica un análisis cuidadoso del incidente. Contexto: {rule_context}... El protocolo VAR requiere errores claros y obvios para ser corregidos.",
                "rule_cited": ["Ley 5 - El Árbitro", "Ley 12 - Faltas y Conducta Incorrecta", "Protocolo VAR Sección 3"],
                "controversy_score": 6,
                "consistency_note": "Esta decisión se alinea con intervenciones VAR similares en torneos recientes.",
                "plain_language_summary": "El VAR revisó el incidente y encontró suficiente evidencia para apoyar la decisión del árbitro."
            },
            Language.FRENCH: {
                "decision_explanation": f"Selon les Lois du Jeu de la FIFA, cette décision VAR implique une analyse minutieuse de l'incident. Contexte: {rule_context}... Le protocole VAR nécessite des erreurs claires et évidentes pour être corrigées.",
                "rule_cited": ["Loi 5 - L'Arbitre", "Loi 12 - Fautes et Comportement Antisportif", "Protocole VAR Section 3"],
                "controversy_score": 6,
                "consistency_note": "Cette décision s'aligne avec des interventions VAR similaires dans les tournois récents.",
                "plain_language_summary": "Le VAR a vérifié l'incident et a trouvé suffisamment de preuves pour soutenir la décision de l'arbitre."
            },
            Language.PORTUGUESE: {
                "decision_explanation": f"De acordo com as Leis do Jogo da FIFA, esta decisão do VAR envolve análise cuidadosa do incidente. Contexto: {rule_context}... O protocolo VAR requer erros claros e óbvios para serem corrigidos.",
                "rule_cited": ["Lei 5 - O Árbitro", "Lei 12 - Faltas e Má Conduta", "Protocolo VAR Seção 3"],
                "controversy_score": 6,
                "consistency_note": "Esta decisão está alinhada com intervenções VAR similares em torneios recentes.",
                "plain_language_summary": "O VAR verificou o incidente e encontrou evidências suficientes para apoiar a decisão do árbitro."
            }
        }
        
        response_data = demo_responses.get(language, demo_responses[Language.ENGLISH])
        
        return VARExplanation(
            decision_explanation=response_data["decision_explanation"],
            rule_cited=response_data["rule_cited"],
            controversy_score=response_data["controversy_score"],
            consistency_note=response_data["consistency_note"],
            plain_language_summary=response_data["plain_language_summary"],
            language=language.value
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
            logger.info(f"Generating VAR explanation for query: {query[:100]}...")
            
            # Demo mode
            if self.demo_mode:
                return self._generate_demo_response(query, context_chunks, language)
            
            # Build prompt
            prompt = self._build_prompt(query, context_chunks, language)
            
            # Generate response
            try:
                response_text = self._generate_with_transformers(prompt, max_tokens, temperature)
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                logger.warning("Falling back to demo mode for this request")
                return self._generate_demo_response(query, context_chunks, language)
            
            # Parse response
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
                # Add error placeholder
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
            "model_id": self.model_id,
            "demo_mode": self.demo_mode,
            "device": self.device,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "model_loaded": self.pipeline is not None,
            "supported_languages": [lang.value for lang in Language]
        }


# Convenience function
def create_granite_engine(demo_mode: bool = False) -> GraniteEngine:
    """
    Create and return a Granite Engine instance
    
    Args:
        demo_mode: Force demo mode
        
    Returns:
        Initialized GraniteEngine instance
    """
    return GraniteEngine(demo_mode=demo_mode)


# Example usage
if __name__ == "__main__":
    # Create engine (will use demo mode if transformers not available)
    engine = create_granite_engine()
    
    # Get engine info
    info = engine.get_engine_info()
    print(f"\nEngine Info: {json.dumps(info, indent=2)}")
    
    # Example query
    query = "Was the penalty decision correct when the defender touched the ball first but also made contact with the attacker's leg?"
    
    # Mock context chunks
    context_chunks = [
        {
            "text": "A direct free kick is awarded if a player commits any of the following offences: handles the ball deliberately, holds an opponent, impedes an opponent with contact, or tackles an opponent to gain possession of the ball, making contact with the opponent before touching the ball.",
            "metadata": {"chunk_id": 0}
        }
    ]
    
    # Generate explanation
    explanation = engine.explain_var_decision(query, context_chunks, Language.ENGLISH)
    
    print(f"\nQuery: {query}")
    print(f"\nExplanation:\n{explanation.to_json()}")
    
    # Test multilingual
    print("\n" + "="*50)
    print("Testing Spanish output:")
    explanation_es = engine.explain_var_decision(query, context_chunks, Language.SPANISH)
    print(explanation_es.to_json())

# Made with Bob
