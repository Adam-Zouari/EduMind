"""
LLM Generator Module
Integrates with Ollama for text generation using RAG context.
"""

import requests
import json
import logging
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaGenerator:
    """
    Generator that uses Ollama models for RAG-based text generation.
    """

    def __init__(
        self,
        model_name: str = "qwen3:1.7b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        """
        Initialize the Ollama generator.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"OllamaGenerator initialized with model: {model_name}")
        
        # Test connection
        self._test_connection()

    def _test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                logger.info(f"Connected to Ollama. Available models: {model_names}")
                
                if self.model_name not in model_names:
                    logger.warning(f"Model '{self.model_name}' not found in available models")
                    logger.warning(f"Available: {model_names}")
                
                return True
            else:
                logger.error(f"Failed to connect to Ollama: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Error connecting to Ollama: {e}")
            logger.error("Make sure Ollama is running (ollama serve)")
            return False

    def generate(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate a response using RAG context.

        Args:
            query: User's question
            context: Retrieved context from RAG pipeline
            system_prompt: Optional system prompt (default: RAG instruction)
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        # Default system prompt for RAG
        if system_prompt is None:
            system_prompt = (
                "You are a helpful AI assistant. Answer the user's question based on the provided context. "
                "If the context doesn't contain enough information to answer the question, say so. "
                "Be concise and accurate."
            )

        # Construct the prompt
        prompt = f"""Context:
{context}

Question: {query}

Answer:"""

        # Prepare request
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system_prompt,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            logger.info(f"Generating response with {self.model_name}...")
            
            if stream:
                return self._generate_stream(url, payload)
            else:
                return self._generate_non_stream(url, payload)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Failed to generate response - {str(e)}"

    def _generate_non_stream(self, url: str, payload: Dict) -> str:
        """Generate response without streaming."""
        response = requests.post(url, json=payload, timeout=120)
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result.get('response', '')
            logger.info(f"Generated {len(generated_text)} characters")
            return generated_text
        else:
            error_msg = f"Ollama API error: {response.status_code}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def _generate_stream(self, url: str, payload: Dict) -> str:
        """Generate response with streaming."""
        response = requests.post(url, json=payload, stream=True, timeout=120)
        
        if response.status_code == 200:
            full_response = ""
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if 'response' in chunk:
                        text = chunk['response']
                        full_response += text
                        print(text, end='', flush=True)  # Print as it streams
                    
                    if chunk.get('done', False):
                        print()  # New line at the end
                        break
            
            logger.info(f"Generated {len(full_response)} characters (streamed)")
            return full_response
        else:
            error_msg = f"Ollama API error: {response.status_code}"
            logger.error(error_msg)
            return f"Error: {error_msg}"

    def generate_with_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response from query results.

        Args:
            query: User's question
            results: List of result dictionaries from RAG query
            system_prompt: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        # Format context from results
        context_parts = []
        for i, result in enumerate(results):
            doc_text = result.get('document', '')
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'N/A')
            
            context_parts.append(
                f"[Document {i+1}]\n"
                f"{doc_text}\n"
                f"Source: {source}, Page: {page}\n"
            )
        
        context = "\n".join(context_parts)
        
        return self.generate(query, context, system_prompt, stream)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> str:
        """
        Chat interface (for multi-turn conversations).

        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream the response

        Returns:
            Generated response text
        """
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            
            if response.status_code == 200:
                if stream:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            chunk = json.loads(line)
                            if 'message' in chunk:
                                text = chunk['message'].get('content', '')
                                full_response += text
                                print(text, end='', flush=True)
                            
                            if chunk.get('done', False):
                                print()
                                break
                    return full_response
                else:
                    result = response.json()
                    return result.get('message', {}).get('content', '')
            else:
                return f"Error: Ollama API error {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"Error: {str(e)}"

    def list_models(self) -> List[str]:
        """
        List available Ollama models.

        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [m['name'] for m in models]
            return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []


if __name__ == "__main__":
    # Example usage
    generator = OllamaGenerator(model_name="qwen3:1.7b")
    
    # Test generation
    context = "Machine learning is a subset of AI that enables computers to learn from data."
    query = "What is machine learning?"
    
    print("\n=== Testing Ollama Generator ===\n")
    response = generator.generate(query, context)
    print(f"\nQuery: {query}")
    print(f"Response: {response}")
    
    # List available models
    print("\n=== Available Models ===")
    models = generator.list_models()
    for model in models:
        print(f"  - {model}")

