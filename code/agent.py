
import os
import logging
from typing import Any, Dict
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field
from dotenv import load_dotenv
import openai
import asyncio

# =========================
# Configuration Management
# =========================

class Config:
    """
    Configuration manager for environment variables and agent settings.
    """
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_TEMPERATURE: float = 0.7
    OPENAI_MAX_TOKENS: int = 2000
    SYSTEM_PROMPT: str = "You are a professional general agent."
    USER_PROMPT_TEMPLATE: str = "How can I help you today?"
    MAX_INPUT_LENGTH: int = 50000

    @classmethod
    def load(cls) -> "Config":
        """
        Loads configuration from environment variables and validates them.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("OPENAI_API_KEY is missing or empty in environment variables.")
        return cls()

    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.OPENAI_MODEL = os.getenv("OPENAI_MODEL", self.OPENAI_MODEL)
        self.OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", self.OPENAI_TEMPERATURE))
        self.OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", self.OPENAI_MAX_TOKENS))
        self.SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", self.SYSTEM_PROMPT)
        self.USER_PROMPT_TEMPLATE = os.getenv("USER_PROMPT_TEMPLATE", self.USER_PROMPT_TEMPLATE)
        self.MAX_INPUT_LENGTH = int(os.getenv("MAX_INPUT_LENGTH", self.MAX_INPUT_LENGTH))


# =========================
# Logging Configuration
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("EmailPayloadValidationAgent")


# =========================
# LLM Client
# =========================

class LLMClient:
    """
    Handles interaction with the OpenAI LLM.
    """
    def __init__(self, config: Config):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=self.config.OPENAI_API_KEY)

    async def chat(self, user_message: str) -> str:
        """
        Sends a message to the LLM and returns its response.
        """
        try:
            response = await self.client.chat.completions.create(
                model=self.config.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.config.OPENAI_TEMPERATURE,
                max_tokens=self.config.OPENAI_MAX_TOKENS
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            raise


# =========================
# Input Model & Validation
# =========================

class MessageInput(BaseModel):
    message: str = Field(..., description="User message to process.")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        """
        Validates and sanitizes the input message.
        """
        if not isinstance(v, str):
            raise ValueError("Message must be a string.")
        v = v.strip()
        if not v:
            raise ValueError("Message cannot be empty.")
        if len(v) > Config().MAX_INPUT_LENGTH:
            raise ValueError(f"Message exceeds maximum allowed length of {Config().MAX_INPUT_LENGTH} characters.")
        # Sanitize special characters if needed (basic example)
        v = v.replace('\x00', '')  # Remove null bytes
        return v


# =========================
# Agent Core
# =========================

class Agent:
    """
    Main agent class for processing messages.
    """
    def __init__(self, config: Config, llm_client: LLMClient):
        self.config = config
        self.llm_client = llm_client

    async def process_message(self, message: str) -> str:
        """
        Processes the user message and returns the LLM's response.
        """
        try:
            logger.info(f"Processing message: {message[:100]}{'...' if len(message) > 100 else ''}")
            response = await self.llm_client.chat(message)
            logger.info("LLM response generated successfully.")
            return response
        except Exception as e:
            logger.error(f"Error in process_message: {e}", exc_info=True)
            return self.handle_error(e)

    def handle_error(self, error: Exception) -> str:
        """
        Handles errors gracefully and returns a helpful message.
        """
        return (
            "Sorry, I couldn't process your request due to an internal error. "
            "Please check your input and try again. If the problem persists, contact support."
        )


# =========================
# FastAPI App & Endpoints
# =========================

app = FastAPI(
    title="Email Payload Validation Agent",
    description="A professional general agent for validating and processing email payloads using OpenAI LLM.",
    version="1.0.0"
)

# Allow CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration and initialize agent
try:
    config = Config.load()
    llm_client = LLMClient(config)
    agent = Agent(config, llm_client)
except Exception as e:
    logger.critical(f"Failed to initialize agent: {e}", exc_info=True)
    raise

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """
    Handles Pydantic validation errors.
    """
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "message": "Input validation failed.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is properly formatted.",
                "Check for missing or extra commas, brackets, or quotes.",
                "The 'message' field must be a non-empty string and under 50,000 characters."
            ]
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles HTTP exceptions.
    """
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "message": exc.detail,
            "tips": [
                "Check your request URL and method.",
                "Ensure your request body is valid JSON."
            ]
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """
    Handles generic exceptions.
    """
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "message": "An unexpected error occurred. Please try again later.",
            "tips": [
                "Check your input for special characters or formatting issues.",
                "If the problem persists, contact support."
            ]
        }
    )

@app.post("/process", response_model=Dict[str, Any])
async def process_message_endpoint(input_data: MessageInput):
    """
    Endpoint to process a user message via the agent.
    """
    try:
        response = await agent.process_message(input_data.message)
        return {
            "success": True,
            "response": response
        }
    except Exception as e:
        logger.error(f"Error in /process endpoint: {e}", exc_info=True)
        return {
            "success": False,
            "error_type": "ProcessingError",
            "message": str(e),
            "tips": [
                "Ensure your input is a valid, non-empty string.",
                "Avoid special characters that may break JSON formatting."
            ]
        }

@app.post("/health", response_model=Dict[str, Any])
async def health_check():
    """
    Health check endpoint.
    """
    return {
        "success": True,
        "status": "ok",
        "message": "Agent is running."
    }

@app.middleware("http")
async def catch_malformed_json(request: Request, call_next):
    """
    Middleware to catch malformed JSON requests and return helpful error messages.
    """
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            if request.headers.get("content-type", "").startswith("application/json"):
                await request.json()
        except Exception as exc:
            logger.warning(f"Malformed JSON: {exc}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "error_type": "MalformedJSON",
                    "message": "Malformed JSON in request body.",
                    "tips": [
                        "Ensure your JSON is properly formatted.",
                        "Check for missing or extra commas, brackets, or quotes.",
                        "Use double quotes for keys and string values."
                    ]
                }
            )
    response = await call_next(request)
    return response

# =========================
# Main Entry Point
# =========================

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Email Payload Validation Agent...")
    uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=False)
