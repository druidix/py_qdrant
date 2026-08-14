"""OpenAI client initialization utilities."""

import os

from dotenv import load_dotenv
from openai import OpenAI


def get_openai_connection(**kwargs) -> OpenAI:
    """
    Initialize and return an OpenAI client.

    Uses OPEN_AI_API_KEY from environment variables to authenticate.

    Args:
        **kwargs: Additional keyword arguments to pass to OpenAI,
            such as organization, base_url, timeout, etc.

    Raises:
        EnvironmentError: If OPEN_AI_API_KEY is not defined in environment
            variables.

    Returns:
        OpenAI: A connection to the OpenAI API.
    """
    load_dotenv()

    openai_api_key = os.getenv("OPEN_AI_API_KEY")

    if not openai_api_key:
        raise EnvironmentError("OPEN_AI_API_KEY not defined in environment variables")

    return OpenAI(api_key=openai_api_key, **kwargs)
