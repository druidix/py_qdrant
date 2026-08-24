"""Hugging Face authentication utilities."""

import os

from dotenv import load_dotenv


def authenticate_huggingface(*, required: bool = True) -> str | None:
    """
    Load the Hugging Face API token into the environment.

    Uses HF_TOKEN from environment variables. Libraries that download models
    from the Hugging Face Hub (such as FastEmbed, via huggingface_hub) read
    HF_TOKEN from the environment themselves, so this only needs to run once
    before the first download is triggered.

    Args:
        required: If True, raise when HF_TOKEN is missing. Set to False for
            public models, which the Hub serves without authentication.

    Raises:
        EnvironmentError: If required is True and HF_TOKEN is not defined in
            environment variables.

    Returns:
        str | None: The token, or None if it is absent and not required.
    """
    load_dotenv()

    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        if required:
            raise EnvironmentError("HF_TOKEN not defined in environment variables")
        return None

    # huggingface_hub reads HF_TOKEN from the environment; load_dotenv() has
    # already placed it there, so no further wiring is needed.
    return hf_token
