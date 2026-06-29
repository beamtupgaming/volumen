"""
ai_analysis.py — AI-powered text extraction and translation using Ollama.
"""

import base64
import json
import tempfile
import os
from pathlib import Path

import cv2
import numpy as np

try:
    import ollama
    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False


def _image_to_base64_png(img: np.ndarray) -> str:
    """Encode a NumPy image array as a base64 PNG string."""
    # Convert grayscale to BGR so JPEG/PNG roundtrip works cleanly
    if img.ndim == 2:
        display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        display = img
    success, buffer = cv2.imencode(".png", display)
    if not success:
        raise RuntimeError("Failed to encode image as PNG.")
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def analyze_image(
    processed_img: np.ndarray,
    cfg: dict,
    progress_callback=None,
) -> str:
    """
    Send the processed papyrus image to a local Ollama multimodal model
    and return the AI's scholarly analysis.

    Parameters
    ----------
    processed_img : np.ndarray
        The output of the processing pipeline.
    cfg : dict
        Full application configuration (config.json content).
    progress_callback : callable | None
        Optional function(str) for status updates.

    Returns
    -------
    str
        The AI response text.
    """
    if not _OLLAMA_AVAILABLE:
        return "Error: the 'ollama' package is not installed. Run: pip install ollama"

    def _report(msg: str):
        if progress_callback:
            progress_callback(msg)

    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("ollama_model", "llava")
    host = ai_cfg.get("ollama_host", "http://localhost:11434")
    system_prompt = ai_cfg.get("system_prompt", "")
    user_prompt = ai_cfg.get("user_prompt_template", "Analyze this papyrus image.")

    _report(f"Encoding image for AI model '{model}'…")
    img_b64 = _image_to_base64_png(processed_img)

    _report("Connecting to Ollama…")
    client = ollama.Client(host=host)

    # Check whether the model is available locally
    try:
        available_models = [m.model for m in client.list().models]
    except Exception as e:
        return (
            f"Could not connect to Ollama at {host}.\n"
            f"Please make sure Ollama is running.\n\nError: {e}"
        )

    if model not in available_models:
        # Try prefix match (e.g. "llava" matches "llava:latest")
        matches = [m for m in available_models if m.startswith(model)]
        if matches:
            model = matches[0]
        else:
            return (
                f"Model '{model}' is not available locally.\n"
                f"Pull it first with:  ollama pull {model}\n\n"
                f"Available models: {', '.join(available_models) or 'none'}"
            )

    _report(f"Sending image to {model} for analysis…")
    try:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({
            "role": "user",
            "content": user_prompt,
            "images": [img_b64],
        })

        response = client.chat(model=model, messages=messages)
        result = response.message.content
    except Exception as e:
        return f"AI analysis failed: {e}"

    _report("AI analysis complete.")
    return result


def analyze_image_stream(
    processed_img: np.ndarray,
    cfg: dict,
    token_callback,
    done_callback=None,
) -> None:
    """
    Streaming variant of analyze_image.
    Calls token_callback(str) for each incoming token.
    Calls done_callback() when the stream ends.
    Run from a background thread.
    """
    if not _OLLAMA_AVAILABLE:
        token_callback("Error: 'ollama' package is not installed.")
        if done_callback:
            done_callback()
        return

    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("ollama_model", "llava")
    host = ai_cfg.get("ollama_host", "http://localhost:11434")
    system_prompt = ai_cfg.get("system_prompt", "")
    user_prompt = ai_cfg.get("user_prompt_template", "Analyze this papyrus image.")

    token_callback("Encoding image…\n")
    img_b64 = _image_to_base64_png(processed_img)

    client = ollama.Client(host=host)
    try:
        available_models = [m.model for m in client.list().models]
    except Exception as e:
        token_callback(f"Cannot connect to Ollama at {host}.\nError: {e}\n")
        if done_callback:
            done_callback()
        return

    if model not in available_models:
        matches = [m for m in available_models if m.startswith(model)]
        if matches:
            model = matches[0]
        else:
            token_callback(
                f"Model '{model}' is not available locally.\n"
                f"Run:  ollama pull {model}\n\n"
                f"Available: {', '.join(available_models) or 'none'}"
            )
            if done_callback:
                done_callback()
            return

    messages: list[dict] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({
        "role": "user",
        "content": user_prompt,
        "images": [img_b64],
    })

    try:
        stream = client.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.message.content
            if token:
                token_callback(token)
    except Exception as e:
        token_callback(f"\nStreaming error: {e}")

    if done_callback:
        done_callback()


def research_character(
    region_img: np.ndarray,
    cfg: dict,
    token_callback,
    done_callback=None,
) -> None:
    """
    Identify a clicked character/word region and stream scholarly cross-referenced research.

    token_callback(str) receives each token as it arrives.
    done_callback() is called when streaming ends.
    Run from a background thread.
    """
    if not _OLLAMA_AVAILABLE:
        token_callback("Error: 'ollama' package is not installed.")
        if done_callback:
            done_callback()
        return

    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("ollama_model", "llava")
    host = ai_cfg.get("ollama_host", "http://localhost:11434")

    img_b64 = _image_to_base64_png(region_img)
    client = ollama.Client(host=host)

    try:
        available_models = [m.model for m in client.list().models]
    except Exception as e:
        token_callback(f"Cannot connect to Ollama: {e}")
        if done_callback:
            done_callback()
        return

    if model not in available_models:
        matches = [m for m in available_models if m.startswith(model)]
        model = matches[0] if matches else model

    system = (
        "You are a leading papyrologist specialising in Herculaneum papyri and ancient Greek manuscripts. "
        "When shown a cropped, processed image of a papyrus fragment you:\n"
        "1. Identify any visible characters or words and give the Greek text with full diacritics.\n"
        "2. Provide the English translation.\n"
        "3. Give morphological analysis (part of speech, lemma, case/tense/number where relevant).\n"
        "4. Cross-reference: name other Herculaneum papyri (PHerc. numbers) or ancient texts "
        "where this exact word or very similar forms have been attested, with brief scholarly notes.\n"
        "5. Flag uncertain or partially legible letters using [ ] for lacunae and \u27e8 \u27e9 for supplements.\n"
        "Be precise and scholarly, but keep each section clearly labelled."
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                "Identify the character or word visible in this papyrus region. "
                "Structure your response with these exact headings:\n"
                "**Greek Text**\n**Translation**\n**Morphology**\n**Cross-References**"
            ),
            "images": [img_b64],
        },
    ]

    try:
        stream = client.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.message.content
            if token:
                token_callback(token)
    except Exception as e:
        token_callback(f"\nError: {e}")

    if done_callback:
        done_callback()


def analyze_section_stream(
    section_img: np.ndarray,
    rect: dict,
    cfg: dict,
    token_callback,
    done_callback=None,
) -> None:
    """
    Stream a comprehensive contextual analysis of a user-selected papyrus section.
    Covers layout, full transcription, translation, damage, text type, scribal
    features, and scholarly parallels. Run from a background thread.
    """
    if not _OLLAMA_AVAILABLE:
        token_callback("Error: 'ollama' package is not installed.")
        if done_callback:
            done_callback()
        return

    ai_cfg = cfg.get("ai", {})
    model = ai_cfg.get("ollama_model", "llava")
    host  = ai_cfg.get("ollama_host", "http://localhost:11434")

    img_b64 = _image_to_base64_png(section_img)
    client  = ollama.Client(host=host)

    try:
        available_models = [m.model for m in client.list().models]
    except Exception as e:
        token_callback(f"Cannot connect to Ollama: {e}")
        if done_callback:
            done_callback()
        return

    if model not in available_models:
        matches = [m for m in available_models if m.startswith(model)]
        model = matches[0] if matches else model

    system = (
        "You are a master papyrologist and classical scholar specialising in Herculaneum papyri. "
        "You are examining a selected section of a processed papyrus fragment. "
        "Provide a comprehensive contextual analysis under these headings:\n\n"
        "**1. Layout** \u2014 describe columns, text lines, line spacing, margins, reading direction.\n"
        "**2. Transcription** \u2014 every legible character using scholarly notation: "
        "[ ] for lacunae, \u27e8 \u27e9 for conjectural supplements, { } for deletions, "
        "underdot (\u1e43) for uncertain letters.\n"
        "**3. Translation** \u2014 translate all intelligible words, phrases, or sentences.\n"
        "**4. Text Type** \u2014 genre identification: hexameter verse, prose treatise, "
        "epistle, commentary, list, marginalia, etc.\n"
        "**5. Damage Assessment** \u2014 carbonisation, material loss, fibre distortion; "
        "estimate the readable percentage of the section.\n"
        "**6. Scribal Features** \u2014 corrections, supralinear additions, paragraphoi, "
        "diples, punctuation marks, column dividers.\n"
        "**7. Scholarly Context** \u2014 possible author or work; parallels with known "
        "Herculaneum texts (Philodemus, Epicurean works, specific PHerc. rolls).\n\n"
        "Be methodical and scholarly. State confidence levels for each identification."
    )

    w = rect.get("w", "?")
    h = rect.get("h", "?")
    x = rect.get("x", "?")
    y = rect.get("y", "?")

    user_prompt = (
        f"Analyse this selected papyrus section ({w}\u00d7{h} px at position ({x}, {y}) "
        "in the full fragment). Work through each of the seven headings in order, "
        "beginning with the transcription."
    )

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_prompt, "images": [img_b64]},
    ]

    try:
        stream = client.chat(model=model, messages=messages, stream=True)
        for chunk in stream:
            token = chunk.message.content
            if token:
                token_callback(token)
    except Exception as e:
        token_callback(f"\nError: {e}")

    if done_callback:
        done_callback()
