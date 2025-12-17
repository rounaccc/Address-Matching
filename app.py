"""
Streamlit app: compare two London addresses and return a confidence score
using the Gemini (free tier) API.

Setup
- Install deps: `pip install streamlit google-generativeai`
- Set env var: `export GEMINI_API_KEY=your_api_key` (or use a .env loader).
- Run: `streamlit run app.py`
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import google.generativeai as genai


# ----------------------------
# Domain types and utilities
# ----------------------------
@dataclass
class MatchResult:
    confidence: float
    decision: str
    reasoning: str


def configure_genai(api_key: str) -> genai.GenerativeModel:
    """Configure the Gemini client and return a model instance."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def parse_json_response(raw_text: str) -> Optional[MatchResult]:
    """Attempt to parse the model's JSON response into a MatchResult."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    try:
        return MatchResult(
            confidence=float(payload.get("confidence", 0)),
            decision=str(payload.get("decision", "")).strip(),
            reasoning=str(payload.get("reasoning", "")).strip(),
        )
    except (TypeError, ValueError):
        return None


def score_addresses(model: genai.GenerativeModel, addr_a: str, addr_b: str) -> MatchResult:
    """Call Gemini to score whether two London addresses are the same place."""
    system_prompt = (
        "You are a UK address-matching assistant focused on London. "
        "Given two addresses, return a JSON object only (no prose) with:\n"
        "- confidence: number 0-100 (higher = more likely same place).\n"
        "- decision: 'match' or 'no_match'.\n"
        "- reasoning: one short sentence on key similarities/differences.\n\n"
        "Instructions:\n"
        "- Normalize street abbreviations (st, rd, ave), flat/unit labels.\n"
        "- Assume both addresses are in the SAME postcode (postcode not provided).\n"
        "- Be robust to missing components (building names, boroughs) and shortforms.\n"
        "- Focus on house/flat numbers, street names, locality/borough cues for evidence.\n"
        "- Penalize clearly different numbers/streets or mismatching locality signals.\n"
        "- Keep response strictly as JSON and concise."
    )

    prompt = (
        f"{system_prompt}\n\n"
        f"Address A: {addr_a}\n"
        f"Address B: {addr_b}"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    print(response)
    raw_text = response.text or ""
    parsed = parse_json_response(raw_text)

    if parsed is None:
        # Fallback in case the model drifted from the expected schema.
        parsed = MatchResult(
            confidence=0.0,
            decision="no_match",
            reasoning="Could not parse model response; defaulting to no match.",
        )

    # Clamp confidence to 0-100 for sanity.
    parsed.confidence = max(0.0, min(100.0, parsed.confidence))
    return parsed


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="London Address Matcher", page_icon="📍", layout="centered")
st.title("📍 London Address Matcher")
st.caption("Compare two London addresses (with shortforms) using Gemini.")

# Input form
with st.form("address_form"):
    addr1 = st.text_area("Address 1", height=100, placeholder="e.g., Flat 2B, 10 Downing St, Westminster, SW1A 2AA")
    addr2 = st.text_area("Address 2", height=100, placeholder="e.g., 10 Downing Street, London SW1A 2AA")
    submitted = st.form_submit_button("Check Match")

if submitted:
    api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        st.error("Missing GEMINI_API_KEY environment variable.")
    elif not addr1.strip() or not addr2.strip():
        st.warning("Please enter both addresses.")
    else:
        with st.spinner("Scoring with Gemini..."):
            model = configure_genai(api_key)
            result = score_addresses(model, addr1.strip(), addr2.strip())

        st.success("Result")
        st.metric("Confidence (0-100)", f"{result.confidence:.1f}")
        st.write(f"Decision: **{result.decision}**")
        st.write(f"Reasoning: {result.reasoning}")

st.sidebar.header("How to use")
st.sidebar.markdown(
    """
    1) Set `GEMINI_API_KEY` in your environment.
    2) Enter two London addresses (shortforms OK).
    3) Submit to get a confidence score and decision.
    """
)

