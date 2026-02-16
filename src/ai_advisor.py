"""
AI Safety Advisor for TigerSafe.

Implements a Claude-powered agentic AI that:
- Accesses crime data, shuttle info, and route analysis via RAG
- Explains safety risks in an informative (not alarming) way
- Provides actionable recommendations
- Uses the Annual Security Report for context

API keys are loaded from .env file.
"""

import json
import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    _HAS_ANTHROPIC = False

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Well-known campus locations for geocoding
CAMPUS_LOCATIONS = {
    "memorial union": (38.9407, -92.3280),
    "mem union": (38.9407, -92.3280),
    "jesse hall": (38.9395, -92.3290),
    "jesse": (38.9395, -92.3290),
    "ellis library": (38.9410, -92.3305),
    "library": (38.9410, -92.3305),
    "ellis": (38.9410, -92.3305),
    "lafferre hall": (38.9422, -92.3258),
    "lafferre": (38.9422, -92.3258),
    "engineering": (38.9418, -92.3250),
    "engineering building": (38.9418, -92.3250),
    "rec center": (38.9380, -92.3310),
    "student recreation": (38.9380, -92.3310),
    "rec complex": (38.9380, -92.3310),
    "student center": (38.9390, -92.3275),
    "mizzou student center": (38.9390, -92.3275),
    "strickland hall": (38.9412, -92.3278),
    "strickland": (38.9412, -92.3278),
    "tate hall": (38.9416, -92.3292),
    "tate": (38.9416, -92.3292),
    "middlebush hall": (38.9405, -92.3298),
    "middlebush": (38.9405, -92.3298),
    "speakers circle": (38.9402, -92.3295),
    "neff hall": (38.9408, -92.3265),
    "neff": (38.9408, -92.3265),
    "townsend hall": (38.9425, -92.3270),
    "townsend": (38.9425, -92.3270),
    "hearnes center": (38.9365, -92.3255),
    "hearnes": (38.9365, -92.3255),
    "faurot field": (38.9360, -92.3320),
    "faurot": (38.9360, -92.3320),
    "mizzou arena": (38.9370, -92.3335),
    "arena": (38.9370, -92.3335),
    "discovery hall": (38.9430, -92.3240),
    "discovery": (38.9430, -92.3240),
    "chemistry building": (38.9415, -92.3275),
    "chemistry": (38.9415, -92.3275),
    "columns": (38.9400, -92.3290),
    "the columns": (38.9400, -92.3290),
    "francis quadrangle": (38.9400, -92.3295),
    "the quad": (38.9400, -92.3295),
    "carnahan quad": (38.9415, -92.3288),
    "switzler hall": (38.9400, -92.3302),
    "switzler": (38.9400, -92.3302),
    "tiger hotel": (38.9462, -92.3295),
    "downtown": (38.9470, -92.3290),
    "broadway": (38.9460, -92.3290),
    "flat branch park": (38.9490, -92.3320),
    "hospital": (38.9355, -92.3295),
    "university hospital": (38.9355, -92.3295),
    "research park": (38.9450, -92.3250),
    "reactor": (38.9455, -92.3245),
}


SYSTEM_PROMPT = """You are TigerSafe, an AI safety advisor for University of Missouri (Mizzou) students, faculty, and staff. Your purpose is to inform and guide — never to alarm or scare.

## Your Role
You help people navigate campus and the surrounding Columbia, MO area safely by:
1. Explaining route safety analysis results in plain language
2. Providing context about crime patterns and what they mean practically
3. Offering actionable safety recommendations tailored to the situation
4. Suggesting alternatives when risks are elevated (different routes, shuttle options, timing)

## Guidelines
- **Be honest but measured**: Present facts without sensationalizing. "This area has seen 3 theft reports in the past month" is better than "This area is dangerous."
- **Focus on what they can control**: Practical advice like keeping valuables out of sight, traveling with others at night, using well-lit paths.
- **Consider context**: A theft-heavy area at 2pm is very different from the same area at 2am. Mode of travel matters — a driver faces different risks than a pedestrian.
- **Cite data**: Reference specific numbers from the analysis. "Based on 5 recent incidents within 200m of this route" is more trustworthy than vague warnings.
- **Recommend, don't command**: "You might consider..." or "A safer alternative would be..." rather than "Do not go there."
- **Include positive factors**: Mention emergency phones, well-patrolled areas, well-lit paths — not just negatives.
- **Shuttle awareness**: When relevant, mention Tiger Line shuttle availability, who can ride (MU ID holders), and where to track them in real time.

## Risk Score Interpretation
- 0-20 (Very Safe): Well-lit, low crime, near emergency phones. Reassure the user.
- 21-40 (Safe): Generally safe with minor concerns. Standard awareness.
- 41-60 (Moderate): Exercise normal caution. Note specific concerns.
- 61-80 (Higher Risk): Be alert. Suggest specific precautions or alternatives.
- 81-100 (High Risk): Strongly recommend alternatives, especially at night.

## Data Sources Available
You have access to:
- Columbia Police Department crime reports (NIBRS-coded)
- MU Police Daily Crime Log
- Traffic stop data (indicates police patrol patterns)
- Shuttle route and schedule data
- Emergency phone locations
- Campus building locations
- The university's Annual Security Report (Clery Act)

Always ground your responses in the actual data provided. Never fabricate statistics.
"""


def _load_clery_context() -> str:
    """Load key excerpts from the Annual Security Report for RAG context.

    Since full PDF parsing requires pdfplumber, this returns a summary
    of key Clery data that can be included in prompts.
    """
    pdf_path = DATA_DIR / "rag_sources" / "2025_Annual_Security_Report.pdf"

    # Try to extract text if pdfplumber is available
    try:
        import pdfplumber
        if pdf_path.exists():
            excerpts = []
            with pdfplumber.open(str(pdf_path)) as pdf:
                # Extract from crime statistics pages (typically pages 140-180)
                for page_num in range(min(len(pdf.pages), 180)):
                    if page_num < 135:
                        continue
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and any(kw in text.lower() for kw in [
                        "criminal offense", "on-campus", "public property",
                        "residence hall", "clery", "vawa", "arrests",
                        "disciplinary", "hate crime"
                    ]):
                        excerpts.append(text[:2000])
                    if len(excerpts) >= 10:
                        break
            if excerpts:
                return "\n---\n".join(excerpts)
    except ImportError:
        pass
    except Exception:
        pass

    # Fallback: provide general Clery context
    return """Annual Security Report Summary (2025):
The University of Missouri publishes annual crime statistics per the Clery Act.
Key data categories include: on-campus crimes, on-campus student housing,
non-campus property, and public property adjacent to campus.
Common reported offenses include theft/larceny, burglary, drug violations,
and liquor law violations. Violent crimes (assault, robbery, sexual offenses)
are less common but do occur, particularly in evening/night hours.
The university maintains emergency blue light phones across campus,
operates the Tiger Line shuttle system, and has 24/7 MUPD patrol coverage."""


def build_route_context(
    routes: list[dict],
    shuttle_info: dict = None,
    hour: int = None,
    mode: str = "walk",
) -> str:
    """Build a RAG context document from route analysis results.

    This is injected into the Claude prompt so the AI can make
    informed, data-backed recommendations.
    """
    if hour is None:
        hour = datetime.now().hour

    parts = []
    parts.append(f"## Current Conditions")
    parts.append(f"- Time: {datetime.now().strftime('%I:%M %p')} (Hour {hour})")
    parts.append(f"- Day: {datetime.now().strftime('%A')}")
    parts.append(f"- Travel mode: {mode}")
    parts.append("")

    for i, route in enumerate(routes):
        risk = route.get("risk_score", {})
        crime = route.get("crime_stats", {})
        rec = route.get("recommendation", f"Route {i+1}")

        parts.append(f"## {rec}")
        parts.append(f"- Distance: {route.get('distance_m', 0):.0f}m")
        parts.append(f"- Estimated time: {route.get('estimated_time_min', 0):.1f} minutes ({mode})")
        parts.append(f"- Risk score: {risk.get('score', 0)}/100 ({risk.get('level', 'Unknown')})")
        parts.append("")

        breakdown = risk.get("breakdown", {})
        parts.append(f"### Risk Breakdown")
        parts.append(f"- Crime density score: {breakdown.get('crime_score', 0)}")
        parts.append(f"- Time period: {breakdown.get('temporal_period', '')}")
        parts.append(f"- Time risk multiplier: {breakdown.get('temporal_multiplier', 1.0)}x")
        parts.append(f"- Recent crime activity score: {breakdown.get('recency_score', 0)}")
        parts.append(f"- Emergency phones nearby: {breakdown.get('emergency_phones_nearby', 0)}")
        parts.append(f"- Patrol level: {breakdown.get('patrol_level', 'unknown')}")
        parts.append(f"- Mode adjustment: {breakdown.get('mode_multiplier', 1.0)}x ({mode})")
        parts.append("")

        parts.append(f"### Crime Statistics Along Route")
        parts.append(f"- Total crimes recorded: {crime.get('total_crimes', 0)}")
        parts.append(f"- Violent crimes: {crime.get('violent_crimes', 0)}")
        parts.append(f"- Crimes in last 30 days: {crime.get('recent_crimes_30d', 0)}")
        parts.append(f"- Crimes in last 7 days: {crime.get('recent_crimes_7d', 0)}")
        parts.append(f"- Average severity: {crime.get('avg_severity', 0):.2f}/1.0")

        by_cat = crime.get("by_category", {})
        if by_cat:
            parts.append(f"- By type: {', '.join(f'{k}: {v}' for k, v in sorted(by_cat.items(), key=lambda x: -x[1]))}")
        parts.append("")

    if shuttle_info:
        parts.append("## Shuttle Information")
        if shuttle_info.get("available"):
            parts.append("- Shuttle service is currently AVAILABLE")
            parts.append(f"- Nearest stop to origin: {shuttle_info.get('nearest_origin_stop', {}).get('name', 'N/A')} ({shuttle_info.get('walk_to_stop_m', 0):.0f}m away)")
            parts.append(f"- Nearest stop to destination: {shuttle_info.get('nearest_dest_stop', {}).get('name', 'N/A')} ({shuttle_info.get('walk_from_stop_m', 0):.0f}m away)")
            parts.append(f"- Available routes: {', '.join(shuttle_info.get('available_routes', []))}")
            parts.append(f"- Eligibility: {shuttle_info.get('eligibility', 'MU ID holders')}")
            parts.append(f"- Real-time tracking: https://tiger.etaspot.net")
        else:
            parts.append("- Shuttle service is NOT currently available")
            parts.append(f"- Reason: {shuttle_info.get('reason', 'Unknown')}")
            if shuttle_info.get("next_service"):
                parts.append(f"- Next service: {shuttle_info['next_service']}")
        parts.append("")

    # Add Clery report context
    clery = _load_clery_context()
    if clery:
        parts.append("## University Safety Report Context")
        parts.append(clery[:3000])  # Limit size

    return "\n".join(parts)


def get_route_analysis(
    routes: list[dict],
    shuttle_info: dict = None,
    hour: int = None,
    mode: str = "walk",
    api_key: str = None,
) -> str:
    """Get AI-powered analysis of route safety.

    Sends route data to Claude and returns a natural language
    safety analysis with recommendations.

    Args:
        routes: List of scored route dicts.
        shuttle_info: Shuttle availability info (optional).
        hour: Hour of day.
        mode: Travel mode.
        api_key: Anthropic API key (falls back to env var).

    Returns:
        String with the AI's safety analysis and recommendations.
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _HAS_ANTHROPIC:
        return _generate_fallback_analysis(routes, shuttle_info, hour, mode)

    context = build_route_context(routes, shuttle_info, hour, mode)
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    user_prompt = f"""Based on the following route analysis data, provide a clear, helpful safety briefing for someone about to travel. Include:

1. A brief summary of which route you recommend and why
2. Key safety factors to be aware of (cite specific numbers)
3. Practical tips based on the crime types in the area (e.g., if theft is common, keep valuables hidden)
4. Whether the shuttle is a good option
5. An overall recommendation (go ahead, take precautions, consider waiting, etc.)

Keep it concise (3-4 short paragraphs). Be informative, not alarming.

---
{context}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
    except Exception:
        return _generate_fallback_analysis(routes, shuttle_info, hour, mode)


def chat_with_advisor(
    user_message: str,
    conversation_history: list[dict],
    route_context: str = "",
    api_key: str = None,
) -> tuple[str, list[dict]]:
    """Have a conversation with the TigerSafe AI advisor.

    Args:
        user_message: The user's message.
        conversation_history: Previous messages.
        route_context: Current route analysis context for RAG.
        api_key: Anthropic API key.

    Returns:
        Tuple of (response_text, updated_history).
    """
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or not _HAS_ANTHROPIC:
        msg = "Please set your Anthropic API key in the .env file or sidebar to use the AI advisor."
        if not _HAS_ANTHROPIC:
            msg = "The `anthropic` package is not installed. Install it with `pip install anthropic` to use the AI advisor."
        return msg, conversation_history

    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

    # Build system prompt with RAG context
    system = SYSTEM_PROMPT
    if route_context:
        system += f"\n\n## Current Route Analysis Data\n{route_context}"

    conversation_history.append({"role": "user", "content": user_message})

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=system,
            messages=conversation_history,
        )

        assistant_text = response.content[0].text
        conversation_history.append({"role": "assistant", "content": assistant_text})
        return assistant_text, conversation_history

    except Exception as e:
        error_msg = f"AI advisor error: {str(e)}"
        conversation_history.append({"role": "assistant", "content": error_msg})
        return error_msg, conversation_history


def _generate_fallback_analysis(
    routes: list[dict],
    shuttle_info: dict = None,
    hour: int = None,
    mode: str = "walk",
) -> str:
    """Generate a basic analysis without the AI API.

    Used when no API key is configured.
    """
    if not routes:
        return "No routes available to analyze."

    if hour is None:
        hour = datetime.now().hour

    best = routes[0]
    risk = best.get("risk_score", {})
    crime = best.get("crime_stats", {})
    score = risk.get("score", 0)
    level = risk.get("level", "Unknown")

    lines = []
    lines.append(f"**Route Analysis** ({mode.title()})")
    lines.append(f"")
    lines.append(f"**Recommended route**: {best.get('distance_m', 0):.0f}m, ~{best.get('estimated_time_min', 0):.0f} min")
    lines.append(f"**Risk level**: {level} ({score}/100)")
    lines.append(f"")

    total = crime.get("total_crimes", 0)
    violent = crime.get("violent_crimes", 0)
    recent = crime.get("recent_crimes_30d", 0)

    if total > 0:
        lines.append(f"There are {total} recorded crime incidents along this route ({violent} violent). {recent} occurred in the last 30 days.")
        by_cat = crime.get("by_category", {})
        if by_cat:
            top = sorted(by_cat.items(), key=lambda x: -x[1])[:3]
            lines.append(f"Most common types: {', '.join(f'{k} ({v})' for k, v in top)}.")
    else:
        lines.append("No recorded crime incidents along this route.")

    phones = risk.get("breakdown", {}).get("emergency_phones_nearby", 0)
    if phones > 0:
        lines.append(f"{phones} emergency phone(s) are located along the route.")

    if hour >= 22 or hour < 6:
        lines.append("")
        lines.append("**Note**: It is currently late night. Consider traveling with a companion or using the shuttle if available.")

    if shuttle_info and shuttle_info.get("available"):
        lines.append("")
        lines.append(f"**Shuttle option**: Tiger Line is currently running. Nearest stop is {shuttle_info.get('nearest_origin_stop', {}).get('name', 'nearby')}. Track at tiger.etaspot.net")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        lines.append("")
        lines.append("*Set ANTHROPIC_API_KEY in .env for detailed AI-powered safety analysis.*")

    return "\n".join(lines)


def resolve_location(name: str) -> tuple[float, float] | None:
    """Try to resolve a campus location name to (lat, lon) coordinates."""
    key = name.strip().lower()
    if key in CAMPUS_LOCATIONS:
        return CAMPUS_LOCATIONS[key]

    # Fuzzy match
    for loc_key, coords in CAMPUS_LOCATIONS.items():
        if loc_key in key or key in loc_key:
            return coords

    return None
