TAXONOMY_VERSION = "rust-aspects-v1"
PROMPT_VERSION = "aspect-sentiment-v1"

ASPECT_DESCRIPTIONS = """
Language: ownership, type_system, safety, performance
Experience: learning_curve, compile_time, error_message, debugging
Engineering: maintainability, readability, extensibility, api_design
Ecosystem: package_manager, libraries, framework_support, community
""".strip()

SYSTEM_PROMPT = f"""
You label aspect-level sentiment in Rust community text.
Return JSON only, exactly in this shape:
{{"annotations":[{{"aspect":"performance","class":"positive"}}]}}

Allowed aspects:
{ASPECT_DESCRIPTIONS}

Allowed classes: positive, neutral, negative.
Omit aspects that are not explicitly discussed. Use neutral only when an aspect is explicitly
discussed without clear positive or negative sentiment. The input contains CONTEXT and TARGET.
CONTEXT only helps disambiguation. Label sentiment expressed by TARGET only; never copy sentiment
from CONTEXT. Do not add explanations or additional keys. An empty list is valid.
""".strip()
