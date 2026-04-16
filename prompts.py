# prompts.py - CriticChain Prompt Loader (Factory Pattern)
# =============================================================================
# This module automatically loads the appropriate prompt set based on:
#   1. Language: CRITIC_LANGUAGE env var (default: "en")
#   2. Tier: Enterprise (if available) or Lite (fallback)
#
# File naming convention:
#   - prompts_{lang}_enterprise.py  (e.g., prompts_en_enterprise.py)
#   - prompts_lite_{lang}.py        (e.g., prompts_lite_en.py)
# =============================================================================

import os
import sys
import importlib

# Determine target language (default: English for global release)

CRITIC_LANGUAGE = os.getenv("CRITIC_LANGUAGE", "en")
# Determine target tier (default: auto - tries Enterprise then Lite)
CRITIC_TIER = os.getenv("CRITIC_TIER", "auto").lower()

def _load_prompts(lang: str, tier_preference: str):
    """Load the appropriate prompt module based on language and tier availability."""
    # Priority order for Enterprise
    enterprise_modules = [
        f"prompts_{lang}_enterprise",  # e.g., prompts_en_enterprise
        "prompts_enterprise",           # legacy fallback
    ]
    
    # Priority order for Lite
    lite_modules = [
        f"prompts_lite_{lang}",         # e.g., prompts_lite_en
        "prompts_lite",                  # legacy fallback
    ]
    
    # Helper to try loading a list of modules
    def try_load(modules, tier_name):
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                return tier_name, module
            except ImportError:
                continue
        return None

    # Logic based on tier preference
    if tier_preference == "enterprise":
        result = try_load(enterprise_modules, "enterprise")
        if result: return result
        raise ImportError(f"CRITIC_TIER='enterprise' set, but no enterprise prompts found for lang='{lang}'")

    elif tier_preference == "lite":
        result = try_load(lite_modules, "lite")
        if result: return result
        raise ImportError(f"CRITIC_TIER='lite' set, but no lite prompts found for lang='{lang}'")

    else: # auto
        # Try Enterprise first
        result = try_load(enterprise_modules, "enterprise")
        if result: return result
        
        # Fallback to Lite
        result = try_load(lite_modules, "lite")
        if result: return result
    
    raise ImportError("No prompt module found! Please ensure prompts_lite.py exists.")

_PROMPT_VERSION, _PROMPT_MODULE = _load_prompts(CRITIC_LANGUAGE, CRITIC_TIER)

# Re-export all symbols from the loaded module
for _name in dir(_PROMPT_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_PROMPT_MODULE, _name)

def get_prompt_version():
    """Returns which prompt version is currently loaded."""
    return _PROMPT_VERSION

def get_prompt_language():
    """Returns which language prompts are loaded."""
    return CRITIC_LANGUAGE

# Log which version is loaded (only on first import)
if "prompts" not in sys.modules or sys.modules["prompts"].__name__ == __name__:
    lang_flag = "🇺🇸" if CRITIC_LANGUAGE == "en" else "🇯🇵"
    tier_flag = "🔐 Enterprise" if _PROMPT_VERSION == "enterprise" else "📦 Lite (OSS)"
    print(f"{lang_flag} CriticChain: Using {tier_flag} prompts ({CRITIC_LANGUAGE.upper()})")


