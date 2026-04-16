"""
Schemas Facade (Switch)

This module automatically loads the appropriate schema definitions based on availability
and the CRITIC_TIER environment variable.

- CRITIC_TIER="enterprise" -> Forces Enterprise schemas (errors if missing)
- CRITIC_TIER="lite"       -> Forces Lite schemas
- CRITIC_TIER="auto"       -> Tries Enterprise, falls back to Lite
"""

import os
import sys

# Determine target tier (default: auto)
CRITIC_TIER = os.getenv("CRITIC_TIER", "auto").lower()

def _load_schemas(tier_preference: str):
    """Load the appropriate schema module based on tier availability."""
    
    if tier_preference == "enterprise":
        try:
            import schemas_enterprise as module
            return "enterprise", module
        except ImportError:
            raise ImportError("CRITIC_TIER='enterprise' set, but schemas_enterprise.py not found.")

    elif tier_preference == "lite":
        try:
            import schemas_lite as module
            return "lite", module
        except ImportError:
            raise ImportError("CRITIC_TIER='lite' set, but schemas_lite.py not found.")

    else: # auto
        # Try Enterprise first
        try:
            import schemas_enterprise as module
            return "enterprise", module
        except ImportError:
            pass
        
        # Fallback to Lite
        try:
            import schemas_lite as module
            return "lite", module
        except ImportError:
            pass

    raise ImportError("No schema module found! Please ensure schemas_lite.py exists.")

_SCHEMA_TIER, _SCHEMA_MODULE = _load_schemas(CRITIC_TIER)

# Re-export all symbols from the loaded module
for _name in dir(_SCHEMA_MODULE):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_SCHEMA_MODULE, _name)

# Log which version is loaded (only on first import)
if "schemas" not in sys.modules or sys.modules["schemas"].__name__ == __name__:
    tier_flag = "🔐 Enterprise" if _SCHEMA_TIER == "enterprise" else "📦 Lite (OSS)"
    print(f"Schemas: Using {tier_flag} definitions")
