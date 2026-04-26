"""Structured build-your-own menu (matches knowledge_base/menu_details.txt)."""

from __future__ import annotations

PREMIUM_BLENDS = frozenset({
    "Brownie",
    "Strawberry",
    "Blueberry",
    "Mango",
})

BUILD_MENU = {
    "steps": [
        {
            "id": "container",
            "label": "Cone or cup?",
            "multi": False,
            "options": [
                {"value": "cone", "label": "Chimney cone"},
                {"value": "cup", "label": "Cup (same build, no cone)"},
            ],
        },
        {
            "id": "cone_type",
            "label": "Cone flavor",
            "multi": False,
            "only_if": {"container": "cone"},
            "options": [
                {"value": "Cinnamon Sugar", "label": "Cinnamon Sugar"},
                {"value": "Oreo", "label": "Oreo"},
                {"value": "Graham Cracker", "label": "Graham Cracker"},
                {"value": "Plain", "label": "Plain"},
            ],
        },
        {
            "id": "base",
            "label": "Ice cream base",
            "multi": False,
            "options": [
                {"value": "Vanilla", "label": "Vanilla"},
                {"value": "Chocolate", "label": "Chocolate"},
            ],
        },
        {
            "id": "filling",
            "label": "Filling (optional)",
            "multi": False,
            "options": [
                {"value": "Nutella", "label": "Nutella"},
                {"value": "Cookie Butter", "label": "Cookie Butter"},
                {"value": "Peanut Butter", "label": "Peanut Butter"},
                {"value": "No Filling", "label": "No filling"},
            ],
        },
        {
            "id": "blends",
            "label": "Blend mix-ins",
            "multi": True,
            "min": 1,
            "help": "Pick one or more. Brownie, Strawberry, Blueberry, and Mango are premium (+$1.50 each).",
            "options": [
                {"value": "Coffee", "label": "Coffee", "premium": False},
                {"value": "Mazapan", "label": "Mazapan", "premium": False},
                {"value": "Oreo", "label": "Oreo", "premium": False},
                {"value": "Cookie Dough", "label": "Cookie Dough", "premium": False},
                {"value": "Nutter Butter", "label": "Nutter Butter", "premium": False},
                {"value": "Pecans", "label": "Pecans", "premium": False},
                {"value": "Walnuts", "label": "Walnuts", "premium": False},
                {"value": "Almonds", "label": "Almonds", "premium": False},
                {"value": "Reeses", "label": "Reeses", "premium": False},
                {"value": "Sprinkles", "label": "Sprinkles", "premium": False},
                {"value": "Pretzels", "label": "Pretzels", "premium": False},
                {"value": "Graham Cracker", "label": "Graham Cracker (blend)", "premium": False},
                {"value": "Brownie", "label": "Brownie (+$1.50)", "premium": True},
                {"value": "Strawberry", "label": "Strawberry (+$1.50)", "premium": True},
                {"value": "Blueberry", "label": "Blueberry (+$1.50)", "premium": True},
                {"value": "Mango", "label": "Mango (+$1.50)", "premium": True},
            ],
        },
        {
            "id": "stickems",
            "label": "Stick'em toppings",
            "multi": True,
            "min": 0,
            "help": "Optional — pick any or tap None.",
            "options": [
                {"value": "Oreo", "label": "Oreo"},
                {"value": "Nutter Butter", "label": "Nutter Butter"},
                {"value": "Pretzel Rod", "label": "Pretzel Rod"},
                {"value": "Reeses", "label": "Reeses"},
                {"value": "Graham Cracker", "label": "Graham Cracker (Stick'em)"},
                {"value": "Kit Kat", "label": "Kit Kat"},
                {"value": "Nilla Wafers", "label": "Nilla Wafers"},
            ],
        },
        {
            "id": "drizzles",
            "label": "Drizzle",
            "multi": True,
            "min": 1,
            "help": "Pick one or more ($0.50 each).",
            "options": [
                {"value": "Chocolate", "label": "Chocolate"},
                {"value": "Strawberry", "label": "Strawberry"},
                {"value": "Salted Caramel", "label": "Salted Caramel"},
                {"value": "Caramel", "label": "Caramel"},
            ],
        },
    ],
}


def _allowed_values(step_id: str) -> frozenset[str]:
    for step in BUILD_MENU["steps"]:
        if step["id"] == step_id:
            return frozenset(o["value"] for o in step["options"])
    return frozenset()


def validate_and_normalize_order(raw: dict) -> tuple[dict | None, str | None]:
    """
    Returns (normalized_order, error_message).
    normalized_order is JSON-serializable for the LLM.
    """
    if not isinstance(raw, dict):
        return None, "order must be an object"

    container = raw.get("container")
    if container not in ("cone", "cup"):
        return None, "container must be cone or cup"

    cone_type = raw.get("cone_type")
    if container == "cone":
        allowed = _allowed_values("cone_type")
        if cone_type not in allowed:
            return None, "invalid or missing cone_type for cone order"
    else:
        cone_type = None
        if cone_type is not None and raw.get("cone_type") not in (None, ""):
            pass  # ignore stray cone_type for cup

    base = raw.get("base")
    if base not in _allowed_values("base"):
        return None, "invalid or missing base"

    filling = raw.get("filling")
    if filling not in _allowed_values("filling"):
        return None, "invalid or missing filling"

    blends = raw.get("blends")
    if not isinstance(blends, list) or len(blends) < 1:
        return None, "pick at least one blend"
    allowed_blends = _allowed_values("blends")
    norm_blends = []
    seen = set()
    for b in blends:
        if not isinstance(b, str) or b not in allowed_blends:
            return None, f"invalid blend: {b!r}"
        if b in seen:
            continue
        seen.add(b)
        norm_blends.append({"name": b, "premium": b in PREMIUM_BLENDS})

    stickems = raw.get("stickems")
    if stickems is None:
        stickems = []
    if not isinstance(stickems, list):
        return None, "stickems must be a list"
    allowed_st = _allowed_values("stickems")
    norm_st = []
    seen_st = set()
    for s in stickems:
        if not isinstance(s, str) or s not in allowed_st:
            return None, f"invalid Stick'em: {s!r}"
        if s in seen_st:
            continue
        seen_st.add(s)
        norm_st.append(s)

    drizzles = raw.get("drizzles")
    if not isinstance(drizzles, list) or len(drizzles) < 1:
        return None, "pick at least one drizzle"
    allowed_dr = _allowed_values("drizzles")
    norm_dr = []
    seen_dr = set()
    for d in drizzles:
        if not isinstance(d, str) or d not in allowed_dr:
            return None, f"invalid drizzle: {d!r}"
        if d in seen_dr:
            continue
        seen_dr.add(d)
        norm_dr.append(d)

    return {
        "container": container,
        "cone_type": cone_type,
        "base": base,
        "filling": filling,
        "blends": norm_blends,
        "stickems": norm_st,
        "drizzles": norm_dr,
    }, None
