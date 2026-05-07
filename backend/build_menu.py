"""Structured build-your-own menu (aligned with knowledge_base/menu_details.txt)."""

from __future__ import annotations

# Premium blend add-ons (per menu; used in help copy and recap context).
PREMIUM_BLEND_NAMES = frozenset({
    "Brownie",
    "Strawberry",
    "Blueberry",
    "Mango",
    "Raspberry",
    "Pineapple",
})

_STANDARD_BLEND_OPTIONS = [
    {"value": "Coffee", "label": "Coffee"},
    {"value": "Mazapan", "label": "Mazapan"},
    {"value": "Oreo", "label": "Oreo"},
    {"value": "Cookie Dough", "label": "Cookie Dough"},
    {"value": "Nutter Butter", "label": "Nutter Butter"},
    {"value": "Pecans", "label": "Pecans"},
    {"value": "Walnuts", "label": "Walnuts"},
    {"value": "Almonds", "label": "Almonds"},
    {"value": "Reeses", "label": "Reeses"},
    {"value": "Sprinkles", "label": "Sprinkles"},
    {"value": "Pretzels", "label": "Pretzels"},
    {"value": "Graham Cracker", "label": "Graham Cracker"},
    {"value": "No Blend", "label": "No Blend"},
]

_EXTRA_BLEND_OPTIONS = [
    {"value": "Coffee", "label": "Coffee"},
    {"value": "Mazapan", "label": "Mazapan"},
    {"value": "Oreo", "label": "Oreo"},
    {"value": "Cookie Dough", "label": "Cookie Dough"},
    {"value": "Nutter Butter", "label": "Nutter Butter"},
    {"value": "Pecans", "label": "Pecans"},
    {"value": "Walnuts", "label": "Walnuts"},
    {"value": "Almonds", "label": "Almonds"},
    {"value": "Reeses", "label": "Reeses"},
    {"value": "Sprinkles", "label": "Sprinkles"},
    {"value": "Pretzels", "label": "Pretzels"},
    {"value": "Graham Cracker", "label": "Graham Cracker"},
    {"value": "No Extra Blend", "label": "No Extra Blend"},
]

_PREMIUM_BLEND_OPTIONS = [
    {"value": "Brownie", "label": "Brownie (+$1.25)"},
    {"value": "Strawberry", "label": "Strawberry (+$1.25)"},
    {"value": "Blueberry", "label": "Blueberry (+$1.25)"},
    {"value": "Mango", "label": "Mango (+$1.25)"},
    {"value": "Raspberry", "label": "Raspberry (+$1.25)"},
    {"value": "Pineapple", "label": "Pineapple (+$1.25)"},
    {"value": "No Premium Blend", "label": "No Premium Blend"},
]

_STICK_EM_OPTIONS = [
    {"value": "Oreo", "label": "Oreo"},
    {"value": "Nutter Butter", "label": "Nutter Butter"},
    {"value": "Pretzel Rod", "label": "Pretzel Rod"},
    {"value": "Reeses", "label": "Reeses"},
    {"value": "Graham Cracker", "label": "Graham Cracker"},
    {"value": "Kit Kat", "label": "Kit Kat"},
    {"value": "Nilla Wafers", "label": "Nilla Wafers"},
    {"value": "No Stick 'Em", "label": "No Stick 'Em"},
]

_DRIZZLE_OPTIONS = [
    {"value": "Chocolate", "label": "Chocolate"},
    {"value": "Strawberry", "label": "Strawberry"},
    {"value": "Salted Caramel", "label": "Salted Caramel"},
    {"value": "Caramel", "label": "Caramel"},
    {"value": "No Drizzle", "label": "No Drizzle"},
]

_CONE_TYPE_OPTIONS = [
    {"value": "Cinnamon Sugar", "label": "Cinnamon Sugar"},
    {"value": "Oreo", "label": "Oreo"},
    {"value": "Graham Cracker", "label": "Graham Cracker"},
    {"value": "Plain", "label": "Plain"},
]

_BASE_OPTIONS = [
    {"value": "Vanilla", "label": "Vanilla"},
    {"value": "Chocolate", "label": "Chocolate"},
]

_FILLING_OPTIONS = [
    {"value": "Nutella", "label": "Nutella"},
    {"value": "Cookie Butter", "label": "Cookie Butter"},
    {"value": "Peanut Butter", "label": "Peanut Butter"},
    {"value": "No Filling", "label": "No Filling"},
]

BUILD_MENU = {
    "steps": [
        {
            "id": "order_type",
            "label": "What are you building?",
            "multi": False,
            "help": "Cone N' Swirl: 8 steps. Cup N' Swirl: 7 steps (cup, no cone). Cone Only: 3 steps (no ice cream).",
            "options": [
                {
                    "value": "cone_n_swirl",
                    "label": "Cone N' Swirl",
                },
                {
                    "value": "cup_n_swirl",
                    "label": "Cup N' Swirl",
                },
                {
                    "value": "cone_only",
                    "label": "Cone Only (no ice cream)",
                },
            ],
        },
        {
            "id": "cone_type",
            "label": "Choose your cone",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cone_only"]},
            "options": _CONE_TYPE_OPTIONS,
        },
        {
            "id": "filling",
            "label": "Choose your filling (optional)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl", "cone_only"]},
            "help": "Spreads are included free on cones and Cone Only. On cups, Nutella, Cookie Butter, or Peanut Butter is +$0.50; No Filling is free.",
            "options": _FILLING_OPTIONS,
        },
        {
            "id": "base",
            "label": "Choose your ice cream base (Swirl Base)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl"]},
            "options": _BASE_OPTIONS,
        },
        {
            "id": "premium_blend",
            "label": "Premium blend (optional, +$1.25 each)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl"]},
            "options": _PREMIUM_BLEND_OPTIONS,
        },
        {
            "id": "standard_blend",
            "label": "Choose your blend (1 free)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl"]},
            "help": "One traditional standard blend is included. Pick any premium blends in the previous step.",
            "options": _STANDARD_BLEND_OPTIONS,
        },
        {
            "id": "extra_blend",
            "label": "Extra standard blend (optional, +$0.75)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl"]},
            "options": _EXTRA_BLEND_OPTIONS,
        },
        {
            "id": "stick_em",
            "label": "Stick'em (optional, +$0.99)",
            "multi": False,
            "only_if": {"order_type": ["cone_n_swirl", "cup_n_swirl"]},
            "options": _STICK_EM_OPTIONS,
        },
        {
            "id": "drizzle",
            "label": "Drizzle (optional, +$0.75)",
            "multi": False,
            "options": _DRIZZLE_OPTIONS,
        },
    ],
}


def _values(options: list[dict]) -> frozenset[str]:
    return frozenset(o["value"] for o in options)


_ALLOWED = {
    "order_type": frozenset({"cone_n_swirl", "cup_n_swirl", "cone_only"}),
    "cone_type": _values(_CONE_TYPE_OPTIONS),
    "base": _values(_BASE_OPTIONS),
    "filling": _values(_FILLING_OPTIONS),
    "standard_blend": _values(_STANDARD_BLEND_OPTIONS),
    "extra_blend": _values(_EXTRA_BLEND_OPTIONS),
    "premium_blend": _values(_PREMIUM_BLEND_OPTIONS),
    "stick_em": _values(_STICK_EM_OPTIONS),
    "drizzle": _values(_DRIZZLE_OPTIONS),
}


def validate_and_normalize_order(raw: dict) -> tuple[dict | None, str | None]:
    """
    Returns (normalized_order, error_message).
    normalized_order is JSON-serializable; keys follow kitchen build order for POS and recap.
    """
    if not isinstance(raw, dict):
        return None, "order must be an object"

    order_type = raw.get("order_type")
    if order_type not in _ALLOWED["order_type"]:
        return None, "order_type must be cone_n_swirl, cup_n_swirl, or cone_only"

    def req_str(key: str, allowed: frozenset[str]) -> tuple[str | None, str | None]:
        v = raw.get(key)
        if not isinstance(v, str) or v not in allowed:
            return None, f"invalid or missing {key}"
        return v, None

    # Keys are ordered kitchen-line / build-card sequence for POS handoff and LLM recap.
    cone_type: str | None = None
    base: str | None = None
    filling: str | None = None
    premium_blend: str | None = None
    standard_blend: str | None = None
    extra_blend: str | None = None
    stick_em: str | None = None
    drizzle: str | None = None

    if order_type in ("cone_n_swirl", "cone_only"):
        v, err = req_str("cone_type", _ALLOWED["cone_type"])
        if err:
            return None, err
        cone_type = v

    v, err = req_str("filling", _ALLOWED["filling"])
    if err:
        return None, err
    filling = v

    if order_type in ("cone_n_swirl", "cup_n_swirl"):
        v, err = req_str("base", _ALLOWED["base"])
        if err:
            return None, err
        base = v
        v, err = req_str("premium_blend", _ALLOWED["premium_blend"])
        if err:
            return None, err
        premium_blend = v
        v, err = req_str("standard_blend", _ALLOWED["standard_blend"])
        if err:
            return None, err
        standard_blend = v
        v, err = req_str("extra_blend", _ALLOWED["extra_blend"])
        if err:
            return None, err
        extra_blend = v
        v, err = req_str("stick_em", _ALLOWED["stick_em"])
        if err:
            return None, err
        stick_em = v

    v, err = req_str("drizzle", _ALLOWED["drizzle"])
    if err:
        return None, err
    drizzle = v

    out: dict = {"order_type": order_type}
    if order_type in ("cone_n_swirl", "cone_only"):
        out["cone_type"] = cone_type
    else:
        out["cone_type"] = None
    out["filling"] = filling
    if order_type in ("cone_n_swirl", "cup_n_swirl"):
        out["base"] = base
        out["premium_blend"] = premium_blend
        out["standard_blend"] = standard_blend
        out["extra_blend"] = extra_blend
        out["stick_em"] = stick_em
        out["drizzle"] = drizzle
    else:
        # Cone only: line order is cone, filling, drizzle; keep null swirl slots for a stable schema.
        out["drizzle"] = drizzle
        out["base"] = None
        out["premium_blend"] = None
        out["standard_blend"] = None
        out["extra_blend"] = None
        out["stick_em"] = None

    return out, None
