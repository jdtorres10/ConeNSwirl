"""Order normalization key order and validation (stdlib only; run: python3 test_order_normalize.py)."""

from __future__ import annotations

import unittest

from build_menu import validate_and_normalize_order


def _swirl_raw(order_type: str) -> dict:
    return {
        "order_type": order_type,
        "cone_type": "Plain",
        "filling": "No Filling",
        "base": "Vanilla",
        "premium_blend": "No Premium Blend",
        "standard_blend": "No Blend",
        "extra_blend": "No Extra Blend",
        "stick_em": "No Stick 'Em",
        "drizzle": "No Drizzle",
    }


class TestOrderNormalize(unittest.TestCase):
    def test_cone_n_swirl_key_order(self) -> None:
        raw = _swirl_raw("cone_n_swirl")
        n, err = validate_and_normalize_order(raw)
        self.assertIsNone(err)
        self.assertEqual(
            list(n.keys()),
            [
                "order_type",
                "cone_type",
                "filling",
                "base",
                "premium_blend",
                "standard_blend",
                "extra_blend",
                "stick_em",
                "drizzle",
            ],
        )

    def test_cup_n_swirl_key_order(self) -> None:
        raw = _swirl_raw("cup_n_swirl")
        n, err = validate_and_normalize_order(raw)
        self.assertIsNone(err)
        self.assertEqual(
            list(n.keys()),
            [
                "order_type",
                "filling",
                "base",
                "premium_blend",
                "standard_blend",
                "extra_blend",
                "stick_em",
                "drizzle",
                "cone_type",
            ],
        )
        self.assertIsNone(n["cone_type"])

    def test_cone_only_key_order(self) -> None:
        raw = {
            "order_type": "cone_only",
            "cone_type": "Oreo",
            "filling": "Nutella",
            "drizzle": "Chocolate",
        }
        n, err = validate_and_normalize_order(raw)
        self.assertIsNone(err)
        self.assertEqual(
            list(n.keys()),
            [
                "order_type",
                "cone_type",
                "filling",
                "drizzle",
                "base",
                "premium_blend",
                "standard_blend",
                "extra_blend",
                "stick_em",
            ],
        )
        self.assertIsNone(n["base"])

    def test_invalid_order_type(self) -> None:
        raw = _swirl_raw("cone_n_swirl")
        raw["order_type"] = "invalid"
        n, err = validate_and_normalize_order(raw)
        self.assertIsNone(n)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
