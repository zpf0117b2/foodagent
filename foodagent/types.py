"""Core data types shared across FoodAgent modules."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

NUTRIENT_FIELDS = ("calories", "mass", "fat", "carb", "protein")
MACRO_FIELDS = ("calories", "fat", "carb", "protein")


@dataclass
class Nutrients:
    """Calories in kcal; mass / fat / carb / protein in grams."""

    calories: float = 0.0
    mass: float = 0.0
    fat: float = 0.0
    carb: float = 0.0
    protein: float = 0.0

    def __add__(self, other: "Nutrients") -> "Nutrients":
        return Nutrients(*(getattr(self, f) + getattr(other, f) for f in NUTRIENT_FIELDS))

    def scale(self, k: float) -> "Nutrients":
        return Nutrients(*(getattr(self, f) * k for f in NUTRIENT_FIELDS))

    def to_dict(self) -> Dict[str, float]:
        return {f: float(getattr(self, f)) for f in NUTRIENT_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Nutrients":
        return cls(**{f: float(d.get(f, 0.0) or 0.0) for f in NUTRIENT_FIELDS})


@dataclass
class FoodElement:
    """One food component produced by the reasoning agent's element analysis."""

    name: str
    description: str = ""
    bbox: Optional[Tuple[float, float, float, float]] = None  # x1, y1, x2, y2 in original image pixels
    importance: str = "main"  # "main" | "minor"
    visually_clear: bool = False
    density_g_per_cm3: float = 1.0
    estimated_mass_g: float = 0.0
    per_100g: Nutrients = field(default_factory=Nutrients)  # agent internal knowledge (mass field unused)
    category: int = 2  # 1 = main & clear, 2 = main & not clear, 3 = minor / easy
    route: str = ""  # expert actually used: "volume" | "rag" | "llm"
    route_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["per_100g"] = self.per_100g.to_dict()
        return d


@dataclass
class FoodReference:
    """A reference serving retrieved from the nutrition database (RAG expert)."""

    name: str
    serving_qty: float
    serving_unit: str
    serving_weight_g: float
    nutrients: Nutrients  # nutrition of ONE reference serving (serving_qty serving_unit)
    photo: Optional[str] = None  # URL or local path of a reference image, if available
    source: str = ""

    def describe(self) -> str:
        n = self.nutrients
        return (
            f"{self.name}, {self.serving_qty:g} {self.serving_unit} ({self.serving_weight_g:.0f} g): "
            f"{n.calories:.0f} kcal, fat {n.fat:.1f} g, carb {n.carb:.1f} g, protein {n.protein:.1f} g"
        )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["nutrients"] = self.nutrients.to_dict()
        return d


@dataclass
class ElementEstimate:
    element: FoodElement
    nutrients: Nutrients
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"element": self.element.to_dict(), "nutrients": self.nutrients.to_dict(), "details": self.details}


@dataclass
class DishEstimate:
    total: Nutrients
    elements: List[ElementEstimate] = field(default_factory=list)
    response: str = ""
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "elements": [e.to_dict() for e in self.elements],
            "response": self.response,
            "trace": self.trace,
        }
