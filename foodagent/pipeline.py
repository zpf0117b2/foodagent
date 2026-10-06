"""FoodAgent end-to-end pipeline (Fig. 1) and the holistic Qwen2.5-VL baseline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from . import prompts
from .agent import NutritionReasoningAgent, match_name
from .experts.llm_direct import LLMDirectExpert
from .experts.nutrition_db import NutritionDB
from .experts.rag import RAGExpert
from .experts.volume import MFP3DVolumeEstimator
from .generator import ResponseGenerator
from .jsonutil import clamp, extract_json, to_float
from .types import DishEstimate, ElementEstimate, FoodElement, Nutrients
from .vlm import VLM, prepare_image, user_message


@dataclass
class FoodAgentConfig:
    use_volume_expert: bool = True  # False -> ablation "w/o MFP3D"
    use_rag_expert: bool = True  # False -> ablation "w/o NutritionX"
    volume_fallback: str = "llm"  # where category-1 elements go when the volume expert is disabled / fails
    rag_fallback: str = "llm"  # where category-2 elements go when RAG is disabled / retrieves nothing
    cat1_profile_from_db: bool = True  # per-gram nutrition for category 1 from NutritionX (else agent knowledge)
    max_new_tokens: int = 2048


class FoodAgent:
    def __init__(self, vlm: VLM, volume_expert: Optional[MFP3DVolumeEstimator] = None,
                 rag_expert: Optional[RAGExpert] = None, config: Optional[FoodAgentConfig] = None):
        self.cfg = config or FoodAgentConfig()
        self.agent = NutritionReasoningAgent(vlm, self.cfg.max_new_tokens)
        self.volume = volume_expert if self.cfg.use_volume_expert else None
        self.rag = rag_expert if self.cfg.use_rag_expert else None
        self.llm = LLMDirectExpert(vlm)
        self.generator = ResponseGenerator(vlm)

    @property
    def db(self) -> Optional[NutritionDB]:
        return self.rag.db if (self.rag and self.cfg.cat1_profile_from_db) else None

    def _reroute(self, e: FoodElement, failed: str, reason: str) -> None:
        nxt = self.cfg.volume_fallback if failed == "volume" else self.cfg.rag_fallback
        if nxt == "rag" and self.rag is None:
            nxt = "llm"
        if nxt == failed:  # never loop back into the expert that just failed
            nxt = "llm"
        e.route_notes.append(f"{failed} failed ({reason}) -> {nxt}")
        e.route = nxt

    def run(self, image: Image.Image, target_names: Optional[Sequence[str]] = None) -> DishEstimate:
        image = image.convert("RGB")
        trace: Dict[str, Any] = {}

        # 1) element analysis & decomposition
        elements, analysis_trace = self.agent.analyze(image, target_names)
        trace.update(analysis_trace)

        # 2) three-category routing
        self.agent.route(elements,
                         available={"volume": self.volume is not None, "rag": self.rag is not None, "llm": True},
                         fallback={"volume": self.cfg.volume_fallback, "rag": self.cfg.rag_fallback, "llm": "llm"})
        trace["routing"] = [{"name": e.name, "category": e.category, "route": e.route} for e in elements]

        estimates: Dict[int, ElementEstimate] = {}

        # 3a) category 1 -> MFP3D volume expert
        vol_idx = [i for i, e in enumerate(elements) if e.route == "volume"]
        if vol_idx:
            try:
                results = self.volume.estimate(image, [elements[i].bbox for i in vol_idx])
            except Exception as exc:  # model loading / CUDA errors must not kill the whole dish
                results = [None] * len(vol_idx)
                trace["volume_error"] = repr(exc)
            for i, res in zip(vol_idx, results):
                if res is None or (res.volume_cm3 is None and res.mass_g is None):
                    self._reroute(elements[i], "volume", (res.info.get("error") if res else "exception") or "no volume")
                    continue
                est = self.generator.from_volume(elements[i], res.volume_cm3, self.db, res.mass_g)
                est.details.update({"volume_method": res.method, "mask_area_px": res.mask_area_px, **res.info})
                estimates[i] = est

        # 3b) category 2 -> NutritionX retrieval
        rag_idx, refs = [], {}
        for i, e in enumerate(elements):
            if e.route != "rag" or i in estimates:
                continue
            try:
                found = self.rag.retrieve(e)
            except Exception as exc:
                found = []
                trace.setdefault("rag_errors", []).append(f"{e.name}: {exc!r}")
            if found:
                refs[len(rag_idx)] = found
                rag_idx.append(i)
            else:
                self._reroute(e, "rag", "no reference retrieved")
        if rag_idx:
            rag_est, trace["rag_raw"] = self.generator.estimate_rag_units(
                image, [elements[i] for i in rag_idx], refs, self.rag)
            for i, est in zip(rag_idx, rag_est):
                estimates[i] = est

        # 3c) category 3 (and fallbacks) -> direct LLM inference
        llm_idx = [i for i, e in enumerate(elements) if i not in estimates]
        for e in (elements[i] for i in llm_idx):
            e.route = "llm"
        llm_out, trace["llm_raw"] = self.llm.estimate(image, [elements[i] for i in llm_idx])
        for pos, i in enumerate(llm_idx):
            estimates[i] = ElementEstimate(elements[i], llm_out[pos], {"source": "llm_direct"})

        # 4) response generator: aggregate
        ordered = [estimates[i] for i in range(len(elements))]
        total = self.generator.aggregate(ordered)
        result = DishEstimate(total=total, elements=ordered, trace=trace)
        if target_names:
            targets = self.select_targets(ordered, target_names)
            result.trace["targets"] = {k: v.to_dict() for k, v in targets.items()}
            result.response = self.generator.render(
                [e for e in ordered if e.element.name in targets], self.generator.aggregate(list_values(targets)))
        else:
            result.response = self.generator.render(ordered, total)
        return result

    @staticmethod
    def select_targets(estimates: List[ElementEstimate], target_names: Sequence[str]) -> Dict[str, Nutrients]:
        names = [e.element.name for e in estimates]
        out = {}
        for t in target_names:
            idx = match_name(t, names)
            out[t] = estimates[idx].nutrients if idx is not None else Nutrients()
        return out


def list_values(d: Dict[str, Nutrients]) -> List[ElementEstimate]:
    return [ElementEstimate(FoodElement(name=k), v) for k, v in d.items()]


def _nutrients_from(d: Dict[str, Any]) -> Nutrients:
    return Nutrients(calories=clamp(to_float(d.get("calories")), 0, 10000),
                     mass=clamp(to_float(d.get("mass", d.get("mass_g"))), 0, 5000),
                     fat=clamp(to_float(d.get("fat")), 0, 1000),
                     carb=clamp(to_float(d.get("carb", d.get("carbohydrates"))), 0, 2000),
                     protein=clamp(to_float(d.get("protein")), 0, 1000))


class QwenDirectBaseline:
    """Holistic single-call baseline: Qwen2.5-VL directly predicts dish (or referred item) nutrition."""

    def __init__(self, vlm: VLM, max_new_tokens: int = 512):
        self.vlm = vlm
        self.max_new_tokens = max_new_tokens

    def run(self, image: Image.Image, target_names: Optional[Sequence[str]] = None) -> DishEstimate:
        model_img, _ = prepare_image(image)
        if not target_names:
            raw = self.vlm.chat(user_message(prompts.DIRECT_TOTAL, [model_img]), self.max_new_tokens)
            parsed = extract_json(raw)
            total = _nutrients_from(parsed if isinstance(parsed, dict) else {})
            return DishEstimate(total=total, response=raw, trace={"raw": raw})
        names = ", ".join(f'"{n}"' for n in target_names)
        raw = self.vlm.chat(user_message(prompts.DIRECT_REFER.format(names=names), [model_img]), self.max_new_tokens)
        parsed = extract_json(raw)
        items = parsed.get("items", []) if isinstance(parsed, dict) else (parsed if isinstance(parsed, list) else [])
        items = [it for it in items if isinstance(it, dict)]
        item_names = [str(it.get("name", "")) for it in items]
        targets = {}
        for pos, t in enumerate(target_names):
            idx = match_name(t, item_names)
            if idx is None and pos < len(items):
                idx = pos
            targets[t] = _nutrients_from(items[idx]) if idx is not None else Nutrients()
        total = Nutrients()
        for v in targets.values():
            total = total + v
        return DishEstimate(total=total, response=raw,
                            trace={"raw": raw, "targets": {k: v.to_dict() for k, v in targets.items()}})
