"""Lightweight GLiNER-based entity extraction for EMS ASR transcripts.

Notes:
- Domain limitation: this model is primarily trained on written biomedical text,
  so performance may degrade on noisy, abbreviated EMS ASR transcripts.
- Operational behavior: first model use downloads Hugging Face artifacts and can
  add noticeable latency before steady-state inference.
- Future path: replace/augment with EMS-specific fine-tuning and calibration
  once labeled dispatch + PCR corpora are available.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, overload

from ems_pipeline.models import Entity, Segment, Transcript
from ems_pipeline.nlp.negation import NegationDetector

NLP_ENTITY_LABELS = [
    "Symptom",
    "Disease",
    "Medical condition",
    "Clinical finding",
    "Drug",
    "Anatomical location",
]

LABEL_TYPE_MAP = {
    "Symptom": "SYMPTOM",
    "Disease": "CONDITION",
    "Medical condition": "CONDITION",
    "Clinical finding": "SYMPTOM",
    "Drug": "MEDICATION",
    "Anatomical location": "ANATOMY",
}

_SENTENCE_SPLIT_RE = re.compile(r"[^.?!]+[.?!]?")


@dataclass(frozen=True, slots=True)
class _SentenceSpan:
    start: int
    end: int
    text: str


class NlpExtractor:
    def __init__(
        self,
        model_name: str = "Ihor/gliner-biomed-base-v1.0",
        confidence_threshold: float = 0.5,
        labels: list[str] = NLP_ENTITY_LABELS,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.labels = list(labels)
        self._model: Any | None = None
        self.negation = NegationDetector()
        self._negation_drops: list[dict[str, Any]] = []

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise ImportError(
                "GLiNER is required for NlpExtractor. Install the 'gliner' package."
            ) from exc

        model = GLiNER.from_pretrained(self.model_name)
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            try:
                maybe_model = to_fn("cpu")
                if maybe_model is not None:
                    model = maybe_model
            except Exception:
                # Keep the loaded model when explicit CPU placement is unavailable.
                pass

        self._model = model
        return model

    @overload
    def extract(
        self,
        transcript: Transcript,
        *,
        index_to_segment_id: dict[int, str],
    ) -> list[Entity]: ...

    @overload
    def extract(self, text: str, segment: Segment) -> list[Entity]: ...

    def extract(
        self,
        transcript_or_text: Transcript | str,
        segment: Segment | None = None,
        *,
        index_to_segment_id: dict[int, str] | None = None,
    ) -> list[Entity]:
        self._negation_drops = []

        if isinstance(transcript_or_text, Transcript):
            if index_to_segment_id is None:
                raise TypeError(
                    "extract(transcript, ...) requires index_to_segment_id "
                    "when transcript input is used."
                )

            entities: list[Entity] = []
            for idx, item in enumerate(transcript_or_text.segments):
                seg_id = index_to_segment_id.get(idx, f"seg_{idx:04d}")
                entities.extend(
                    self._extract_segment(
                        text=item.text,
                        segment=item,
                        segment_id=seg_id,
                    )
                )
            return entities

        if isinstance(transcript_or_text, str):
            if segment is None:
                raise TypeError("extract(text, segment) requires a Segment instance.")
            seg_id = "seg_0000"
            return self._extract_segment(
                text=transcript_or_text,
                segment=segment,
                segment_id=seg_id,
            )

        raise TypeError(
            "extract expects either (Transcript, index_to_segment_id=...) "
            "or (text, segment)."
        )

    def pop_negation_drops(self) -> list[dict[str, Any]]:
        drops = self._negation_drops
        self._negation_drops = []
        return drops

    def _extract_segment(self, *, text: str, segment: Segment, segment_id: str) -> list[Entity]:
        if not text.strip():
            return []

        model = self._get_model()
        raw_predictions = model.predict_entities(
            text,
            self.labels,
            threshold=self.confidence_threshold,
        )
        sentences = self._split_sentences(text)

        entities: list[Entity] = []
        for prediction in raw_predictions:
            label = self._prediction_get(prediction, "label")
            if not isinstance(label, str):
                continue

            mapped_type = LABEL_TYPE_MAP.get(label)
            if mapped_type is None:
                continue

            start = self._coerce_int(self._prediction_get(prediction, "start"))
            end = self._coerce_int(self._prediction_get(prediction, "end"))
            if start is None or end is None or start < 0 or end <= start or end > len(text):
                continue

            confidence = self._coerce_float(
                self._prediction_get(prediction, "score")
                or self._prediction_get(prediction, "confidence")
            )
            surface = self._prediction_get(prediction, "text")
            if not isinstance(surface, str) or not surface.strip():
                surface = text[start:end]

            sentence = self._sentence_for_span(sentences, start, end, fallback_text=text)
            sentence_start = sentence.start
            local_span = (start - sentence_start, end - sentence_start)
            negation = self.negation.check_sentence(sentence.text, [local_span])[0]
            if negation.is_negated:
                self._negation_drops.append(
                    {
                        "type": mapped_type,
                        "text": surface,
                        "source": "nlp",
                        "segment_id": segment_id,
                        "confidence": confidence,
                        "trigger_term": negation.trigger_term,
                        "trigger_type": negation.trigger_type,
                    }
                )
                continue

            attributes: dict[str, Any] = {
                "source": "nlp",
                "segment_id": segment_id,
                "char_start": start,
                "char_end": end,
            }
            if negation.is_family_history:
                attributes["experiencer"] = "family"

            entities.append(
                Entity(
                    type=mapped_type,
                    text=surface,
                    normalized=surface,
                    start=segment.start,
                    end=segment.end,
                    speaker=segment.speaker,
                    confidence=confidence,
                    attributes=attributes,
                )
            )

        return entities

    def _split_sentences(self, text: str) -> list[_SentenceSpan]:
        # ASR text often has weak punctuation; this uses a simple .?! splitter.
        spans: list[_SentenceSpan] = []
        for match in _SENTENCE_SPLIT_RE.finditer(text):
            raw = match.group(0)
            sentence = raw.strip()
            if not sentence:
                continue
            leading_ws = len(raw) - len(raw.lstrip())
            trailing_ws = len(raw) - len(raw.rstrip())
            start = match.start() + leading_ws
            end = match.end() - trailing_ws
            spans.append(_SentenceSpan(start=start, end=end, text=text[start:end]))

        if spans:
            return spans
        return [_SentenceSpan(start=0, end=len(text), text=text)]

    @staticmethod
    def _sentence_for_span(
        spans: list[_SentenceSpan], start: int, end: int, *, fallback_text: str
    ) -> _SentenceSpan:
        for span in spans:
            if start >= span.start and end <= span.end:
                return span
        for span in spans:
            if start < span.end and end > span.start:
                return span
        return _SentenceSpan(start=0, end=len(fallback_text), text=fallback_text)

    @staticmethod
    def _prediction_get(prediction: Any, key: str) -> Any:
        if isinstance(prediction, Mapping):
            return prediction.get(key)
        return getattr(prediction, key, None)

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return None

    @staticmethod
    def _coerce_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return None


__all__ = [
    "LABEL_TYPE_MAP",
    "NLP_ENTITY_LABELS",
    "NlpExtractor",
]
