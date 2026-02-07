"""
SKIN_TELLIGENT - Structured Inference Contracts

Pydantic schemas for type-safe inference pipeline with decision governance.
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from datetime import datetime


class InferenceState(str, Enum):
    """Confidence-based inference states for decision governance."""
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"      # >= 80%
    UNCERTAIN = "UNCERTAIN"                   # 60-80%
    ABSTAIN = "ABSTAIN"                       # < 60%


class BoundingBox(BaseModel):
    """Bounding box for detected region."""
    x: int
    y: int
    width: int
    height: int


class ClassificationResult(BaseModel):
    """Result for a single ROI classification."""
    roi_index: int
    class_name: str
    class_idx: int
    confidence: float = Field(ge=0.0, le=1.0)
    inference_state: InferenceState
    gradcam_path: Optional[str] = None
    
    @classmethod
    def from_raw(cls, raw_result: dict, roi_index: int) -> "ClassificationResult":
        """Create from raw classifier output."""
        confidence = raw_result.get("confidence", 0.0)
        
        if confidence >= 0.80:
            state = InferenceState.HIGH_CONFIDENCE
        elif confidence >= 0.60:
            state = InferenceState.UNCERTAIN
        else:
            state = InferenceState.ABSTAIN
        
        return cls(
            roi_index=roi_index,
            class_name=raw_result.get("class_name", "Unknown"),
            class_idx=raw_result.get("class_idx", -1),
            confidence=confidence,
            inference_state=state,
            gradcam_path=raw_result.get("gradcam")
        )


class DetectionResult(BaseModel):
    """Result from detection stage."""
    num_detections: int
    boxes: List[BoundingBox]
    confidences: List[float]
    classes: List[int]


class InferenceResult(BaseModel):
    """Complete inference result with decision governance metadata."""
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str
    model_version: str
    
    # Detection
    detection: Optional[DetectionResult] = None
    
    # Classification (per ROI)
    classifications: List[ClassificationResult] = []
    
    # Aggregated inference state (based on max confidence)
    overall_state: InferenceState = InferenceState.ABSTAIN
    max_confidence: float = 0.0
    
    # Metadata
    processing_time_ms: Optional[float] = None
    error: Optional[str] = None
    
    def compute_overall_state(self) -> None:
        """Compute overall state from classification results."""
        if not self.classifications:
            self.overall_state = InferenceState.ABSTAIN
            self.max_confidence = 0.0
            return
        
        self.max_confidence = max(c.confidence for c in self.classifications)
        
        if self.max_confidence >= 0.80:
            self.overall_state = InferenceState.HIGH_CONFIDENCE
        elif self.max_confidence >= 0.60:
            self.overall_state = InferenceState.UNCERTAIN
        else:
            self.overall_state = InferenceState.ABSTAIN


class AuditRecord(BaseModel):
    """Structured audit log entry for compliance and analysis."""
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: str
    event_type: str  # INFERENCE, CHAT, ERROR
    
    # Inference details
    model_version: Optional[str] = None
    num_rois: Optional[int] = None
    classifications: Optional[List[dict]] = None
    overall_state: Optional[str] = None
    max_confidence: Optional[float] = None
    
    # Chat details 
    user_query: Optional[str] = None
    assistant_response: Optional[str] = None
    chat_inference_state: Optional[str] = None
    
    # Error details
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    def to_log_dict(self) -> dict:
        """Convert to dictionary for JSON logging."""
        return self.model_dump(exclude_none=True)
