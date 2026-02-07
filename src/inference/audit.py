"""
SKIN_TELLIGENT - Structured Audit Logging

JSON-structured audit logs for compliance, analysis, and debugging.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List
from src.inference.contracts import AuditRecord, InferenceResult


class AuditLogger:
    """Structured audit logger for inference decisions."""
    
    def __init__(self, log_dir: str = "output/audit"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create daily log file
        self.log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        # Setup Python logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            fh = logging.FileHandler(self.log_file)
            fh.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(fh)
    
    def log_inference(
        self,
        session_id: str,
        result: InferenceResult,
        model_version: str = "unknown"
    ) -> None:
        """Log an inference event."""
        record = AuditRecord(
            session_id=session_id,
            event_type="INFERENCE",
            model_version=model_version,
            num_rois=len(result.classifications),
            classifications=[
                {
                    "roi": c.roi_index,
                    "class": c.class_name,
                    "confidence": round(c.confidence, 4),
                    "state": c.inference_state.value
                }
                for c in result.classifications
            ],
            overall_state=result.overall_state.value,
            max_confidence=round(result.max_confidence, 4)
        )
        
        self._write(record)
    
    def log_chat(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        inference_state: str
    ) -> None:
        """Log a chat interaction."""
        record = AuditRecord(
            session_id=session_id,
            event_type="CHAT",
            user_query=user_query,
            assistant_response=assistant_response[:500],  # Truncate for storage
            chat_inference_state=inference_state
        )
        
        self._write(record)
    
    def log_error(
        self,
        session_id: str,
        error_type: str,
        error_message: str
    ) -> None:
        """Log an error event."""
        record = AuditRecord(
            session_id=session_id,
            event_type="ERROR",
            error_type=error_type,
            error_message=error_message[:500]
        )
        
        self._write(record)
    
    def _write(self, record: AuditRecord) -> None:
        """Write record to log file as JSON line."""
        try:
            log_dict = record.to_log_dict()
            log_dict["timestamp"] = log_dict["timestamp"].isoformat()
            self.logger.info(json.dumps(log_dict))
        except Exception as e:
            # Fallback to console if file logging fails
            print(f"Audit log error: {e}")


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get singleton audit logger instance."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
