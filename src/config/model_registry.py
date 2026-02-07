"""
SKIN_TELLIGENT - Model Version Registry

Track model versions using file hashes for reproducibility and rollback.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict
from logger import logger


class ModelRegistry:
    """Registry for tracking model versions and metadata."""
    
    def __init__(self, registry_path: str = "models/registry.json"):
        self.registry_path = Path(registry_path)
        self.registry: Dict = self._load_registry()
    
    def _load_registry(self) -> Dict:
        """Load existing registry or create empty one."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load registry: {e}")
        return {"models": {}, "active": {}}
    
    def _save_registry(self) -> None:
        """Save registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def _compute_hash(self, file_path: str) -> str:
        """Compute MD5 hash of a file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def register_model(
        self,
        model_type: str,  # "detector" or "classifier"
        model_path: str,
        description: str = ""
    ) -> str:
        """Register a model and return its version ID."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        file_hash = self._compute_hash(model_path)
        version_id = f"{model_type}_{file_hash[:8]}"
        
        if version_id not in self.registry["models"]:
            self.registry["models"][version_id] = {
                "type": model_type,
                "path": model_path,
                "hash": file_hash,
                "description": description,
                "registered_at": datetime.now().isoformat(),
                "file_size_bytes": os.path.getsize(model_path)
            }
            self._save_registry()
            logger.info(f"Registered model: {version_id}")
        
        # Set as active
        self.registry["active"][model_type] = version_id
        self._save_registry()
        
        return version_id
    
    def get_active_version(self, model_type: str) -> Optional[str]:
        """Get the active version ID for a model type."""
        return self.registry["active"].get(model_type)
    
    def get_model_info(self, version_id: str) -> Optional[Dict]:
        """Get metadata for a specific model version."""
        return self.registry["models"].get(version_id)
    
    def get_combined_version(self) -> str:
        """Get a combined version string for all active models."""
        detector = self.registry["active"].get("detector", "unknown")
        classifier = self.registry["active"].get("classifier", "unknown")
        return f"d:{detector[:8]}_c:{classifier[:8]}"
    
    def list_versions(self, model_type: Optional[str] = None) -> Dict:
        """List all registered model versions."""
        if model_type:
            return {
                k: v for k, v in self.registry["models"].items()
                if v["type"] == model_type
            }
        return self.registry["models"]


# Singleton instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get singleton model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry
