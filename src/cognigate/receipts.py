"""Receipt storage for standalone mode."""

from pathlib import Path
import json
from datetime import datetime
from typing import Optional

from .models import Receipt
from .observability import get_logger

logger = get_logger(__name__)


class ReceiptStore:
    """File-based receipt storage for standalone mode.
    
    Stores receipts as JSON files in the configured directory,
    enabling retrieval and listing of past job results.
    """
    
    def __init__(self, storage_dir: Path):
        """Initialize receipt store.
        
        Args:
            storage_dir: Directory path for storing receipt files
        """
        self.storage_dir = Path(storage_dir)
        self._ensure_directory()
    
    def _ensure_directory(self) -> None:
        """Create storage directory if it doesn't exist."""
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("receipt_store_initialized", storage_dir=str(self.storage_dir))
    
    def _receipt_path(self, lease_id: str) -> Path:
        """Get the file path for a receipt by lease ID."""
        # Sanitize lease_id to prevent path traversal
        safe_id = lease_id.replace("/", "_").replace("\\", "_").replace("..", "_")
        return self.storage_dir / f"{safe_id}.json"
    
    def save(self, receipt: Receipt) -> Path:
        """Save a receipt to storage.
        
        Args:
            receipt: Receipt to save
            
        Returns:
            Path to the saved receipt file
        """
        path = self._receipt_path(receipt.lease_id)
        
        # Convert receipt to JSON-serializable dict
        receipt_data = receipt.model_dump()
        
        # Handle datetime serialization
        for key, value in receipt_data.items():
            if isinstance(value, datetime):
                receipt_data[key] = value.isoformat()
        
        # Handle JobStatus enum
        if "status" in receipt_data:
            receipt_data["status"] = receipt_data["status"].value if hasattr(receipt_data["status"], "value") else str(receipt_data["status"])
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(receipt_data, f, indent=2, default=str)
        
        logger.info(
            "receipt_saved",
            lease_id=receipt.lease_id,
            task_id=receipt.task_id,
            status=str(receipt.status),
            path=str(path)
        )
        
        return path
    
    def get(self, lease_id: str) -> Optional[Receipt]:
        """Retrieve a receipt by lease ID.
        
        Args:
            lease_id: The lease ID to look up
            
        Returns:
            Receipt if found, None otherwise
        """
        path = self._receipt_path(lease_id)
        
        if not path.exists():
            logger.debug("receipt_not_found", lease_id=lease_id)
            return None
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Parse datetime back
            if "timestamp" in data and isinstance(data["timestamp"], str):
                data["timestamp"] = datetime.fromisoformat(data["timestamp"])
            
            return Receipt(**data)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                "receipt_parse_error",
                lease_id=lease_id,
                error=str(e)
            )
            return None
    
    def list(self, limit: int = 100) -> list[dict]:
        """List recent receipts.
        
        Args:
            limit: Maximum number of receipts to return
            
        Returns:
            List of receipt summaries (lease_id, task_id, status, timestamp)
        """
        receipts = []
        
        # Get all JSON files sorted by modification time (newest first)
        receipt_files = sorted(
            self.storage_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        
        for path in receipt_files[:limit]:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                receipts.append({
                    "lease_id": data.get("lease_id"),
                    "task_id": data.get("task_id"),
                    "status": data.get("status"),
                    "timestamp": data.get("timestamp"),
                    "worker_id": data.get("worker_id"),
                    "summary": data.get("summary", "")[:200]  # Truncate summary
                })
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(
                    "receipt_list_parse_error",
                    path=str(path),
                    error=str(e)
                )
                continue
        
        return receipts
    
    def delete(self, lease_id: str) -> bool:
        """Delete a receipt by lease ID.
        
        Args:
            lease_id: The lease ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        path = self._receipt_path(lease_id)
        
        if not path.exists():
            return False
        
        path.unlink()
        logger.info("receipt_deleted", lease_id=lease_id)
        return True
    
    def count(self) -> int:
        """Get the total number of stored receipts."""
        return len(list(self.storage_dir.glob("*.json")))
