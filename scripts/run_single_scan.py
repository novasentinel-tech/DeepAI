"""
Example: Single domain scan.
Demonstrates complete pipeline usage.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.logging_config import get_logger
from src.security import domain_validator, rate_limiter, audit_log
from src.pipeline import ScanPipeline

logger = get_logger(__name__)


def scan_domain(domain: str, user_ip: str = "127.0.0.1") -> dict:
    """
    Scan a single domain end-to-end.
    
    Args:
        domain: Target domain to scan
        user_ip: User's IP address (for rate limiting)
        
    Returns:
        Complete analysis result
    """
    
    logger.info(f"Starting scan for {domain}")
    
    # STEP 1: Validate domain
    try:
        domain_validator.validate_strict(domain)
        logger.info(f"Domain {domain} passed validation")
    except Exception as e:
        logger.error(f"Domain validation failed: {e}")
        return {"status": "blocked", "error": str(e)}
    
    # STEP 2: Check rate limits
    try:
        rate_limiter.check_and_raise(user_ip, domain)
        logger.info(f"Rate limits OK for {user_ip}")
    except Exception as e:
        logger.error(f"Rate limit exceeded: {e}")
        return {"status": "rate_limited", "error": str(e)}
    
    # STEP 3: Log audit event
    audit_log.log_event(
        event_type="scan_initiated",
        details={
            "target": domain,
            "user_ip": user_ip,
            "validation": "passed"
        }
    )
    
    # STEP 4: Run analysis pipeline
    try:
        pipeline = ScanPipeline()
        result = pipeline.analyze(domain)
        
        # STEP 5: Log result
        audit_log.log_event(
            event_type="scan_completed",
            details={
                "target": domain,
                "classification": result.get("classification"),
                "confidence": result.get("confidence")
            }
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        
        audit_log.log_event(
            event_type="scan_failed",
            details={
                "target": domain,
                "error": str(e)
            }
        )
        
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    import json
    
    # Example usage
    domain = "example.com"
    result = scan_domain(domain)
    
    print("\n" + "="*60)
    print(f"SCAN RESULT FOR: {domain}")
    print("="*60)
    print(json.dumps(result, indent=2, default=str))
    print("="*60 + "\n")
