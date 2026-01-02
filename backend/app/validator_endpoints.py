"""
Validator Endpoints for R3MES Dashboard

Provides REST API endpoints for validator information including trust scores.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from .blockchain_query_client import get_blockchain_client
from .cache_middleware import cache_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validators", tags=["validators"])


class TrustScoreResponse(BaseModel):
    """Trust score data for a validator."""
    trust_score: float
    total_verifications: int
    successful_verifications: int
    false_verdicts: int
    lazy_validation_count: int


class ValidatorTrustScoresResponse(BaseModel):
    """Response containing trust scores for all validators."""
    trust_scores: Dict[str, TrustScoreResponse]
    total: int


@router.get("/trust-scores", response_model=ValidatorTrustScoresResponse)
@cache_response(ttl=60, key_prefix="validator_trust_scores")
async def get_validator_trust_scores(
    limit: int = Query(default=100, ge=1, le=500, description="Maximum validators to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
) -> ValidatorTrustScoresResponse:
    """
    Get trust scores for all validators.
    
    Trust score is calculated based on:
    - Verification success rate
    - False verdict penalties (-50% per false verdict ratio)
    - Lazy validation penalties (-10% per lazy validation ratio)
    
    Returns a dictionary mapping validator operator addresses to their trust score data.
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all validators
        validators_data = blockchain_client.get_all_validators(limit=limit, offset=offset)
        validators = validators_data.get("validators", [])
        total = validators_data.get("total", 0)
        
        trust_scores: Dict[str, TrustScoreResponse] = {}
        
        for validator in validators:
            operator_address = validator.get("operator_address", "")
            if not operator_address:
                continue
            
            # Get detailed validator info with trust score
            validator_info = blockchain_client.get_validator_info(operator_address)
            
            if validator_info:
                trust_scores[operator_address] = TrustScoreResponse(
                    trust_score=validator_info.get("trust_score", 0.5) * 100,  # Convert to percentage
                    total_verifications=validator_info.get("total_verifications", 0),
                    successful_verifications=validator_info.get("successful_verifications", 0),
                    false_verdicts=validator_info.get("false_verdicts", 0),
                    lazy_validation_count=validator_info.get("lazy_validation_count", 0),
                )
            else:
                # Default trust score for validators without verification records
                trust_scores[operator_address] = TrustScoreResponse(
                    trust_score=50.0,  # Default 50%
                    total_verifications=0,
                    successful_verifications=0,
                    false_verdicts=0,
                    lazy_validation_count=0,
                )
        
        return ValidatorTrustScoresResponse(
            trust_scores=trust_scores,
            total=total,
        )
    
    except Exception as e:
        logger.error(f"Failed to get validator trust scores: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch validator trust scores: {str(e)}"
        )


@router.get("/{validator_address}/trust-score", response_model=TrustScoreResponse)
@cache_response(ttl=30, key_prefix="validator_trust_score")
async def get_single_validator_trust_score(
    validator_address: str,
) -> TrustScoreResponse:
    """
    Get trust score for a specific validator.
    
    Args:
        validator_address: Validator operator address (e.g., remesvaloper1...)
    
    Returns:
        Trust score data for the validator
    """
    try:
        blockchain_client = get_blockchain_client()
        
        validator_info = blockchain_client.get_validator_info(validator_address)
        
        if not validator_info:
            raise HTTPException(
                status_code=404,
                detail=f"Validator not found: {validator_address}"
            )
        
        return TrustScoreResponse(
            trust_score=validator_info.get("trust_score", 0.5) * 100,  # Convert to percentage
            total_verifications=validator_info.get("total_verifications", 0),
            successful_verifications=validator_info.get("successful_verifications", 0),
            false_verdicts=validator_info.get("false_verdicts", 0),
            lazy_validation_count=validator_info.get("lazy_validation_count", 0),
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trust score for validator {validator_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch validator trust score: {str(e)}"
        )


@router.get("/{validator_address}/verification-history")
@cache_response(ttl=60, key_prefix="validator_verification_history")
async def get_validator_verification_history(
    validator_address: str,
    limit: int = Query(default=50, ge=1, le=200, description="Maximum records to return"),
) -> Dict[str, Any]:
    """
    Get verification history for a specific validator.
    
    Returns recent verification records including:
    - Gradient hash verified
    - Verification result (success/failure)
    - Timestamp
    - Block height
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Query verification history from R3MES keeper
        # Note: This endpoint may need to be implemented in the blockchain module
        endpoint = f"/remes/remes/v1/validator_verification_history/{validator_address}"
        
        try:
            data = blockchain_client._query_rest(endpoint, {"pagination.limit": limit})
            
            history = []
            for record in data.get("verification_history", []):
                history.append({
                    "gradient_hash": record.get("gradient_hash", ""),
                    "result": record.get("result", "unknown"),
                    "timestamp": record.get("timestamp", ""),
                    "block_height": record.get("block_height", 0),
                    "miner_address": record.get("miner_address", ""),
                })
            
            return {
                "validator_address": validator_address,
                "history": history,
                "total": len(history),
            }
        except Exception:
            # If endpoint doesn't exist yet, return empty history
            return {
                "validator_address": validator_address,
                "history": [],
                "total": 0,
                "message": "Verification history endpoint not yet available",
            }
    
    except Exception as e:
        logger.error(f"Failed to get verification history for validator {validator_address}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch verification history: {str(e)}"
        )
