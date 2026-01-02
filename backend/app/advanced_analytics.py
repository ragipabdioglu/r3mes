"""
Advanced Analytics for R3MES

Provides detailed analytics and insights:
- Network growth trends
- Mining efficiency metrics
- Economic analysis
- Performance benchmarks
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from .database_async import AsyncDatabase
from .config_manager import get_config_manager
from .blockchain_query_client import get_blockchain_client
from .metrics import api_request_duration_seconds, cache_hits_total, cache_misses_total
from .constants import (
    ANALYTICS_TIMELINE_MAX_POINTS,
    ANALYTICS_HASHRATE_MULTIPLIER,
    ANALYTICS_DEFAULT_GRANULARITY_DAYS
)
from prometheus_client import REGISTRY
import statistics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

config_manager = get_config_manager()
config = config_manager.load()
database = AsyncDatabase(db_path=config.database_path, chain_json_path=config.chain_json_path)


async def _build_timeline_data(days: int, granularity: str, blockchain_client) -> List[Dict]:
    """
    Build timeline data for network growth metrics.
    
    Uses indexed historical data from indexer if available, otherwise falls back to current data.
    
    Args:
        days: Number of days to analyze
        granularity: Time granularity (day, week, month)
        blockchain_client: Blockchain query client
        
    Returns:
        List of timeline data points
    """
    try:
        # Try to use indexed historical data first
        if database.config.is_postgresql():
            try:
                from .indexer import get_indexer
                indexer = get_indexer(database)
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Get network snapshots from indexer
                snapshots = await indexer.get_network_snapshots(start_date=start_date, end_date=end_date, limit=1000)
                
                if snapshots:
                    # Convert snapshots to timeline format
                    timeline = []
                    for snapshot in snapshots:
                        snapshot_date = snapshot.get('snapshot_date')
                        if isinstance(snapshot_date, str):
                            from datetime import datetime as dt
                            snapshot_date = dt.fromisoformat(snapshot_date).date()
                        
                        timeline.append({
                            "date": snapshot_date.strftime("%Y-%m-%d") if snapshot_date else "",
                            "miners": snapshot.get('total_miners', 0),
                            "validators": snapshot.get('total_validators', 0),
                            "total_stake": float(snapshot.get('total_stake', 0.0)),
                            "hashrate": float(snapshot.get('network_hashrate', 0.0)),
                        })
                    
                    # Sort by date
                    timeline.sort(key=lambda x: x.get("date", ""))
                    
                    # Limit to requested number of points
                    if len(timeline) > ANALYTICS_TIMELINE_MAX_POINTS:
                        step = len(timeline) // ANALYTICS_TIMELINE_MAX_POINTS
                        timeline = timeline[::step]
                    
                    logger.debug(f"Built timeline from {len(snapshots)} indexed snapshots")
                    return timeline
            except Exception as e:
                logger.error(f"Could not use indexed data, falling back to current data: {e}")
                # Continue with fallback
        
        # Fallback: use current data (old behavior)
        timeline = []
        end_date = datetime.now()
        
        # Determine time intervals based on granularity
        interval_days = ANALYTICS_DEFAULT_GRANULARITY_DAYS.get(granularity, 30)
        
        # Generate timeline points
        current_date = end_date - timedelta(days=days)
        while current_date <= end_date:
            # Try to get network stats from blockchain (will be current data)
            try:
                miners_result = blockchain_client.get_all_miners(limit=1000, offset=0)
                miners_count = miners_result.get("total", 0)
                
                validators_result = blockchain_client.get_all_validators(limit=1000, offset=0)
                validators_count = validators_result.get("total", 0)
                
                staking_info = blockchain_client.get_staking_info()
                total_stake = staking_info.get("total_stake", 0.0) if staking_info else 0.0
                
            except Exception as e:
                logger.error(f"Could not fetch blockchain data for timeline: {e}")
                miners_count = 0
                validators_count = 0
                total_stake = 0.0
            
            timeline.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "miners": miners_count,
                "validators": validators_count,
                "total_stake": total_stake,
                "hashrate": miners_count * ANALYTICS_HASHRATE_MULTIPLIER,  # Estimate
            })
            
            current_date += timedelta(days=interval_days)
        
        # Limit to requested number of points (avoid too many data points)
        if len(timeline) > ANALYTICS_TIMELINE_MAX_POINTS:
            # Sample evenly
            step = len(timeline) // ANALYTICS_TIMELINE_MAX_POINTS
            timeline = timeline[::step]
        
        return timeline
    except Exception as e:
        logger.warning(f"Failed to build timeline data: {e}")
        return []


def _calculate_avg_gradient_time(blockchain_client, blockchain_stats: Optional[Dict]) -> float:
    """
    Calculate average gradient computation time from blockchain statistics.
    
    Args:
        blockchain_client: Blockchain query client
        blockchain_stats: Network statistics from blockchain
        
    Returns:
        Average gradient time in seconds
    """
    try:
        # Estimate from blockchain statistics
        # If we have gradient count and time period, we can estimate average time
        # For now, use a reasonable default based on network activity
        if blockchain_stats:
            total_gradients = blockchain_stats.get("total_gradients", 0)
            # Estimate: if many gradients, miners are active, so average time is lower
            if total_gradients > 1000:
                return 25.0  # Fast network
            elif total_gradients > 100:
                return 30.0  # Normal network
            else:
                return 45.0  # Slow/new network
        return 30.0  # Default estimate
    except Exception as e:
        logger.debug(f"Could not calculate avg gradient time: {e}")
        return 30.0  # Default estimate


def _calculate_avg_verification_time(blockchain_client) -> float:
    """
    Calculate average verification time from validator records.
    
    Args:
        blockchain_client: Blockchain query client
        
    Returns:
        Average verification time in seconds
    """
    try:
        # Get validators to estimate verification time
        validators_result = blockchain_client.get_all_validators(limit=100, offset=0)
        validators = validators_result.get("validators", [])
        
        if validators:
            # Estimate based on validator count and activity
            # More validators = faster verification (parallel)
            # Fewer validators = slower verification
            validator_count = len(validators)
            if validator_count > 20:
                return 3.0  # Fast verification
            elif validator_count > 10:
                return 5.0  # Normal verification
            else:
                return 8.0  # Slower verification
        return 5.0  # Default estimate
    except Exception as e:
        logger.debug(f"Could not calculate avg verification time: {e}")
        return 5.0  # Default estimate


def _calculate_percentile_from_histogram(metric_family, percentile: float) -> float:
    """
    Calculate percentile from Prometheus histogram buckets.
    
    Args:
        metric_family: Prometheus MetricFamily object
        percentile: Percentile to calculate (0.0-1.0, e.g., 0.95 for p95)
        
    Returns:
        Percentile value in seconds
    """
    try:
        # Collect bucket samples
        buckets = []  # List of (upper_bound, count) tuples
        total_count = 0
        
        # Iterate through all samples in the metric family
        for sample in metric_family.samples:
            if sample.name.endswith('_bucket'):
                # Extract bucket upper bound from label 'le'
                le_label = sample.labels.get('le', '+Inf')
                if le_label == '+Inf':
                    total_count = sample.value
                else:
                    try:
                        upper_bound = float(le_label)
                        buckets.append((upper_bound, sample.value))
                    except ValueError:
                        continue
            elif sample.name.endswith('_count'):
                total_count = max(total_count, sample.value)
        
        if not buckets or total_count == 0:
            return 0.0
        
        # Sort buckets by upper bound
        buckets.sort(key=lambda x: x[0])
        
        # Calculate percentile
        target_count = total_count * percentile
        
        # Find the bucket containing the percentile
        cumulative_count = 0
        prev_upper = 0.0
        prev_count = 0
        
        for upper_bound, count in buckets:
            cumulative_count += count
            if cumulative_count >= target_count:
                # Linear interpolation within the bucket
                if prev_count > 0:
                    # Interpolate between previous bucket and current
                    ratio = (target_count - prev_count) / (cumulative_count - prev_count) if (cumulative_count - prev_count) > 0 else 0.0
                    return prev_upper + ratio * (upper_bound - prev_upper)
                else:
                    return upper_bound
            prev_upper = upper_bound
            prev_count = cumulative_count
        
        # If percentile is beyond all buckets, return the maximum
        return buckets[-1][0] if buckets else 0.0
        
    except Exception as e:
        logger.warning(f"Failed to calculate percentile from histogram: {e}")
        return 0.0


@router.get("/network-growth")
async def get_network_growth(
    days: int = Query(30, ge=1, le=365),
    granularity: str = Query("day", regex="^(day|week|month)$")
):
    """
    Get network growth metrics over time.
    
    Args:
        days: Number of days to analyze
        granularity: Time granularity (day, week, month)
        
    Returns:
        Network growth data with trends
    """
    try:
        blockchain_client = get_blockchain_client()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Query current network stats from blockchain
        miners_result = blockchain_client.get_all_miners(limit=1000, offset=0)
        current_miners = miners_result.get("total", 0)
        
        # Get network stats from database
        network_stats = await database.get_network_stats()
        
        # Calculate growth (simplified: compare with previous period)
        # In production, this would query historical data
        previous_miners = max(0, current_miners - 10)  # Estimate previous count
        miner_growth = current_miners - previous_miners
        miner_growth_rate = (miner_growth / previous_miners * 100) if previous_miners > 0 else 0.0
        
        # Network hashrate estimation (based on active miners)
        # In production, this would be calculated from actual mining data
        active_miners = network_stats.get("active_miners", 0)
        estimated_hashrate = active_miners * 10.0  # Estimate: 10 H/s per active miner
        
        # Get timeline data for growth calculations
        timeline = await _build_timeline_data(days, granularity, blockchain_client)
        
        # Calculate growth from timeline data (first vs last data point)
        def calculate_growth_from_timeline(metric_key: str, current_value: float) -> Tuple[float, float]:
            """Calculate growth and growth rate from timeline data."""
            if not timeline or len(timeline) < 2:
                return 0.0, 0.0
            
            first_value = timeline[0].get(metric_key, 0.0)
            last_value = timeline[-1].get(metric_key, current_value)  # Use current as last if timeline data is stale
            
            # If first_value is 0, we can't calculate growth rate
            if first_value == 0:
                growth = last_value - first_value
                growth_rate = 0.0 if growth == 0 else float('inf')  # Infinite growth from zero
            else:
                growth = last_value - first_value
                growth_rate = (growth / first_value) * 100.0
            
            return growth, growth_rate
        
        # Get current values
        current_validators = blockchain_client.get_all_validators(limit=1, offset=0).get("total", 0) if blockchain_client else 0
        current_stake = network_stats.get("total_credits", 0.0)
        
        # Calculate growth from timeline
        validator_growth, validator_growth_rate = calculate_growth_from_timeline("validators", float(current_validators))
        stake_growth, stake_growth_rate = calculate_growth_from_timeline("total_stake", current_stake)
        hashrate_growth, hashrate_growth_rate = calculate_growth_from_timeline("hashrate", estimated_hashrate)
        
        growth_data = {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days,
                "granularity": granularity,
            },
            "metrics": {
                "total_miners": {
                    "current": current_miners,
                    "growth": miner_growth,
                    "growth_rate": miner_growth_rate,
                },
                "total_validators": {
                    "current": current_validators,
                    "growth": int(validator_growth),
                    "growth_rate": validator_growth_rate,
                },
                "total_stake": {
                    "current": current_stake,
                    "growth": stake_growth,
                    "growth_rate": stake_growth_rate,
                },
                "network_hashrate": {
                    "current": estimated_hashrate,
                    "growth": hashrate_growth,
                    "growth_rate": hashrate_growth_rate,
                },
            },
            "timeline": timeline,  # Time-series data aggregation
        }
        
        return growth_data
    except Exception as e:
        logger.error(f"Error getting network growth: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch network growth data")


@router.get("/mining-efficiency")
async def get_mining_efficiency(
    wallet_address: Optional[str] = None,
    days: int = Query(7, ge=1, le=90)
):
    """
    Get mining efficiency metrics.
    
    Args:
        wallet_address: Specific miner address (None = network average)
        days: Number of days to analyze
        
    Returns:
        Mining efficiency metrics
    """
    try:
        if wallet_address:
            # Get miner-specific efficiency
            miner_stats = await database.get_miner_stats(wallet_address)
            blockchain_client = get_blockchain_client()
            miner_score = blockchain_client.get_miner_score(wallet_address)
            
            hashrate = miner_stats.get("hashrate", 0.0)
            total_earnings = miner_stats.get("total_earnings", 0.0)
            uptime = miner_stats.get("uptime_percentage", 0.0)
            
            # Calculate GPU utilization from successful submissions
            # Estimate: higher successful submissions = higher GPU utilization
            successful_submissions = miner_score.get("successful_submissions", 0) if miner_score else 0
            total_submissions = miner_score.get("total_submissions", 0) if miner_score else 0
            gpu_utilization = min(100.0, (successful_submissions / max(total_submissions, 1)) * 100) if total_submissions > 0 else 0.0
            
            # Power efficiency: gradients per estimated watt
            # Estimate based on submissions and uptime
            power_efficiency = (successful_submissions / max(uptime, 1.0)) * 0.01 if uptime > 0 else 0.0
            
            efficiency = {
                "wallet_address": wallet_address,
                "hashrate": hashrate,
                "gpu_utilization": gpu_utilization,
                "power_efficiency": power_efficiency,
                "uptime_percentage": uptime,
                "earnings_per_hashrate": total_earnings / max(hashrate, 1.0) if hashrate > 0 else 0.0,
            }
        else:
            # Network average
            blockchain_client = get_blockchain_client()
            network_stats = await database.get_network_stats()
            
            # Get all miners to calculate averages
            miners_result = blockchain_client.get_all_miners(limit=1000, offset=0)
            miners = miners_result.get("miners", [])
            
            if miners:
                total_hashrate = sum(m.get("reputation", 0.0) / 10.0 for m in miners)  # Estimate hashrate from reputation
                total_successful = sum(m.get("successful_submissions", 0) for m in miners)
                total_submissions = sum(m.get("total_submissions", 0) for m in miners)
                total_uptime = sum(m.get("reputation", 0.0) / 10.0 for m in miners)  # Estimate uptime
                
                avg_hashrate = total_hashrate / len(miners)
                avg_gpu_utilization = (total_successful / max(total_submissions, 1)) * 100 if total_submissions > 0 else 0.0
                avg_power_efficiency = (total_successful / max(total_uptime, 1.0)) * 0.01 if total_uptime > 0 else 0.0
                avg_uptime = min(100.0, (total_uptime / len(miners)) / 10.0) if miners else 0.0
            else:
                avg_hashrate = 0.0
                avg_gpu_utilization = 0.0
                avg_power_efficiency = 0.0
                avg_uptime = 0.0
            
            efficiency = {
                "network_average": True,
                "avg_hashrate": avg_hashrate,
                "avg_gpu_utilization": avg_gpu_utilization,
                "avg_power_efficiency": avg_power_efficiency,
                "avg_uptime": avg_uptime,
                "total_miners": network_stats.get("active_miners", 0),
            }
        
        return efficiency
    except Exception as e:
        logger.error(f"Error getting mining efficiency: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch mining efficiency")


@router.get("/economic-analysis")
async def get_economic_analysis(
    days: int = Query(30, ge=1, le=365)
):
    """
    Get economic analysis of the network.
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Economic metrics and trends
    """
    try:
        blockchain_client = get_blockchain_client()
        network_stats = await database.get_network_stats()
        blockchain_stats = blockchain_client.get_network_statistics()
        
        # Get tokenomics from blockchain
        supply_info = blockchain_client.get_token_supply()
        staking_info = blockchain_client.get_staking_info()
        reward_params = blockchain_client.get_reward_params()
        
        # Get total credits (stake) from database as fallback
        total_stake = network_stats.get("total_credits", 0.0)
        
        # Use blockchain data if available, otherwise use estimates
        if supply_info:
            total_supply = supply_info.get("total_supply", 0.0)
            circulating_supply = supply_info.get("circulating_supply", total_supply)
        else:
            # Fallback: estimate from stake
            total_supply = total_stake * 2.0 if total_stake > 0 else 0.0
            circulating_supply = total_supply
        
        # Get staking ratio from blockchain if available
        if staking_info:
            staking_ratio = staking_info.get("staking_ratio", 0.0)
            total_stake = staking_info.get("total_stake", total_stake)
        else:
            # Fallback: estimate from stake
            staking_ratio = (total_stake / circulating_supply * 100) if circulating_supply > 0 else 0.0
        
        # Calculate rewards from blockchain statistics
        total_gradients = blockchain_stats.get("total_gradients", 0) if blockchain_stats else 0
        total_aggregations = blockchain_stats.get("total_aggregations", 0) if blockchain_stats else 0
        
        # Get reward parameters from blockchain if available
        if reward_params:
            base_reward_per_gradient = reward_params.get("base_reward_per_gradient", 10.0)
            miner_reward_ratio = reward_params.get("miner_reward_ratio", 0.7)
            validator_reward_ratio = reward_params.get("validator_reward_ratio", 0.2)
            treasury_ratio = reward_params.get("treasury_ratio", 0.1)
        else:
            # Fallback: use default values
            base_reward_per_gradient = 10.0
            miner_reward_ratio = 0.7
            validator_reward_ratio = 0.2
            treasury_ratio = 0.1
        
        total_distributed = total_gradients * base_reward_per_gradient
        miner_rewards = total_distributed * miner_reward_ratio
        validator_rewards = total_distributed * validator_reward_ratio
        treasury = total_distributed * treasury_ratio
        
        # Calculate average earnings
        active_miners = blockchain_stats.get("active_miners", 0) if blockchain_stats else network_stats.get("active_miners", 0)
        avg_miner_earnings = (miner_rewards / max(active_miners, 1)) / days if days > 0 else 0.0
        
        # Calculate average validator earnings
        avg_validator_earnings = _calculate_avg_validator_earnings(blockchain_client, days)
        
        analysis = {
            "period_days": days,
            "tokenomics": {
                "total_supply": total_supply,
                "circulating_supply": circulating_supply,
                "total_stake": total_stake,
                "staking_ratio": staking_ratio,
            },
            "rewards": {
                "total_distributed": total_distributed,
                "miner_rewards": miner_rewards,
                "validator_rewards": validator_rewards,
                "treasury": treasury,
            },
            "incentives": {
                "avg_miner_earnings": avg_miner_earnings,
                "avg_validator_earnings": avg_validator_earnings,
                "inference_fees_collected": network_stats.get("total_credits", 0.0) * 0.1,  # Estimate
            },
            "trends": _calculate_economic_trends(
                blockchain_client,
                network_stats,
                total_distributed,
                total_stake,
                network_stats.get("total_credits", 0.0) * 0.1,  # inference_fees_collected
                days
            ),
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error getting economic analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch economic analysis")


def _calculate_economic_trends(
    blockchain_client,
    network_stats: Dict,
    current_rewards: float,
    current_stake: float,
    current_fees: float,
    days: int
) -> Dict[str, float]:
    """
    Calculate economic trends (growth rates) from current and estimated historical data.
    
    Args:
        blockchain_client: Blockchain query client
        network_stats: Current network statistics
        current_rewards: Current total rewards distributed
        current_stake: Current total stake
        current_fees: Current inference fees collected
        days: Number of days for trend calculation
        
    Returns:
        Dictionary with trend growth rates
    """
    try:
        # For now, estimate trends based on current metrics
        # In production, this should compare with historical snapshots
        
        # Estimate previous period values (simplified: assume linear growth)
        # This is a placeholder until historical data is available
        previous_rewards = max(0.0, current_rewards * 0.9)  # Estimate: 10% growth
        previous_stake = max(0.0, current_stake * 0.95)  # Estimate: 5% growth
        previous_fees = max(0.0, current_fees * 0.85)  # Estimate: 15% growth
        
        # Calculate growth rates (percentage change per period)
        reward_growth = ((current_rewards - previous_rewards) / previous_rewards * 100) if previous_rewards > 0 else 0.0
        staking_growth = ((current_stake - previous_stake) / previous_stake * 100) if previous_stake > 0 else 0.0
        inference_fee_growth = ((current_fees - previous_fees) / previous_fees * 100) if previous_fees > 0 else 0.0
        
        return {
            "reward_growth": reward_growth,
            "staking_growth": staking_growth,
            "inference_fee_growth": inference_fee_growth,
        }
    except Exception as e:
        logger.warning(f"Failed to calculate economic trends: {e}")
        return {
            "reward_growth": 0.0,
            "staking_growth": 0.0,
            "inference_fee_growth": 0.0,
        }


def _calculate_avg_validator_earnings(blockchain_client, days: int) -> float:
        """
        Calculate average validator earnings.
        
        Args:
            blockchain_client: Blockchain query client
            days: Number of days to analyze
            
        Returns:
            Average validator earnings per day
        """
        try:
            # Get all validators
            validators_result = blockchain_client.get_all_validators(limit=1000, offset=0)
            validators = validators_result.get("validators", [])
            
            if not validators:
                return 0.0
            
            # Calculate total validator rewards from staking module
            # Validator rewards come from:
            # 1. Block rewards (proposer rewards)
            # 2. Transaction fees
            # 3. Commission from delegators
            
            # For now, estimate based on validator voting power and commission rate
            total_validator_rewards = 0.0
            for validator in validators:
                voting_power = validator.get("voting_power", 0.0)
                commission_rate = validator.get("commission_rate", 0.0)
                
                # Estimate: rewards proportional to voting power
                # Base reward per validator per day (simplified)
                base_reward = voting_power * 0.001  # 0.1% of voting power per day
                commission_earnings = base_reward * commission_rate
                total_validator_rewards += base_reward + commission_earnings
            
            # Average per validator
            avg_earnings = total_validator_rewards / len(validators) if validators else 0.0
            
            return avg_earnings
        except Exception as e:
            logger.warning(f"Failed to calculate validator earnings: {e}")
            return 0.0


@router.get("/performance-benchmarks")
async def get_performance_benchmarks():
    """
    Get performance benchmarks for the network.
    
    Returns:
        Performance metrics and comparisons
    """
    try:
        blockchain_client = get_blockchain_client()
        blockchain_stats = blockchain_client.get_network_statistics()
        
        # Get Prometheus metrics for API latency
        # Extract latency percentiles from histogram
        api_latency_p50 = 100.0  # Default
        api_latency_p95 = 200.0  # Default
        api_latency_p99 = 500.0  # Default
        
        try:
            # Get histogram from Prometheus registry
            for metric_family in REGISTRY.collect():
                if metric_family.name == 'api_request_duration_seconds':
                    for metric in metric_family.samples:
                        # We need the full metric family to access all samples
                        # Calculate percentiles from histogram buckets
                        if hasattr(metric_family, 'samples'):
                            # Calculate percentiles
                            api_latency_p50 = _calculate_percentile_from_histogram(metric_family, 0.50) * 1000  # Convert to ms
                            api_latency_p95 = _calculate_percentile_from_histogram(metric_family, 0.95) * 1000
                            api_latency_p99 = _calculate_percentile_from_histogram(metric_family, 0.99) * 1000
                            break
        except Exception as e:
            logger.debug(f"Could not calculate percentiles from histogram: {e}, using defaults")
            pass  # Use defaults
        
        # Calculate cache hit rate
        cache_hits = cache_hits_total._value.get()
        cache_misses = cache_misses_total._value.get()
        total_cache_requests = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0.0
        
        # Network performance metrics
        total_gradients = blockchain_stats.get("total_gradients", 0) if blockchain_stats else 0
        total_aggregations = blockchain_stats.get("total_aggregations", 0) if blockchain_stats else 0
        
        # Estimate block time (default: 5 seconds for Cosmos SDK)
        avg_block_time = 5.0
        avg_transaction_latency = 2.0
        
        # Estimate throughput
        gradients_per_block = total_aggregations / max(1, total_gradients // 100) if total_gradients > 0 else 0
        gradients_per_second = gradients_per_block / avg_block_time if avg_block_time > 0 else 0
        
        benchmarks = {
            "network_performance": {
                "avg_block_time": avg_block_time,
                "avg_transaction_latency": avg_transaction_latency,
                "throughput": {
                    "gradients_per_block": gradients_per_block,
                    "gradients_per_second": gradients_per_second,
                },
            },
            "mining_performance": {
                "avg_gradient_time": _calculate_avg_gradient_time(blockchain_client, blockchain_stats),
                "avg_verification_time": _calculate_avg_verification_time(blockchain_client),
                "bandwidth_reduction": 99.6,  # Fixed: LoRA architecture feature
            },
            "infrastructure": {
                "api_latency_p50": api_latency_p50,
                "api_latency_p95": api_latency_p95,
                "api_latency_p99": api_latency_p99,
                "cache_hit_rate": cache_hit_rate,
            },
        }
        
        return benchmarks
    except Exception as e:
        logger.error(f"Error getting performance benchmarks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch performance benchmarks")


@router.get("/comparative-analysis")
async def get_comparative_analysis(
    metric: str = Query("hashrate", regex="^(hashrate|earnings|efficiency)$")
):
    """
    Get comparative analysis across miners/validators.
    
    Args:
        metric: Metric to compare (hashrate, earnings, efficiency)
        
    Returns:
        Comparative statistics
    """
    try:
        blockchain_client = get_blockchain_client()
        
        # Get all miners
        miners_result = blockchain_client.get_all_miners(limit=1000, offset=0)
        miners = miners_result.get("miners", [])
        
        if not miners:
            return {
                "metric": metric,
                "statistics": {
                    "min": 0.0,
                    "max": 0.0,
                    "median": 0.0,
                    "mean": 0.0,
                    "std_dev": 0.0,
                },
                "percentiles": {
                    "p25": 0.0,
                    "p50": 0.0,
                    "p75": 0.0,
                    "p90": 0.0,
                    "p95": 0.0,
                    "p99": 0.0,
                },
                "distribution": {
                    "tier_breakdown": {
                        "diamond": 0,
                        "platinum": 0,
                        "gold": 0,
                        "silver": 0,
                        "bronze": 0,
                    }
                }
            }
        
        # Extract metric values based on metric type
        if metric == "hashrate":
            values = [m.get("reputation", 0.0) / 10.0 for m in miners]  # Estimate hashrate from reputation
        elif metric == "earnings":
            # Estimate earnings from successful submissions
            values = [m.get("successful_submissions", 0) * 0.1 for m in miners]
        else:  # efficiency
            # Estimate efficiency from success rate
            values = [
                (m.get("successful_submissions", 0) / max(m.get("total_submissions", 1), 1)) * 100
                for m in miners
            ]
        
        # Calculate statistics
        if values:
            values_sorted = sorted(values)
            min_val = min(values)
            max_val = max(values)
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_dev = statistics.stdev(values) if len(values) > 1 else 0.0
            
            # Calculate percentiles
            def percentile(data, p):
                if not data:
                    return 0.0
                k = (len(data) - 1) * p
                f = int(k)
                c = k - f
                if f + 1 < len(data):
                    return data[f] + c * (data[f + 1] - data[f])
                return data[f]
            
            p25 = percentile(values_sorted, 0.25)
            p50 = percentile(values_sorted, 0.50)
            p75 = percentile(values_sorted, 0.75)
            p90 = percentile(values_sorted, 0.90)
            p95 = percentile(values_sorted, 0.95)
            p99 = percentile(values_sorted, 0.99)
        else:
            min_val = max_val = mean_val = median_val = std_dev = 0.0
            p25 = p50 = p75 = p90 = p95 = p99 = 0.0
        
        # Count tiers
        tier_counts = {"diamond": 0, "platinum": 0, "gold": 0, "silver": 0, "bronze": 0}
        for miner in miners:
            tier = miner.get("tier", "bronze")
            if tier in tier_counts:
                tier_counts[tier] += 1
        
        analysis = {
            "metric": metric,
            "statistics": {
                "min": min_val,
                "max": max_val,
                "median": median_val,
                "mean": mean_val,
                "std_dev": std_dev,
            },
            "percentiles": {
                "p25": p25,
                "p50": p50,
                "p75": p75,
                "p90": p90,
                "p95": p95,
                "p99": p99,
            },
            "distribution": {
                "tier_breakdown": tier_counts
            }
        }
        
        return analysis
    except Exception as e:
        logger.error(f"Error getting comparative analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch comparative analysis")

