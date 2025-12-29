package keeper

import (
	"math/big"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	sdkmath "cosmossdk.io/math"
)

var (
	// Block metrics
	blockHeight = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "r3mes_blockchain_block_height",
			Help: "Current block height",
		},
		[]string{"chain_id"},
	)

	blockTime = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "r3mes_blockchain_block_time_seconds",
			Help:    "Block time in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		},
		[]string{"chain_id"},
	)

	// Transaction metrics
	transactionsTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "r3mes_blockchain_transactions_total",
			Help: "Total number of transactions",
		},
		[]string{"chain_id", "type", "status"},
	)

	// Miner metrics
	activeMiners = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "r3mes_blockchain_active_miners",
			Help: "Number of active miners",
		},
		[]string{"chain_id"},
	)

	totalGradients = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "r3mes_blockchain_gradients_total",
			Help: "Total number of gradients submitted",
		},
		[]string{"chain_id"},
	)

	// Aggregation metrics
	totalAggregations = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "r3mes_blockchain_aggregations_total",
			Help: "Total number of aggregations",
		},
		[]string{"chain_id"},
	)

	// Verification metrics
	verificationDuration = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "r3mes_blockchain_verification_duration_seconds",
			Help:    "Verification duration in seconds",
			Buckets: prometheus.ExponentialBuckets(0.1, 2, 10),
		},
		[]string{"chain_id", "layer"},
	)

	verificationTotal = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "r3mes_blockchain_verifications_total",
			Help: "Total number of verifications",
		},
		[]string{"chain_id", "layer", "result"},
	)

	// Staking metrics
	totalStaked = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "r3mes_blockchain_total_staked",
			Help: "Total staked tokens",
		},
		[]string{"chain_id", "denom"},
	)

	// Governance metrics
	activeProposals = prometheus.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "r3mes_blockchain_active_proposals",
			Help: "Number of active governance proposals",
		},
		[]string{"chain_id"},
	)
)

func init() {
	// Register all metrics
	prometheus.MustRegister(blockHeight)
	prometheus.MustRegister(blockTime)
	prometheus.MustRegister(transactionsTotal)
	prometheus.MustRegister(activeMiners)
	prometheus.MustRegister(totalGradients)
	prometheus.MustRegister(totalAggregations)
	prometheus.MustRegister(verificationDuration)
	prometheus.MustRegister(verificationTotal)
	prometheus.MustRegister(totalStaked)
	prometheus.MustRegister(activeProposals)
}

// UpdateBlockHeight updates the block height metric
func UpdateBlockHeight(chainID string, height int64) {
	blockHeight.WithLabelValues(chainID).Set(float64(height))
}

// RecordBlockTime records block time metric
func RecordBlockTime(chainID string, duration time.Duration) {
	blockTime.WithLabelValues(chainID).Observe(duration.Seconds())
}

// IncrementTransactions increments transaction counter
func IncrementTransactions(chainID, txType, status string) {
	transactionsTotal.WithLabelValues(chainID, txType, status).Inc()
}

// UpdateActiveMiners updates active miners count
func UpdateActiveMiners(chainID string, count int) {
	activeMiners.WithLabelValues(chainID).Set(float64(count))
}

// IncrementGradients increments gradient counter
func IncrementGradients(chainID string) {
	totalGradients.WithLabelValues(chainID).Inc()
}

// IncrementAggregations increments aggregation counter
func IncrementAggregations(chainID string) {
	totalAggregations.WithLabelValues(chainID).Inc()
}

// RecordVerification records verification metrics
func RecordVerification(chainID, layer, result string, duration time.Duration) {
	verificationDuration.WithLabelValues(chainID, layer).Observe(duration.Seconds())
	verificationTotal.WithLabelValues(chainID, layer, result).Inc()
}

// UpdateTotalStaked updates total staked tokens
func UpdateTotalStaked(chainID, denom string, amount sdkmath.Int) {
	// Convert sdkmath.Int to float64 for Prometheus
	amountBig := amount.BigInt()
	amountFloat := new(big.Float).SetInt(amountBig)
	amountFloat64, _ := amountFloat.Float64()
	totalStaked.WithLabelValues(chainID, denom).Set(amountFloat64)
}

// UpdateActiveProposals updates active proposals count
func UpdateActiveProposals(chainID string, count int) {
	activeProposals.WithLabelValues(chainID).Set(float64(count))
}

// MetricsHandler returns HTTP handler for Prometheus metrics endpoint
func MetricsHandler() http.Handler {
	return promhttp.Handler()
}

