package types

// NonceWindow represents a nonce window for a miner
// Used for replay attack prevention with sliding window
type NonceWindow struct {
	MinNonce uint64 `json:"min_nonce"`
	MaxNonce uint64 `json:"max_nonce"`
}

// NewNonceWindow creates a new nonce window
func NewNonceWindow(minNonce, maxNonce uint64) NonceWindow {
	return NonceWindow{
		MinNonce: minNonce,
		MaxNonce: maxNonce,
	}
}

// IsInWindow checks if a nonce is within the window
func (w NonceWindow) IsInWindow(nonce uint64) bool {
	return nonce >= w.MinNonce && nonce <= w.MaxNonce
}

// Slide slides the window forward
func (w *NonceWindow) Slide(newMaxNonce uint64) {
	windowSize := w.MaxNonce - w.MinNonce
	w.MaxNonce = newMaxNonce
	w.MinNonce = newMaxNonce - windowSize
}
