package keeper

import (
	"sync"
	"time"
)

// CachedGradient represents a cached gradient tensor
type CachedGradient struct {
	Gradient []float64
	ExpiresAt time.Time
}

// GradientCache provides in-memory caching for gradient tensors
// Used to avoid repeated IPFS downloads for the same gradient
type GradientCache struct {
	cache map[string]*CachedGradient
	mu    sync.RWMutex
	ttl   time.Duration // Time-to-live for cache entries
}

// NewGradientCache creates a new gradient cache with specified TTL
func NewGradientCache(ttl time.Duration) *GradientCache {
	return &GradientCache{
		cache: make(map[string]*CachedGradient),
		ttl:   ttl,
	}
}

// Get retrieves a gradient from cache if available and not expired
func (gc *GradientCache) Get(ipfsHash string) ([]float64, bool) {
	gc.mu.RLock()
	defer gc.mu.RUnlock()

	cached, exists := gc.cache[ipfsHash]
	if !exists {
		return nil, false
	}

	// Check if expired
	if time.Now().After(cached.ExpiresAt) {
		// Entry expired, remove it (async cleanup)
		go gc.remove(ipfsHash)
		return nil, false
	}

	// Return cached gradient (make a copy to avoid mutation)
	gradient := make([]float64, len(cached.Gradient))
	copy(gradient, cached.Gradient)
	return gradient, true
}

// Set stores a gradient in cache with TTL
func (gc *GradientCache) Set(ipfsHash string, gradient []float64) {
	gc.mu.Lock()
	defer gc.mu.Unlock()

	// Make a copy to avoid mutation
	gradientCopy := make([]float64, len(gradient))
	copy(gradientCopy, gradient)

	gc.cache[ipfsHash] = &CachedGradient{
		Gradient:  gradientCopy,
		ExpiresAt: time.Now().Add(gc.ttl),
	}
}

// remove removes an entry from cache (internal use, assumes lock is held)
func (gc *GradientCache) remove(ipfsHash string) {
	gc.mu.Lock()
	defer gc.mu.Unlock()
	delete(gc.cache, ipfsHash)
}

// ClearExpired removes all expired entries from cache
func (gc *GradientCache) ClearExpired() {
	gc.mu.Lock()
	defer gc.mu.Unlock()

	now := time.Now()
	for hash, cached := range gc.cache {
		if now.After(cached.ExpiresAt) {
			delete(gc.cache, hash)
		}
	}
}

// Size returns the number of entries in cache
func (gc *GradientCache) Size() int {
	gc.mu.RLock()
	defer gc.mu.RUnlock()
	return len(gc.cache)
}

