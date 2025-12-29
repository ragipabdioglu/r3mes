package keeper_test

import (
	"testing"

	sdk "github.com/cosmos/cosmos-sdk/types"
	"github.com/stretchr/testify/require"
)

func TestValidateGPUArchitecture(t *testing.T) {
	f := initFixture(t)
	ctx := sdk.UnwrapSDKContext(f.ctx)

	// Test: Valid architectures
	validArchitectures := []string{"Pascal", "Volta", "Turing", "Ampere", "Ada", "Blackwell"}
	for _, arch := range validArchitectures {
		err := f.keeper.ValidateGPUArchitecture(ctx, arch)
		require.NoError(t, err, "architecture %s should be valid", arch)
	}

	// Test: Case insensitive
	err := f.keeper.ValidateGPUArchitecture(ctx, "ampere")
	require.NoError(t, err, "architecture should be case insensitive")

	err = f.keeper.ValidateGPUArchitecture(ctx, "AMPERE")
	require.NoError(t, err, "architecture should be case insensitive")

	// Test: Invalid architectures
	invalidArchitectures := []string{"Maxwell", "Kepler", "Unknown", "AMD", "Intel"}
	for _, arch := range invalidArchitectures {
		err := f.keeper.ValidateGPUArchitecture(ctx, arch)
		require.Error(t, err, "architecture %s should be invalid", arch)
		require.Contains(t, err.Error(), "unsupported GPU architecture")
	}

	// Test: Empty architecture
	err = f.keeper.ValidateGPUArchitecture(ctx, "")
	require.Error(t, err)
	require.Contains(t, err.Error(), "cannot be empty")
}

func TestIsGPUArchitectureSupported(t *testing.T) {
	f := initFixture(t)

	// Test: Supported architectures
	require.True(t, f.keeper.IsGPUArchitectureSupported("Ampere"))
	require.True(t, f.keeper.IsGPUArchitectureSupported("ampere"))
	require.True(t, f.keeper.IsGPUArchitectureSupported("AMPERE"))

	// Test: Unsupported architectures
	require.False(t, f.keeper.IsGPUArchitectureSupported("Maxwell"))
	require.False(t, f.keeper.IsGPUArchitectureSupported(""))
	require.False(t, f.keeper.IsGPUArchitectureSupported("Unknown"))
}

func TestGetSupportedArchitecturesList(t *testing.T) {
	f := initFixture(t)

	list := f.keeper.GetSupportedArchitecturesList()
	require.NotEmpty(t, list)
	require.Contains(t, list, "Ampere")
	require.Contains(t, list, "Pascal")
	require.Contains(t, list, "Volta")
	require.Contains(t, list, "Turing")
	require.Contains(t, list, "Ada")
	require.Contains(t, list, "Blackwell")
}

