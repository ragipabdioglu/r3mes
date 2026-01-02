package main

import (
	"fmt"
	"os"

	clienthelpers "cosmossdk.io/client/v2/helpers"
	svrcmd "github.com/cosmos/cosmos-sdk/server/cmd"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/app"
	"remes/cmd/remesd/cmd"
	remeskeeper "remes/x/remes/keeper"
)

func main() {
	// Set SDK config with correct bech32 prefixes BEFORE anything else
	config := sdk.GetConfig()
	config.SetBech32PrefixForAccount(app.AccountAddressPrefix, app.AccountAddressPrefix+"pub")
	config.SetBech32PrefixForValidator(app.AccountAddressPrefix+"valoper", app.AccountAddressPrefix+"valoperpub")
	config.SetBech32PrefixForConsensusNode(app.AccountAddressPrefix+"valcons", app.AccountAddressPrefix+"valconspub")
	config.SetCoinType(app.ChainCoinType)
	config.Seal()

	// Validate environment configuration before starting
	if validationErrors := app.ValidateEnvironment(); len(validationErrors) > 0 {
		app.PrintValidationErrors(validationErrors)
		// In development, show warnings but continue
		// In production, this could be made fatal
		fmt.Println("⚠️  Continuing with validation warnings...")
	}

	// Initialize Sentry for error tracking
	if err := remeskeeper.InitSentry(); err != nil {
		// Sentry initialization failure should not prevent app from starting
		fmt.Fprintf(os.Stderr, "Warning: Failed to initialize Sentry: %v\n", err)
	}

	rootCmd := cmd.NewRootCmd()
	if err := svrcmd.Execute(rootCmd, clienthelpers.EnvPrefix, app.DefaultNodeHome); err != nil {
		// Capture error to Sentry before exiting
		remeskeeper.CaptureException(err, map[string]string{
			"component": "main",
			"command":   rootCmd.Name(),
		})
		fmt.Fprintln(rootCmd.OutOrStderr(), err)
		os.Exit(1)
	}
}
