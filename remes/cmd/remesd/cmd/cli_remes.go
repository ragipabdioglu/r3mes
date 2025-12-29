package cmd

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/spf13/cobra"
	"github.com/cosmos/cosmos-sdk/client"
	"github.com/cosmos/cosmos-sdk/client/flags"
	"github.com/cosmos/cosmos-sdk/version"

	"remes/x/remes/types"
)

// GetRemesQueryCmd returns the remes module query commands
func GetRemesQueryCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:                        types.ModuleName,
		Short:                      "Querying commands for the remes module",
		DisableFlagParsing:         true,
		SuggestionsMinimumDistance: 2,
		RunE:                       client.ValidateCmd,
	}

	cmd.AddCommand(
		GetCmdQueryParams(),
		GetCmdQueryGradient(),
		GetCmdQueryMinerScore(),
		GetCmdQueryStoredGradient(),
		GetCmdQueryDataset(),
		GetCmdQueryNode(),
		GetCmdQueryNetworkStats(),
		GetCmdQueryModelInfo(),
	)

	return cmd
}

// GetCmdQueryParams returns the query params command
func GetCmdQueryParams() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "params",
		Short: "Query the current remes module parameters",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.Params(cmd.Context(), &types.QueryParamsRequest{})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryGradient returns the query gradient command
func GetCmdQueryGradient() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "gradient [gradient-id]",
		Short: "Query a gradient by ID",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			gradientID, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				return fmt.Errorf("invalid gradient ID: %w", err)
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetGradient(cmd.Context(), &types.QueryGetGradientRequest{
				Id: gradientID,
			})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryMinerScore returns the query miner score command
func GetCmdQueryMinerScore() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "miner-score [miner-address]",
		Short: "Query miner score and reputation",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetMinerScore(cmd.Context(), &types.QueryGetMinerScoreRequest{
				Miner: args[0],
			})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryStoredGradient returns the query stored gradient command
func GetCmdQueryStoredGradient() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "stored-gradient [id]",
		Short: "Query a stored gradient by ID",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			gradientID, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				return fmt.Errorf("invalid gradient ID: %w", err)
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetStoredGradient(cmd.Context(), &types.QueryGetStoredGradientRequest{
				Id: gradientID,
			})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryDataset returns the query dataset command
func GetCmdQueryDataset() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "dataset [dataset-id]",
		Short: "Query an approved dataset by ID",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			datasetID, err := strconv.ParseUint(args[0], 10, 64)
			if err != nil {
				return fmt.Errorf("invalid dataset ID: %w", err)
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetApprovedDataset(cmd.Context(), &types.QueryGetApprovedDatasetRequest{
				DatasetId: datasetID,
			})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryNode returns the query node command
func GetCmdQueryNode() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "node [node-address]",
		Short: "Query node registration information",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetNodeRegistration(cmd.Context(), &types.QueryGetNodeRegistrationRequest{
				NodeAddress: args[0],
			})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryNetworkStats returns the query network statistics command
func GetCmdQueryNetworkStats() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "network-stats",
		Short: "Query network-wide statistics",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			// Query multiple endpoints to build network stats
			queryClient := types.NewQueryClient(clientCtx)

			// Get all stored gradients (with pagination)
			gradientsRes, err := queryClient.ListStoredGradient(cmd.Context(), &types.QueryListStoredGradientRequest{
				Pagination: nil, // Get all for stats
			})
			if err != nil {
				return err
			}

			// Get all nodes
			nodesRes, err := queryClient.ListNodeRegistrations(cmd.Context(), &types.QueryListNodeRegistrationsRequest{
				Pagination: nil,
			})
			if err != nil {
				return err
			}

			// Build stats
			stats := map[string]interface{}{
				"total_gradients": len(gradientsRes.StoredGradients),
				"total_nodes":     len(nodesRes.Registrations),
				"version":         version.Version,
			}

			// Count active miners
			activeMiners := 0
			for _, node := range nodesRes.Registrations {
				if node.Status == types.NODE_STATUS_ACTIVE {
					activeMiners++
				}
			}
			stats["active_miners"] = activeMiners

			// Output as JSON
			output, err := json.MarshalIndent(stats, "", "  ")
			if err != nil {
				return err
			}

			fmt.Println(string(output))
			return nil
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

// GetCmdQueryModelInfo returns the query model information command
func GetCmdQueryModelInfo() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "model-info",
		Short: "Query current global model parameters (IPFS hash and version)",
		Args:  cobra.NoArgs,
		RunE: func(cmd *cobra.Command, args []string) error {
			clientCtx, err := client.GetClientQueryContext(cmd)
			if err != nil {
				return err
			}

			queryClient := types.NewQueryClient(clientCtx)

			res, err := queryClient.GetModelParams(cmd.Context(), &types.QueryGetModelParamsRequest{})
			if err != nil {
				return err
			}

			return clientCtx.PrintProto(res)
		},
	}

	flags.AddQueryFlagsToCmd(cmd)
	return cmd
}

