package remes

import (
	autocliv1 "cosmossdk.io/api/cosmos/autocli/v1"

	"remes/x/remes/types"
)

// AutoCLIOptions implements the autocli.HasAutoCLIConfig interface.
func (am AppModule) AutoCLIOptions() *autocliv1.ModuleOptions {
	return &autocliv1.ModuleOptions{
		Query: &autocliv1.ServiceCommandDescriptor{
			Service: types.Query_serviceDesc.ServiceName,
			RpcCommandOptions: []*autocliv1.RpcCommandOptions{
				{
					RpcMethod: "Params",
					Use:       "params",
					Short:     "Shows the parameters of the module",
				},
				{
					RpcMethod: "GetGradient",
					Use:       "get-gradient [id]",
					Short:     "Query gradient by ID",
					Long:      "Query a stored gradient by its ID",
					PositionalArgs: []*autocliv1.PositionalArgDescriptor{
						{
							ProtoField: "id",
						},
					},
				},
				{
					RpcMethod: "GetModelParams",
					Use:       "model-params",
					Short:     "Query current model parameters",
					Long:      "Query the current global model state and parameters",
				},
				{
					RpcMethod: "GetAggregation",
					Use:       "get-aggregation [id]",
					Short:     "Query aggregation by ID",
					Long:      "Query an aggregation record by its ID",
					PositionalArgs: []*autocliv1.PositionalArgDescriptor{
						{
							ProtoField: "id",
						},
					},
				},
				{
					RpcMethod: "GetMinerScore",
					Use:       "get-miner-score [miner]",
					Short:     "Query miner score and statistics",
					Long:      "Query mining contribution statistics for a miner address",
					PositionalArgs: []*autocliv1.PositionalArgDescriptor{
						{
							ProtoField: "miner",
						},
					},
				},
				{
					RpcMethod: "ListStoredGradient",
					Use:       "list-gradients",
					Short:     "List stored gradients",
					Long:      "List all stored gradients with pagination",
				},
				{
					RpcMethod: "GetStoredGradient",
					Use:       "get-stored-gradient [id]",
					Short:     "Query stored gradient by ID",
					Long:      "Query a stored gradient metadata by its ID",
					PositionalArgs: []*autocliv1.PositionalArgDescriptor{
						{
							ProtoField: "id",
						},
					},
				},
			},
		},
		Tx: &autocliv1.ServiceCommandDescriptor{
			Service:              types.Msg_serviceDesc.ServiceName,
			EnhanceCustomCommand: true, // only required if you want to use the custom command
			RpcCommandOptions: []*autocliv1.RpcCommandOptions{
				{
					RpcMethod: "UpdateParams",
					Skip:      true, // skipped because authority gated
				},
				{
					RpcMethod: "SubmitGradient",
					Use:       "submit-gradient [ipfs-hash] [gradient-hash] [model-version] [gpu-architecture]",
					Short:     "Submit a gradient for mining",
					Long:      "Submit a gradient stored on IPFS to the blockchain for mining rewards",
				},
				{
					RpcMethod: "SubmitAggregation",
					Use:       "submit-aggregation [aggregated-gradient-ipfs-hash] [merkle-root]",
					Short:     "Submit an aggregated gradient",
					Long:      "Submit an aggregated gradient as a proposer to earn proposer rewards",
				},
				{
					RpcMethod: "ChallengeAggregation",
					Use:       "challenge-aggregation [aggregation-id] [evidence-ipfs-hash]",
					Short:     "Challenge an aggregation",
					Long:      "Challenge an aggregation with evidence if it's invalid",
				},
			},
		},
	}
}
