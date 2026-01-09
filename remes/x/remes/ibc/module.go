package ibc

import (
	"encoding/json"
	"fmt"

	"github.com/cosmos/cosmos-sdk/codec"
	sdk "github.com/cosmos/cosmos-sdk/types"
	capabilitytypes "github.com/cosmos/ibc-go/modules/capability/types"
	channeltypes "github.com/cosmos/ibc-go/v8/modules/core/04-channel/types"
	porttypes "github.com/cosmos/ibc-go/v8/modules/core/05-port/types"
	host "github.com/cosmos/ibc-go/v8/modules/core/24-host"
	ibcexported "github.com/cosmos/ibc-go/v8/modules/core/exported"
	clienttypes "github.com/cosmos/ibc-go/v8/modules/core/02-client/types"

	"remes/x/remes/keeper"
	"remes/x/remes/types"
)

var (
	_ porttypes.IBCModule = (*IBCModule)(nil)
)

// IBCModule implements the ICS26 interface for the R3MES module
type IBCModule struct {
	keeper keeper.Keeper
	cdc    codec.Codec
}

// NewIBCModule creates a new IBCModule instance
func NewIBCModule(keeper keeper.Keeper, cdc codec.Codec) IBCModule {
	return IBCModule{
		keeper: keeper,
		cdc:    cdc,
	}
}

// OnChanOpenInit implements the IBCModule interface
func (im IBCModule) OnChanOpenInit(
	ctx sdk.Context,
	order channeltypes.Order,
	connectionHops []string,
	portID string,
	channelID string,
	channelCap *capabilitytypes.Capability,
	counterparty channeltypes.Counterparty,
	version string,
) (string, error) {
	if order != channeltypes.ORDERED {
		return "", fmt.Errorf("only ORDERED channels are supported for R3MES gradient synchronization")
	}

	if version != types.IBCVersion {
		return "", fmt.Errorf("invalid IBC version: expected %s, got %s", types.IBCVersion, version)
	}

	if err := im.keeper.GetCoreKeeper().ClaimCapability(ctx, channelCap, host.ChannelCapabilityPath(portID, channelID)); err != nil {
		return "", err
	}

	ctx.Logger().Info("IBC channel opened", "port_id", portID, "channel_id", channelID)
	return version, nil
}

// OnChanOpenTry implements the IBCModule interface
func (im IBCModule) OnChanOpenTry(
	ctx sdk.Context,
	order channeltypes.Order,
	connectionHops []string,
	portID string,
	channelID string,
	channelCap *capabilitytypes.Capability,
	counterparty channeltypes.Counterparty,
	counterpartyVersion string,
) (string, error) {
	if order != channeltypes.ORDERED {
		return "", fmt.Errorf("only ORDERED channels are supported for R3MES gradient synchronization")
	}

	if counterpartyVersion != types.IBCVersion {
		return "", fmt.Errorf("invalid counterparty IBC version: expected %s, got %s", types.IBCVersion, counterpartyVersion)
	}

	if err := im.keeper.GetCoreKeeper().ClaimCapability(ctx, channelCap, host.ChannelCapabilityPath(portID, channelID)); err != nil {
		return "", err
	}

	ctx.Logger().Info("IBC channel try opened", "port_id", portID, "channel_id", channelID)
	return types.IBCVersion, nil
}

// OnChanOpenAck implements the IBCModule interface
func (im IBCModule) OnChanOpenAck(
	ctx sdk.Context,
	portID string,
	channelID string,
	counterpartyChannelID string,
	counterpartyVersion string,
) error {
	if counterpartyVersion != types.IBCVersion {
		return fmt.Errorf("invalid counterparty IBC version: expected %s, got %s", types.IBCVersion, counterpartyVersion)
	}
	ctx.Logger().Info("IBC channel acknowledged", "port_id", portID, "channel_id", channelID)
	return nil
}

// OnChanOpenConfirm implements the IBCModule interface
func (im IBCModule) OnChanOpenConfirm(ctx sdk.Context, portID string, channelID string) error {
	ctx.Logger().Info("IBC channel confirmed", "port_id", portID, "channel_id", channelID)
	return nil
}

// OnChanCloseInit implements the IBCModule interface
func (im IBCModule) OnChanCloseInit(ctx sdk.Context, portID string, channelID string) error {
	return fmt.Errorf("user cannot close channel")
}

// OnChanCloseConfirm implements the IBCModule interface
func (im IBCModule) OnChanCloseConfirm(ctx sdk.Context, portID string, channelID string) error {
	ctx.Logger().Info("IBC channel closed", "port_id", portID, "channel_id", channelID)
	return nil
}

// OnRecvPacket implements the IBCModule interface
func (im IBCModule) OnRecvPacket(
	ctx sdk.Context,
	packet channeltypes.Packet,
	relayer sdk.AccAddress,
) ibcexported.Acknowledgement {
	var data types.IBCGradientPacketData
	if err := json.Unmarshal(packet.GetData(), &data); err != nil {
		ctx.Logger().Error("failed to unmarshal IBC packet data", "error", err)
		return channeltypes.NewErrorAcknowledgement(fmt.Errorf("failed to unmarshal packet data: %w", err))
	}

	ctx.Logger().Info("received IBC gradient packet",
		"source_chain", packet.SourceChannel,
		"gradient_id", data.GradientID,
		"ipfs_hash", data.IPFSHash,
	)

	if err := im.processGradientPacket(ctx, data); err != nil {
		ctx.Logger().Error("failed to process gradient packet", "error", err)
		return channeltypes.NewErrorAcknowledgement(err)
	}

	ack := types.IBCGradientPacketAck{
		Success: true,
		Message: "Gradient received and processed successfully",
	}

	ackBytes, err := json.Marshal(ack)
	if err != nil {
		return channeltypes.NewErrorAcknowledgement(fmt.Errorf("failed to marshal acknowledgement: %w", err))
	}

	return channeltypes.NewResultAcknowledgement(ackBytes)
}

// OnAcknowledgementPacket implements the IBCModule interface
func (im IBCModule) OnAcknowledgementPacket(
	ctx sdk.Context,
	packet channeltypes.Packet,
	acknowledgement []byte,
	relayer sdk.AccAddress,
) error {
	var ack types.IBCGradientPacketAck
	if err := json.Unmarshal(acknowledgement, &ack); err != nil {
		ctx.Logger().Error("failed to unmarshal acknowledgement", "error", err)
		return err
	}

	ctx.Logger().Info("received IBC acknowledgement", "success", ack.Success, "message", ack.Message)

	if !ack.Success {
		ctx.Logger().Error("gradient packet failed on counterparty chain", "message", ack.Message)
	}

	return nil
}

// OnTimeoutPacket implements the IBCModule interface
func (im IBCModule) OnTimeoutPacket(
	ctx sdk.Context,
	packet channeltypes.Packet,
	relayer sdk.AccAddress,
) error {
	ctx.Logger().Error("IBC packet timed out",
		"source_channel", packet.SourceChannel,
		"destination_channel", packet.DestinationChannel,
	)
	return nil
}

// processGradientPacket processes a received gradient packet
func (im IBCModule) processGradientPacket(ctx sdk.Context, data types.IBCGradientPacketData) error {
	if data.GradientID == 0 {
		return fmt.Errorf("invalid gradient ID: %d", data.GradientID)
	}

	if data.IPFSHash == "" {
		return fmt.Errorf("empty IPFS hash")
	}

	// Create stored gradient from IBC packet
	// Note: SubmittedAtHeight will be set by the keeper
	gradient := types.StoredGradient{
		Id:              data.GradientID,
		Miner:           data.MinerAddress,
		IpfsHash:        data.IPFSHash,
		ModelVersion:    data.ModelVersion,
		TrainingRoundId: data.TrainingRoundID,
		ShardId:         data.ShardID,
		GradientHash:    data.GradientHash,
		GpuArchitecture: data.GPUArchitecture,
		SubmittedAtHeight: ctx.BlockHeight(),
		Status:          "received_via_ibc",
	}

	if err := im.keeper.GetTrainingKeeper().SubmitGradient(ctx, gradient); err != nil {
		return fmt.Errorf("failed to store gradient: %w", err)
	}

	ctx.Logger().Info("gradient stored from IBC packet",
		"gradient_id", gradient.Id,
		"miner", gradient.Miner,
	)

	return nil
}

// SendGradientPacket sends a gradient packet to another chain via IBC
func (im IBCModule) SendGradientPacket(
	ctx sdk.Context,
	sourcePort string,
	sourceChannel string,
	gradient types.StoredGradient,
	timeoutHeight uint64,
	timeoutTimestamp uint64,
) error {
	packetData := types.IBCGradientPacketData{
		GradientID:      gradient.Id,
		MinerAddress:    gradient.Miner,
		IPFSHash:        gradient.IpfsHash,
		ModelVersion:    gradient.ModelVersion,
		TrainingRoundID: gradient.TrainingRoundId,
		ShardID:         gradient.ShardId,
		GradientHash:    gradient.GradientHash,
		GPUArchitecture: gradient.GpuArchitecture,
		SourceChain:     ctx.ChainID(),
	}

	packetBytes, err := json.Marshal(packetData)
	if err != nil {
		return fmt.Errorf("failed to marshal packet data: %w", err)
	}

	packet := channeltypes.NewPacket(
		packetBytes,
		1,
		sourcePort,
		sourceChannel,
		"",
		"",
		clienttypes.NewHeight(0, timeoutHeight),
		timeoutTimestamp,
	)

	channelCap, ok := im.keeper.GetCoreKeeper().GetCapability(ctx, host.ChannelCapabilityPath(sourcePort, sourceChannel))
	if !ok {
		return fmt.Errorf("channel capability not found")
	}

	if err := im.keeper.GetCoreKeeper().SendPacket(ctx, channelCap, packet); err != nil {
		return fmt.Errorf("failed to send IBC packet: %w", err)
	}

	ctx.Logger().Info("gradient packet sent via IBC", "gradient_id", gradient.Id, "source_channel", sourceChannel)
	return nil
}
