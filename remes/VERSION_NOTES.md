# Version Notes

## Current Versions

- **Go**: 1.22 (as specified in task, system has 1.24.1 which is compatible)
- **Cosmos SDK**: v0.50.6 (LTS version as specified in task)
- **CometBFT**: v0.38.19 (closest available to v0.38.27 specified in task)

## Version Compatibility Notes

### CometBFT v0.38.27
The exact version v0.38.27 specified in the task does not exist. The closest available version is v0.38.19, which is compatible with Cosmos SDK v0.50.x.

### Cosmos SDK v0.50.6
This is the LTS (Long Term Support) version as specified in the task requirements. It is compatible with:
- Go 1.21+ (we use 1.22)
- CometBFT v0.38.x (we use v0.38.19)

## Future Updates

If CometBFT v0.38.27 becomes available in the future, update go.mod accordingly:
```go
github.com/cometbft/cometbft v0.38.27
```

## Verification

To verify versions are correct:
```bash
go mod verify
go list -m github.com/cosmos/cosmos-sdk
go list -m github.com/cometbft/cometbft
```

