# Proto Generate Required

After updating `remes/proto/remes/remes/v1/params.proto` with ScalabilityParams, you need to regenerate the Go code:

```bash
cd remes
make proto-gen
```

This will generate:
- `remes/x/remes/types/params.pb.go` (with ScalabilityParams type)
- `remes/x/remes/types/remes/remes/v1/params.pb.go`

The code in `remes/x/remes/types/params.go` and `remes/x/remes/keeper/scalability.go` is ready and will compile after proto generation.

