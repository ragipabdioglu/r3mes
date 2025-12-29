package keeper

import (
	"context"
	"encoding/json"

	"remes/x/remes/types"
)

// GetRewardFormula queries transparent reward formulas
func (k Keeper) GetRewardFormula(ctx context.Context, req *types.QueryGetRewardFormulaRequest) (*types.QueryGetRewardFormulaResponse, error) {
	formulas := k.GetTransparentRewardFormula()
	
	// Convert to JSON string
	formulasJSON, err := json.Marshal(formulas)
	if err != nil {
		return nil, err
	}
	
	return &types.QueryGetRewardFormulaResponse{
		Formulas: string(formulasJSON),
	}, nil
}

