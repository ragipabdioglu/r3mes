package types

// SubmissionRecord tracks submission rate for a miner
// This is used for rate limiting
type SubmissionRecord struct {
	MinerAddress    string
	SubmissionTimes []int64 // Unix timestamps of recent submissions
}

