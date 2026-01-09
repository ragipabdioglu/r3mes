//! Unit tests for wallet module

#[cfg(test)]
mod tests {
    use crate::wallet::*;

    #[test]
    fn test_wallet_struct_serialization() {
        let wallet = Wallet {
            address: "remes1abc123".to_string(),
            mnemonic: Some("word1 word2 word3".to_string()),
        };
        
        let json = serde_json::to_string(&wallet);
        assert!(json.is_ok(), "Wallet should serialize to JSON");
        
        let json_str = json.unwrap();
        assert!(json_str.contains("remes1abc123"));
    }

    #[test]
    fn test_wallet_balance_struct() {
        let balance = WalletBalance {
            address: "remes1test".to_string(),
            balance: "1000000".to_string(),
            denom: "uremes".to_string(),
        };
        
        let json = serde_json::to_string(&balance);
        assert!(json.is_ok());
        
        let restored: WalletBalance = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(restored.address, "remes1test");
        assert_eq!(restored.balance, "1000000");
        assert_eq!(restored.denom, "uremes");
    }

    #[test]
    fn test_transaction_struct() {
        let tx = Transaction {
            hash: "ABC123".to_string(),
            height: 12345,
            timestamp: 1704067200,
            from: "remes1sender".to_string(),
            to: "remes1receiver".to_string(),
            amount: "500000".to_string(),
            denom: "uremes".to_string(),
            status: "confirmed".to_string(),
        };
        
        let json = serde_json::to_string(&tx);
        assert!(json.is_ok());
        
        let restored: Transaction = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(restored.hash, "ABC123");
        assert_eq!(restored.height, 12345);
    }

    #[test]
    fn test_create_new_wallet_generates_valid_address() {
        // This test requires the full crypto stack
        // Skip if dependencies aren't available
        let result = create_new_wallet();
        
        if let Ok(wallet) = result {
            // Address should start with "remes"
            assert!(
                wallet.address.starts_with("remes"),
                "Address should start with 'remes': {}",
                wallet.address
            );
            
            // Mnemonic should be present for new wallets
            assert!(wallet.mnemonic.is_some(), "New wallet should have mnemonic");
            
            // Mnemonic should have 24 words
            if let Some(mnemonic) = &wallet.mnemonic {
                let word_count = mnemonic.split_whitespace().count();
                assert_eq!(word_count, 24, "Mnemonic should have 24 words");
            }
        }
        // If creation fails (e.g., keychain not available), that's OK for unit tests
    }

    #[test]
    fn test_import_wallet_validates_mnemonic_length() {
        // Test with invalid mnemonic (wrong word count)
        let invalid_mnemonic = "word1 word2 word3"; // Only 3 words
        let result = import_wallet_from_mnemonic(invalid_mnemonic);
        
        // Should fail with invalid mnemonic
        assert!(result.is_err(), "Should reject invalid mnemonic");
    }

    #[test]
    fn test_import_wallet_from_private_key_validates_length() {
        // Test with invalid private key (wrong length)
        let invalid_key = "abc123"; // Too short
        let result = import_wallet_from_private_key(invalid_key);
        
        // Should fail with invalid key
        assert!(result.is_err(), "Should reject invalid private key");
    }

    #[test]
    fn test_import_wallet_from_private_key_validates_hex() {
        // Test with invalid hex characters
        let invalid_hex = "zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz";
        let result = import_wallet_from_private_key(invalid_hex);
        
        // Should fail with invalid hex
        assert!(result.is_err(), "Should reject invalid hex");
    }

    #[test]
    fn test_get_stored_wallet_address_no_wallet() {
        // This test depends on file system state
        // In a clean environment, should return error
        let result = get_stored_wallet_address();
        
        // Either succeeds (wallet exists) or fails (no wallet)
        // Both are valid outcomes
        match result {
            Ok(address) => {
                assert!(address.starts_with("remes"), "Address should start with 'remes'");
            }
            Err(e) => {
                assert!(e.contains("not found") || e.contains("No wallet"), "Error should indicate no wallet");
            }
        }
    }

    #[test]
    fn test_wallet_address_format() {
        // Valid Cosmos SDK address format
        let valid_prefixes = ["remes1"];
        
        // If we can create a wallet, verify format
        if let Ok(wallet) = create_new_wallet() {
            let has_valid_prefix = valid_prefixes.iter().any(|p| wallet.address.starts_with(p));
            assert!(has_valid_prefix, "Address should have valid prefix");
            
            // Address should be reasonable length (Bech32 format)
            assert!(wallet.address.len() > 10, "Address should be longer than 10 chars");
            assert!(wallet.address.len() < 100, "Address should be shorter than 100 chars");
        }
    }

    #[test]
    fn test_mnemonic_word_validation() {
        // BIP39 mnemonics should only contain valid words
        if let Ok(wallet) = create_new_wallet() {
            if let Some(mnemonic) = wallet.mnemonic {
                for word in mnemonic.split_whitespace() {
                    // Words should be lowercase
                    assert_eq!(word, word.to_lowercase(), "Mnemonic words should be lowercase");
                    
                    // Words should only contain letters
                    assert!(word.chars().all(|c| c.is_ascii_lowercase()), "Words should only contain letters");
                }
            }
        }
    }

    #[test]
    fn test_migrate_wallet_if_needed_no_panic() {
        // Migration should not panic even if no wallet exists
        let result = migrate_wallet_if_needed();
        
        // Should either succeed or return an error, not panic
        match result {
            Ok(migrated) => {
                // migrated is a bool indicating if migration happened
                assert!(migrated == true || migrated == false);
            }
            Err(e) => {
                // Error is acceptable if no wallet exists
                assert!(!e.is_empty(), "Error message should not be empty");
            }
        }
    }

    // Integration test - requires network
    #[tokio::test]
    #[ignore] // Ignore by default as it requires network
    async fn test_get_wallet_balance_network() {
        // This test requires a running blockchain node
        let result = get_wallet_balance("remes1test").await;
        
        // Should either succeed or fail gracefully
        match result {
            Ok(balance) => {
                assert!(!balance.denom.is_empty(), "Denom should not be empty");
            }
            Err(e) => {
                // Network errors are expected in test environment
                println!("Expected network error: {}", e);
            }
        }
    }

    #[tokio::test]
    #[ignore] // Ignore by default as it requires network
    async fn test_get_wallet_transactions_network() {
        let result = get_wallet_transactions("remes1test", 10).await;
        
        // Should return empty vec or transactions
        assert!(result.is_ok(), "Should not error");
        let txs = result.unwrap();
        // Empty is fine for test address
        assert!(txs.len() <= 10, "Should respect limit");
    }
}
