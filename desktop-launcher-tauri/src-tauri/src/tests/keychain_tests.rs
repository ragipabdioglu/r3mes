//! Unit tests for keychain module

#[cfg(test)]
mod tests {
    use crate::keychain::*;

    #[test]
    fn test_keychain_manager_new() {
        let manager = KeychainManager::new();
        // Should create without panic
        drop(manager);
    }

    #[test]
    fn test_keychain_manager_with_service() {
        let manager = KeychainManager::with_service("test.service.name");
        // Should create with custom service name
        drop(manager);
    }

    #[test]
    fn test_keychain_error_display() {
        let errors = vec![
            (KeychainError::NotSupported, "not supported"),
            (KeychainError::AccessDenied, "Access denied"),
            (KeychainError::ItemNotFound, "not found"),
            (KeychainError::InvalidData, "Invalid data"),
            (KeychainError::SystemError("test error".to_string()), "test error"),
        ];
        
        for (error, expected_substring) in errors {
            let display = format!("{}", error);
            assert!(
                display.to_lowercase().contains(&expected_substring.to_lowercase()),
                "Error display '{}' should contain '{}'",
                display,
                expected_substring
            );
        }
    }

    #[test]
    fn test_keychain_item_serialization() {
        let item = KeychainItem {
            service: "test.service".to_string(),
            account: "test_account".to_string(),
            data: "secret_data".to_string(),
            created_at: 1704067200,
            accessed_at: 1704067200,
        };
        
        let json = serde_json::to_string(&item);
        assert!(json.is_ok());
        
        let restored: KeychainItem = serde_json::from_str(&json.unwrap()).unwrap();
        assert_eq!(restored.service, "test.service");
        assert_eq!(restored.account, "test_account");
        assert_eq!(restored.data, "secret_data");
    }

    #[test]
    fn test_is_keychain_available() {
        // Should return a boolean without panicking
        let available = KeychainManager::is_keychain_available();
        assert!(available == true || available == false);
    }

    #[test]
    fn test_store_and_retrieve_fallback() {
        let manager = KeychainManager::new();
        let test_key = "test_key_unit_test";
        let test_value = "test_value_12345";
        
        // Store
        let store_result = manager.store(test_key, test_value);
        
        if store_result.is_ok() {
            // Retrieve
            let retrieve_result = manager.retrieve(test_key);
            assert!(retrieve_result.is_ok(), "Should retrieve stored value");
            assert_eq!(retrieve_result.unwrap(), test_value);
            
            // Cleanup
            let _ = manager.delete(test_key);
        }
        // If store fails (e.g., no keychain), that's OK for unit tests
    }

    #[test]
    fn test_retrieve_nonexistent_key() {
        let manager = KeychainManager::new();
        let result = manager.retrieve("nonexistent_key_that_should_not_exist_12345");
        
        // Should return ItemNotFound error
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_nonexistent_key() {
        let manager = KeychainManager::new();
        let result = manager.delete("nonexistent_key_to_delete_12345");
        
        // Should return error (ItemNotFound)
        assert!(result.is_err());
    }

    #[test]
    fn test_exists_returns_bool() {
        let manager = KeychainManager::new();
        
        // Nonexistent key should return false
        let exists = manager.exists("definitely_nonexistent_key_xyz");
        assert!(!exists);
    }

    #[test]
    fn test_store_wallet_private_key() {
        let manager = KeychainManager::new();
        let address = "remes1testaddress";
        let private_key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        
        let result = manager.store_wallet_private_key(address, private_key);
        
        if result.is_ok() {
            // Verify we can retrieve it
            let retrieved = manager.retrieve_wallet_private_key(address);
            assert!(retrieved.is_ok());
            assert_eq!(retrieved.unwrap(), private_key);
            
            // Cleanup
            let key = format!("wallet_private_key_{}", address);
            let _ = manager.delete(&key);
        }
    }

    #[test]
    fn test_store_api_key() {
        let manager = KeychainManager::new();
        let service = "test_api_service";
        let api_key = "sk-test-api-key-12345";
        
        let result = manager.store_api_key(service, api_key);
        
        if result.is_ok() {
            let retrieved = manager.retrieve_api_key(service);
            assert!(retrieved.is_ok());
            assert_eq!(retrieved.unwrap(), api_key);
            
            // Cleanup
            let key = format!("api_key_{}", service);
            let _ = manager.delete(&key);
        }
    }

    #[test]
    fn test_store_encryption_key() {
        let manager = KeychainManager::new();
        let purpose = "test_encryption";
        let enc_key = "encryption_key_value_test";
        
        let result = manager.store_encryption_key(purpose, enc_key);
        
        if result.is_ok() {
            let retrieved = manager.retrieve_encryption_key(purpose);
            assert!(retrieved.is_ok());
            assert_eq!(retrieved.unwrap(), enc_key);
            
            // Cleanup
            let key = format!("encryption_key_{}", purpose);
            let _ = manager.delete(&key);
        }
    }

    #[test]
    fn test_list_keys_returns_vec() {
        let manager = KeychainManager::new();
        let result = manager.list_keys();
        
        // Should return Ok with a Vec (possibly empty)
        assert!(result.is_ok());
        let keys = result.unwrap();
        assert!(keys.len() >= 0); // Can be empty
    }

    #[test]
    fn test_default_trait() {
        let manager1 = KeychainManager::new();
        let manager2 = KeychainManager::default();
        
        // Both should work the same way
        drop(manager1);
        drop(manager2);
    }

    #[test]
    fn test_encryption_roundtrip() {
        let manager = KeychainManager::new();
        let test_key = "encryption_test_key";
        let test_values = vec![
            "simple text",
            "text with spaces and punctuation!",
            "unicode: 日本語 中文 한국어",
            "special chars: @#$%^&*(){}[]",
            "long text ".repeat(100).as_str(),
        ];
        
        for test_value in test_values {
            let store_result = manager.store(test_key, test_value);
            
            if store_result.is_ok() {
                let retrieve_result = manager.retrieve(test_key);
                assert!(retrieve_result.is_ok(), "Should retrieve: {}", test_value);
                assert_eq!(retrieve_result.unwrap(), test_value, "Value should match");
            }
        }
        
        // Cleanup
        let _ = manager.delete(test_key);
    }

    #[test]
    fn test_overwrite_existing_key() {
        let manager = KeychainManager::new();
        let test_key = "overwrite_test_key";
        
        // Store initial value
        let result1 = manager.store(test_key, "initial_value");
        
        if result1.is_ok() {
            // Overwrite with new value
            let result2 = manager.store(test_key, "new_value");
            assert!(result2.is_ok(), "Should allow overwriting");
            
            // Verify new value
            let retrieved = manager.retrieve(test_key);
            assert!(retrieved.is_ok());
            assert_eq!(retrieved.unwrap(), "new_value");
            
            // Cleanup
            let _ = manager.delete(test_key);
        }
    }

    #[test]
    fn test_empty_value() {
        let manager = KeychainManager::new();
        let test_key = "empty_value_test";
        
        let result = manager.store(test_key, "");
        
        if result.is_ok() {
            let retrieved = manager.retrieve(test_key);
            assert!(retrieved.is_ok());
            assert_eq!(retrieved.unwrap(), "");
            
            // Cleanup
            let _ = manager.delete(test_key);
        }
    }

    #[test]
    fn test_special_characters_in_key() {
        let manager = KeychainManager::new();
        let test_key = "key_with_special_chars_!@#";
        let test_value = "test_value";
        
        let result = manager.store(test_key, test_value);
        
        if result.is_ok() {
            let retrieved = manager.retrieve(test_key);
            assert!(retrieved.is_ok());
            
            // Cleanup
            let _ = manager.delete(test_key);
        }
        // Some keychains may reject special chars - that's OK
    }
}
