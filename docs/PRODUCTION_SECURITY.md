# Production Security Configuration

Bu dokümantasyon, R3MES projesinin production ortamında güvenli bir şekilde çalışması için gereken güvenlik kontrollerini açıklar.

## 🔒 Güvenlik Kontrolleri

### 1. R3MES_TEST_MODE Environment Variable

**Kritik**: Production ortamında `R3MES_TEST_MODE` environment variable'ı **SET EDİLMEMELİ**.

**Neden**: Test modu, güvenlik kontrollerini bypass eder ve fail-closed mekanizmalarını devre dışı bırakır.

**Kontrol**: Keeper initialization sırasında otomatik kontrol edilir. Eğer production modunda test mode aktifse, uygulama başlatılamaz.

**Test Modu**: Sadece development/testing ortamlarında kullanılmalıdır.

```bash
# ❌ PRODUCTION'DA YAPILMAMALI
export R3MES_TEST_MODE=true

# ✅ PRODUCTION'DA
# R3MES_TEST_MODE set edilmemeli veya unset edilmeli
unset R3MES_TEST_MODE
```

### 2. IPFS Manager Configuration

**Kritik**: Production ortamında IPFS manager **MUTLAKA** yapılandırılmalı.

**Neden**: Dataset verification için IPFS manager gereklidir. IPFS manager yoksa dataset verification yapılamaz ve güvenlik açığı oluşur.

**Yapılandırma**: IPFS API URL'i şu yollarla belirlenebilir:

1. **Environment Variable** (Öncelikli):
   ```bash
   export IPFS_API_URL=http://127.0.0.1:5001
   ```

2. **Module Config** (app_config.go):
   ```go
   {
       Name: remesmoduletypes.ModuleName,
       Config: appconfig.WrapAny(&remesmoduletypes.Module{
           IpfsApiUrl: "http://127.0.0.1:5001",
       }),
   }
   ```

3. **Default** (Fallback):
   - Default: `http://127.0.0.1:5001`

**Kontrol**: Keeper initialization sırasında otomatik kontrol edilir. Eğer production modunda IPFS URL boşsa, uygulama başlatılamaz.

### 3. Fail-Closed Security Model

**Prensip**: Güvenlik kontrolleri başarısız olduğunda işlem **REDDEDİLMELİ**.

**Uygulama**:
- Dataset verification başarısız olursa → `false` döndürülür
- IPFS manager yoksa → Hata fırlatılır
- Global seed alınamazsa → Miner durdurulur (production mode)

**Test Modu**: Test modunda (`R3MES_TEST_MODE=true`) fallback mekanizmaları aktif olabilir.

## 📋 Production Deployment Checklist

### Pre-Deployment

- [ ] `R3MES_TEST_MODE` environment variable **SET EDİLMEMİŞ**
- [ ] `IPFS_API_URL` environment variable **AYARLANMIŞ**
- [ ] IPFS daemon çalışıyor ve erişilebilir
- [ ] IPFS API endpoint doğrulandı (`curl http://127.0.0.1:5001/api/v0/version`)

### Deployment

- [ ] Keeper initialization sırasında security validation geçti
- [ ] IPFS manager başarıyla oluşturuldu
- [ ] Dataset verification test edildi
- [ ] Fail-closed mekanizmaları test edildi

### Post-Deployment

- [ ] Monitoring: IPFS manager health check
- [ ] Monitoring: Dataset verification success rate
- [ ] Alerting: R3MES_TEST_MODE set edilirse uyarı
- [ ] Alerting: IPFS manager unavailable uyarısı

## 🔧 Configuration Examples

### Production Configuration

```bash
# .env file (production)
# R3MES_TEST_MODE set edilmemeli
IPFS_API_URL=http://127.0.0.1:5001
```

### Development/Testing Configuration

```bash
# .env file (development)
R3MES_TEST_MODE=true
IPFS_API_URL=http://127.0.0.1:5001
# veya boş bırakılabilir (test modunda)
```

## ⚠️ Security Warnings

1. **Test Mode in Production**: `R3MES_TEST_MODE=true` production'da **ASLA** kullanılmamalı. Bu, güvenlik kontrollerini bypass eder.

2. **IPFS Manager Missing**: IPFS manager yoksa dataset verification yapılamaz. Bu, güvenlik açığı yaratır.

3. **Fail-Open Behavior**: Güvenlik kontrolleri başarısız olduğunda işlem kabul edilmemeli (fail-closed).

## 📝 Code References

- Security Validation: `remes/x/remes/keeper/security_validation.go`
- Keeper Initialization: `remes/x/remes/keeper/keeper.go:NewKeeper()`
- Dataset Verification: `remes/x/remes/keeper/dataset_governance.go:VerifyDatasetIntegrity()`
- Module Configuration: `remes/x/remes/module/depinject.go:ProvideModule()`

## 🔍 Validation Flow

```
1. App Startup
   ↓
2. Module Initialization (depinject.go)
   ↓
3. Keeper Creation (NewKeeper)
   ↓
4. Security Validation (ValidateProductionSecurity)
   ├─ Check R3MES_TEST_MODE (must not be set in production)
   └─ Check IPFS_API_URL (must be set in production)
   ↓
5. IPFS Manager Creation (if validation passes)
   ↓
6. Keeper Ready
```

## 🚨 Error Messages

### R3MES_TEST_MODE Set in Production

```
SECURITY ERROR: R3MES_TEST_MODE=true is set in production environment.
This is a security risk. Test mode should only be used in development/testing.
Please unset R3MES_TEST_MODE environment variable before running in production.
```

### IPFS Manager Not Configured

```
SECURITY ERROR: IPFS API URL is not configured.
IPFS manager is required for dataset verification in production.
Please set IPFS_API_URL environment variable or configure it in app configuration.
If you are testing, set R3MES_TEST_MODE=true to bypass this check.
```

---

**Last Updated**: 2025-01-XX  
**Status**: ✅ Production Security Controls Implemented

