#!/bin/bash
# R3MES Quick Test Script

echo "🧪 R3MES Sistem Testi"
echo "===================="
echo ""

# 1. Node Durumu
echo "1️⃣  Blockchain Node:"
NODE_RESPONSE=$(curl -s http://localhost:26657/status 2>/dev/null)
if echo "$NODE_RESPONSE" | grep -q "latest_block_height"; then
    NODE_HEIGHT=$(echo "$NODE_RESPONSE" | grep -o '"latest_block_height":"[^"]*"' | cut -d'"' -f4)
    echo "   ✅ Node çalışıyor (Block Height: $NODE_HEIGHT)"
else
    echo "   ❌ Node çalışmıyor"
    echo "      Response: ${NODE_RESPONSE:0:100}..."
fi

# 2. Dashboard API
echo ""
echo "2️⃣  Dashboard API:"
API_STATUS=$(curl -s http://localhost:1317/api/dashboard/status 2>/dev/null)
if echo "$API_STATUS" | grep -q "block_height"; then
    BLOCK_HEIGHT=$(echo "$API_STATUS" | grep -o '"block_height":[0-9]*' | cut -d':' -f2)
    ACTIVE_MINERS=$(echo "$API_STATUS" | grep -o '"active_miners":[0-9]*' | cut -d':' -f2)
    TOTAL_GRADIENTS=$(echo "$API_STATUS" | grep -o '"total_gradients":[0-9]*' | cut -d':' -f2)
    echo "   ✅ API çalışıyor"
    echo "      - Active Miners: $ACTIVE_MINERS"
    echo "      - Total Gradients: $TOTAL_GRADIENTS"
    echo "      - Block Height: $BLOCK_HEIGHT"
else
    echo "   ❌ API çalışmıyor veya hata dönüyor"
    echo "      Response: ${API_STATUS:0:100}..."
fi

# 3. IPFS
echo ""
echo "3️⃣  IPFS:"
IPFS_PEERS=$(ipfs swarm peers 2>/dev/null | wc -l)
if [ "$IPFS_PEERS" -gt 0 ] 2>/dev/null; then
    echo "   ✅ IPFS çalışıyor ($IPFS_PEERS peers)"
else
    echo "   ⚠️  IPFS daemon çalışmıyor veya peer yok"
fi

# 4. Web Dashboard
echo ""
echo "4️⃣  Web Dashboard:"
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "   ✅ Web Dashboard çalışıyor (http://localhost:3000)"
else
    echo "   ❌ Web Dashboard çalışmıyor"
fi

# 5. Miner Engine
echo ""
echo "5️⃣  Miner Engine:"
if [ -f ~/R3MES/miner-engine/venv/bin/activate ]; then
    echo "   ✅ Miner engine kurulu"
    if command -v r3mes-miner >/dev/null 2>&1; then
        echo "   ✅ CLI komutu mevcut"
    else
        echo "   ⚠️  CLI komutu bulunamadı (venv aktif değil olabilir)"
    fi
else
    echo "   ❌ Miner engine kurulu değil"
fi

echo ""
echo "===================="
echo "✅ Sistem testi tamamlandı!"
echo ""
echo "📋 Sonraki Adımlar:"
echo "   1. Miner'ı başlat: cd ~/R3MES/miner-engine && source venv/bin/activate && r3mes-miner start"
echo "   2. Dashboard'u aç: http://localhost:3000/dashboard"
echo "   3. Detaylı test: END_TO_END_TEST.md dosyasına bakın"

