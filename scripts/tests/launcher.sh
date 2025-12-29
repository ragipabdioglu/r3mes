#!/bin/bash
# Desktop Launcher Test Script

echo "🖥️  R3MES Desktop Launcher Test"
echo "================================"
echo ""

# 1. Check dependencies
echo "1️⃣  Dependencies:"
cd ~/R3MES/desktop-launcher

if [ -d "node_modules" ]; then
    echo "   ✅ node_modules mevcut"
else
    echo "   ⚠️  node_modules yok, kuruluyor..."
    npm install
fi

if npm list electron >/dev/null 2>&1; then
    ELECTRON_VERSION=$(npm list electron 2>/dev/null | grep electron | head -1 | awk '{print $2}')
    echo "   ✅ Electron kurulu ($ELECTRON_VERSION)"
else
    echo "   ❌ Electron kurulu değil"
    exit 1
fi

# 2. Check web dashboard
echo ""
echo "2️⃣  Web Dashboard:"
if curl -s http://localhost:3000 >/dev/null 2>&1; then
    echo "   ✅ Web dashboard çalışıyor (http://localhost:3000)"
else
    echo "   ⚠️  Web dashboard çalışmıyor"
    echo "      Başlatmak için: cd ~/R3MES/web-dashboard && npm run dev"
fi

# 3. Check binaries
echo ""
echo "3️⃣  Binaries:"
if command -v remesd >/dev/null 2>&1; then
    echo "   ✅ remesd mevcut ($(which remesd))"
else
    echo "   ⚠️  remesd PATH'te değil (workspace path kullanılacak)"
fi

if command -v r3mes-miner >/dev/null 2>&1; then
    echo "   ✅ r3mes-miner mevcut ($(which r3mes-miner))"
else
    echo "   ⚠️  r3mes-miner PATH'te değil (venv aktif edilmeli)"
fi

# 4. Workspace path
echo ""
echo "4️⃣  Workspace Path:"
WORKSPACE="${R3MES_WORKSPACE:-$HOME/R3MES}"
echo "   Workspace: $WORKSPACE"
if [ -d "$WORKSPACE/remes" ]; then
    echo "   ✅ remes dizini mevcut"
else
    echo "   ❌ remes dizini bulunamadı"
fi

# 5. Test instructions
echo ""
echo "================================"
echo "✅ Hazır! Desktop launcher'ı başlatmak için:"
echo ""
echo "   cd ~/R3MES/desktop-launcher"
echo "   npm run dev"
echo ""
echo "📋 Test Adımları:"
echo "   1. Electron penceresi açılmalı"
echo "   2. Web dashboard yüklenmeli"
echo "   3. Menüden 'Start Node' seçin"
echo "   4. Menüden 'Start Miner' seçin"
echo "   5. Dashboard'da sonuçları gözlemleyin"
echo ""

