#!/bin/bash
# WiSE-FT Multi-Validation Set Test Script

echo "========================================="
echo "WiSE-FT Multi-Validation Set Test"
echo "========================================="
echo ""

# Check if wiseft_sweep.py exists
if [ ! -f "wiseft_sweep.py" ]; then
    echo "❌ Error: wiseft_sweep.py not found!"
    exit 1
fi

echo "✅ wiseft_sweep.py found"
echo ""

# Test syntax
echo "📝 Testing Python syntax..."
python -m py_compile wiseft_sweep.py
if [ $? -eq 0 ]; then
    echo "✅ Syntax check passed"
else
    echo "❌ Syntax error detected!"
    exit 1
fi
echo ""

# Show help
echo "📚 Showing help..."
python wiseft_sweep.py --help | grep -A 5 "val-sets"
echo ""

echo "========================================="
echo "✅ All checks passed!"
echo ""
echo "사용 예시:"
echo ""
echo "python wiseft_sweep.py \\"
echo "    --scratch models/600.pt \\"
echo "    --finetuned models/620.pt \\"
echo "    --data data/wiseft_test.yaml \\"
echo "    --val-sets valid1 valid2 \\"
echo "    --focus-range 0.2 \\"
echo "    --alpha-min 0.0 \\"
echo "    --alpha-max 0.5 \\"
echo "    --enable-tradeoff-viz"
echo ""
echo "주의사항:"
echo "1. data.yaml에서 val 경로를 확인하세요"
echo "2. valid1.txt와 valid2.txt가 같은 디렉토리에 있어야 합니다"
echo "3. 예: data/valid1.txt, data/valid2.txt"
echo ""
echo "========================================="
