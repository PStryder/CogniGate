#!/bin/bash
# Test CogniGate standalone mode
# Usage: ./test_standalone.sh [API_KEY] [BASE_URL]

API_KEY="${1:-cg_test_key_12345}"
BASE_URL="${2:-http://localhost:8000}"

echo "=== Testing CogniGate Standalone Mode ==="
echo "Base URL: $BASE_URL"
echo ""

# Health check
echo "1. Health check:"
curl -s "$BASE_URL/health" | python -m json.tool 2>/dev/null || curl -s "$BASE_URL/health"
echo ""

# Detailed health
echo "2. Detailed health check:"
curl -s "$BASE_URL/health/detailed" | python -m json.tool 2>/dev/null || curl -s "$BASE_URL/health/detailed"
echo ""

# List profiles
echo "3. List profiles:"
curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/config/profiles" | python -m json.tool 2>/dev/null || curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/config/profiles"
echo ""

# Submit job synchronously
echo "4. Execute job synchronously:"
RECEIPT=$(curl -s -X POST "$BASE_URL/v1/jobs/execute" \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "test-001",
    "payload": {
      "instruction": "Say hello and describe your purpose in one sentence",
      "context": "You are being tested in standalone mode"
    },
    "profile": "default"
  }')

echo "$RECEIPT" | python -m json.tool 2>/dev/null || echo "$RECEIPT"
LEASE_ID=$(echo "$RECEIPT" | python -c "import sys,json; print(json.load(sys.stdin).get('lease_id',''))" 2>/dev/null)
echo ""

if [ -n "$LEASE_ID" ]; then
  # Get receipt
  echo "5. Get receipt by lease_id ($LEASE_ID):"
  curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/receipts/$LEASE_ID" | python -m json.tool 2>/dev/null || curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/receipts/$LEASE_ID"
  echo ""
fi

# List receipts
echo "6. List all receipts:"
curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/receipts?limit=10" | python -m json.tool 2>/dev/null || curl -s -H "X-API-Key: $API_KEY" "$BASE_URL/v1/receipts?limit=10"
echo ""

echo "=== Test complete ==="
