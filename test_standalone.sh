#!/bin/bash
# Test CogniGate standalone mode (MCP)
# Usage: ./test_standalone.sh [API_KEY] [BASE_URL]

API_KEY="${1:-cg_test_key_12345}"
BASE_URL="${2:-http://localhost:8000}"

rpc() {
  local payload="$1"
  curl -s "$BASE_URL/mcp"     -H "X-API-Key: $API_KEY"     -H "Content-Type: application/json"     -d "$payload"
}

echo "=== Testing CogniGate Standalone Mode ==="
echo "Base URL: $BASE_URL"
echo ""

# Health check
echo "1. Health check:"
rpc '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"cognigate.health","arguments":{}}}'   | python -m json.tool 2>/dev/null || true
echo ""

# Detailed health
echo "2. Detailed health check:"
rpc '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"cognigate.health_detailed","arguments":{}}}'   | python -m json.tool 2>/dev/null || true
echo ""

# List profiles
echo "3. List profiles:"
rpc '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"cognigate.list_profiles","arguments":{}}}'   | python -m json.tool 2>/dev/null || true
echo ""

# Execute job synchronously
echo "4. Execute job synchronously:"
RECEIPT=$(rpc '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"cognigate.execute_job","arguments":{"task_id":"test-001","payload":{"instruction":"Say hello and describe your purpose in one sentence","context":"You are being tested in standalone mode"},"profile":"default"}}}')

echo "$RECEIPT" | python -m json.tool 2>/dev/null || echo "$RECEIPT"
LEASE_ID=$(echo "$RECEIPT" | python -c "import sys,json; print((json.load(sys.stdin).get('result') or {}).get('lease_id',''))" 2>/dev/null)
echo ""

if [ -n "$LEASE_ID" ]; then
  # Get receipt
  echo "5. Get receipt by lease_id ($LEASE_ID):"
  rpc '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"cognigate.get_receipt","arguments":{"lease_id":"'"$LEASE_ID"'"}}}'     | python -m json.tool 2>/dev/null || true
  echo ""
fi

# List receipts
echo "6. List all receipts:"
rpc '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"cognigate.list_receipts","arguments":{"limit":10}}}'   | python -m json.tool 2>/dev/null || true
echo ""

echo "=== Test complete ==="
