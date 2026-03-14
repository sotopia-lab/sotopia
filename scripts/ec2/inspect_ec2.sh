#!/bin/bash
# =============================================================================
# Inspect EC2 deployment - run ON EC2 to diagnose version/code issues
# Usage: bash scripts/ec2/inspect_ec2.sh
# =============================================================================

echo "=== EC2 Sotopia Inspection ==="
echo ""

echo "1. Check page.tsx - should have AuthHeader and Guest Mode (not YOUR ARENA IDENTITY):"
echo "---"
grep -n "AuthHeader\|Guest Mode\|YOUR ARENA IDENTITY\|Save Identity" /opt/sotopia/sotopia-chat/frontend/app/page.tsx 2>/dev/null | head -20
echo ""

echo "2. Search for 'YOUR ARENA IDENTITY' in frontend (should find nothing in current code):"
echo "---"
grep -r "YOUR ARENA IDENTITY" /opt/sotopia/sotopia-chat/frontend/src /opt/sotopia/sotopia-chat/frontend/app 2>/dev/null || echo "(none found)"
echo ""

echo "3. Frontend .env.local contents:"
echo "---"
cat /opt/sotopia/sotopia-chat/frontend/.env.local 2>/dev/null || echo "(file not found)"
echo ""

echo "4. Nginx games routing - /games/queue should proxy to backend:"
echo "---"
grep -A2 "location /games/queue" /etc/nginx/sites-available/sotopia 2>/dev/null || echo "(check /etc/nginx/sites-enabled/sotopia)"
echo ""

echo "5. Backend /games/queue reachable?"
echo "---"
curl -s -o /dev/null -w "%{http_code}" http://localhost:8800/games/queue && echo " (200=OK)" || echo " (failed)"
echo ""

echo "6. File modification times (when was code last updated?):"
echo "---"
ls -la /opt/sotopia/sotopia-chat/frontend/app/page.tsx 2>/dev/null
ls -la /opt/sotopia/sotopia-chat/frontend/.next/BUILD_ID 2>/dev/null
echo ""

echo "7. First 30 lines of page.tsx:"
echo "---"
head -30 /opt/sotopia/sotopia-chat/frontend/app/page.tsx 2>/dev/null
echo ""

echo "8. What does the server actually return? (curl localhost:3000 and grep):"
echo "---"
HTML=$(curl -s http://localhost:3000/ 2>/dev/null || echo "")
if echo "$HTML" | grep -q "Guest Mode"; then
    echo "  ✓ HTML contains 'Guest Mode' (correct UI)"
elif echo "$HTML" | grep -q "YOUR ARENA IDENTITY"; then
    echo "  ✗ HTML contains 'YOUR ARENA IDENTITY' (OLD UI - EC2 has stale build)"
else
    echo "  (neither found in HTML - might be client-rendered; check _next chunks)"
fi
echo ""
echo "=== Done ==="
