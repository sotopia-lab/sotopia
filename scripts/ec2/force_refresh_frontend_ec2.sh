#!/bin/bash
# =============================================================================
# Force a clean frontend rebuild on EC2 - run ON EC2 when browser cache bypass
# doesn't fix old UI ("YOUR ARENA IDENTITY" instead of "Guest Mode")
# =============================================================================

set -e
cd /opt/sotopia/sotopia-chat/frontend

echo "=== Force refresh frontend (clean build) ==="

# Verify we have the correct page.tsx (Guest Mode, not Your Arena Identity)
if ! grep -q "Guest Mode" app/page.tsx 2>/dev/null; then
    echo "ERROR: app/page.tsx does not contain 'Guest Mode'. You have the OLD code."
    echo "Rsync from your worktree FIRST (with the auth UI changes) before building."
    echo "  From Mac: rsync -avz --exclude .venv --exclude node_modules --exclude __pycache__ --exclude .next -e 'ssh -i sotopia-web-ssh.pem' ./ ubuntu@YOUR_EC2_IP:/opt/sotopia/"
    exit 1
fi

# 1. Remove Next.js build cache - forces new chunk filenames
echo "Removing .next and build cache..."
rm -rf .next
rm -rf node_modules/.cache 2>/dev/null || true

# 2. Ensure .env.local has correct API URL
DEPLOY_HOST="${DEPLOY_HOST:-54.89.222.156}"
if [ -n "$DEPLOY_HOST" ]; then
    cat > .env.local << EOF
NEXT_PUBLIC_API_BASE_URL=http://${DEPLOY_HOST}
NEXT_PUBLIC_WS_BASE=ws://${DEPLOY_HOST}
EOF
    echo "Updated .env.local for $DEPLOY_HOST"
fi

# 3. Rebuild
echo "Building..."
pnpm build

# 4. Restart frontend
echo "Restarting frontend service..."
sudo systemctl restart sotopia-frontend
sleep 2

echo ""
echo "Done. Visit http://${DEPLOY_HOST:-54.89.222.156} in a NEW incognito window."
echo "If still old: run 'curl -s http://localhost:3000/ | grep -E \"Guest Mode|YOUR ARENA\"' to see what server returns."
