#!/bin/bash
# =============================================================================
# Clear caches on EC2 for a clean rebuild
# Run on EC2: bash scripts/ec2/clear_ec2_caches.sh
# =============================================================================

set -e
cd /opt/sotopia

echo "Clearing caches..."

# Python
rm -rf .venv
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true

# Frontend
rm -rf sotopia-chat/frontend/node_modules
rm -rf sotopia-chat/frontend/.next
rm -rf sotopia-chat/frontend/.turbo 2>/dev/null || true
rm -rf sotopia-chat/frontend/out 2>/dev/null || true

# Pip cache (optional - saves space)
rm -rf ~/.cache/pip 2>/dev/null || true

# Pnpm store (optional - forces fresh install)
# rm -rf ~/.local/share/pnpm/store 2>/dev/null || true

echo "Done. Run scripts/ec2/redeploy_ec2.sh to rebuild."
