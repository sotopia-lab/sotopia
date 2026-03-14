#!/bin/bash
# =============================================================================
# Completely wipe Sotopia from EC2
# =============================================================================
# Removes:
#   - Systemd services (sotopia-backend, sotopia-frontend)
#   - Nginx sotopia site config
#   - Redis container (sotopia-redis) and volume
#   - /opt/sotopia directory
#   - /tmp/sotopia_env_backup
#
# Run on EC2: bash scripts/ec2/wipe_ec2.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Wiping all Sotopia components from EC2...${NC}"

# -----------------------------------------------------------------------------
# Stop and remove systemd services
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Stopping systemd services...${NC}"
sudo systemctl stop sotopia-backend 2>/dev/null || true
sudo systemctl stop sotopia-frontend 2>/dev/null || true
sudo systemctl disable sotopia-backend 2>/dev/null || true
sudo systemctl disable sotopia-frontend 2>/dev/null || true
echo -e "${GREEN}✓ Services stopped and disabled${NC}"

echo -e "\n${YELLOW}Removing systemd service files...${NC}"
sudo rm -f /etc/systemd/system/sotopia-backend.service
sudo rm -f /etc/systemd/system/sotopia-frontend.service
sudo systemctl daemon-reload
echo -e "${GREEN}✓ Service files removed${NC}"

# -----------------------------------------------------------------------------
# Nginx
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Removing Nginx sotopia config...${NC}"
sudo rm -f /etc/nginx/sites-enabled/sotopia
# Restore default site so nginx doesn't break
if [ -f /etc/nginx/sites-available/default ]; then
    sudo ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default
fi
sudo nginx -t 2>/dev/null && sudo systemctl reload nginx || echo -e "${YELLOW}Nginx reload skipped (may already be clean)${NC}"
echo -e "${GREEN}✓ Nginx cleaned${NC}"

# -----------------------------------------------------------------------------
# Redis container
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Removing Redis container...${NC}"
sudo docker stop sotopia-redis 2>/dev/null || true
sudo docker rm sotopia-redis 2>/dev/null || true
sudo docker volume rm redis-data 2>/dev/null || true
echo -e "${GREEN}✓ Redis container and volume removed${NC}"

# -----------------------------------------------------------------------------
# Application directory and backups
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Removing /opt/sotopia...${NC}"
sudo rm -rf /opt/sotopia
echo -e "${GREEN}✓ /opt/sotopia removed${NC}"

echo -e "\n${YELLOW}Removing backup files...${NC}"
rm -f /tmp/sotopia_env_backup
echo -e "${GREEN}✓ Backup files removed${NC}"

# -----------------------------------------------------------------------------
# Optional: remove nginx sotopia config from sites-available too
# -----------------------------------------------------------------------------
sudo rm -f /etc/nginx/sites-available/sotopia

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Sotopia completely wiped from EC2${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "Removed:"
echo -e "  - sotopia-backend, sotopia-frontend systemd services"
echo -e "  - sotopia-redis Docker container and volume"
echo -e "  - Nginx sotopia site"
echo -e "  - /opt/sotopia"
echo ""
echo -e "To deploy fresh: rsync code to /opt/sotopia, then run scripts/ec2/deploy_ec2.sh"
echo ""
