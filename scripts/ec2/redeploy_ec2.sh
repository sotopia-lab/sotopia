#!/bin/bash
# =============================================================================
# Clean Redeploy to EC2
# =============================================================================
# How it works on EC2:
#   1. Code lives at /opt/sotopia (Python backend + Next.js frontend)
#   2. Backend: systemd runs uvicorn from /opt/sotopia on port 8800
#   3. Frontend: systemd runs "pnpm start" (Next.js) on port 3000
#   4. Nginx: reverse proxy on 80, serves frontend and proxies /games/, /ws/, etc to backend
#   5. Redis: Docker container on 6379
#
# Clean redeploy steps:
#   A. On EC2: backup .env and wipe /opt/sotopia
#   B. From local: rsync code to EC2
#   C. On EC2: run this script
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

DEPLOY_HOST="${DEPLOY_HOST:-54.89.222.156}"

# -----------------------------------------------------------------------------
# Check we have code
# -----------------------------------------------------------------------------
if [ ! -f /opt/sotopia/pyproject.toml ]; then
    echo -e "${RED}Error: No code at /opt/sotopia. Run steps A and B first:${NC}"
    echo ""
    echo -e "${YELLOW}Step A (on EC2):${NC}"
    echo "  cp /opt/sotopia/.env /tmp/sotopia_env_backup 2>/dev/null || true"
    echo "  sudo rm -rf /opt/sotopia/{*,.[!.]*}"
    echo ""
    echo -e "${YELLOW}Step B (from your Mac, in project dir):${NC}"
    echo '  rsync -avz --exclude .venv --exclude node_modules --exclude __pycache__ --exclude .next \'
    echo '    -e "ssh -i sotopia-web-ssh.pem" \'
    echo '    ./ ubuntu@ec2-54-89-222-156.compute-1.amazonaws.com:/opt/sotopia/'
    echo ""
    exit 1
fi

echo -e "${GREEN}Clean redeploy (host: $DEPLOY_HOST)${NC}"

# Restore .env
if [ -f /tmp/sotopia_env_backup ]; then
    cp /tmp/sotopia_env_backup /opt/sotopia/.env
    chmod 600 /opt/sotopia/.env
    echo -e "${GREEN}Restored .env${NC}"
else
    echo -e "${RED}No /tmp/sotopia_env_backup. Create .env with OPENAI_API_KEY, JWT_SECRET, REDIS_OM_URL${NC}"
    exit 1
fi

# -----------------------------------------------------------------------------
# Python deps
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Installing Python deps...${NC}"
cd /opt/sotopia
rm -rf .venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
pip install -e . -q
pip install "fastapi[standard]" uvicorn -q
echo -e "${GREEN}✓ Python deps installed${NC}"

# -----------------------------------------------------------------------------
# Frontend (clean build - no stale chunks)
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Building frontend...${NC}"
cd /opt/sotopia/sotopia-chat/frontend
rm -rf .next node_modules/.cache 2>/dev/null || true
cat > .env.local << EOF
NEXT_PUBLIC_API_BASE_URL=http://${DEPLOY_HOST}
NEXT_PUBLIC_WS_BASE=ws://${DEPLOY_HOST}
EOF
pnpm install
pnpm build
DEPLOY_TS=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')
BUILD_ID="${DEPLOY_TS}-$(openssl rand -hex 4 2>/dev/null || echo "local")"
echo "{\"deployed_at\":\"${DEPLOY_TS}\",\"build_id\":\"${BUILD_ID}\"}" > public/deploy-info.json
echo -e "${GREEN}✓ Frontend built (build_id: ${BUILD_ID})${NC}"

# -----------------------------------------------------------------------------
# Nginx - update routing so /games/werewolf (page) goes to frontend
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Updating Nginx routing...${NC}"
SERVER_NAME="${DEPLOY_HOST:-54.89.222.156}"
sudo tee /etc/nginx/sites-available/sotopia > /dev/null << NGINXEOF
# Sotopia Nginx Configuration
upstream backend { server 127.0.0.1:8800; }
upstream frontend { server 127.0.0.1:3000; }

server {
    listen 80;
    server_name $SERVER_NAME;
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }
    location /api/ {
        rewrite ^/api/(.*) /\$1 break;
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        proxy_read_timeout 86400;
    }
    location /auth/ { proxy_pass http://backend/auth/; proxy_set_header Host \$host; proxy_set_header X-Real-IP \$remote_addr; }
    location /oauth/ { proxy_pass http://backend/oauth/; proxy_set_header Host \$host; proxy_set_header X-Real-IP \$remote_addr; }
    location /leaderboard { proxy_pass http://backend/leaderboard; proxy_set_header Host \$host; proxy_set_header X-Real-IP \$remote_addr; }
    location /profile/ { proxy_pass http://backend/profile/; proxy_set_header Host \$host; proxy_set_header X-Real-IP \$remote_addr; }
    location /games/werewolf/sessions/ {
        proxy_pass http://backend/games/werewolf/sessions/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
    location /games/werewolf/config {
        proxy_pass http://backend/games/werewolf/config;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /games/queue {
        proxy_pass http://backend/games/queue;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /games/history/ {
        proxy_pass http://backend/games/history/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /games/leaderboard {
        proxy_pass http://backend/games/leaderboard;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    location /games/ {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }
    location /ws/ {
        proxy_pass http://backend/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
    location /health { proxy_pass http://backend/health; }
}
NGINXEOF
sudo nginx -t && sudo systemctl reload nginx
echo -e "${GREEN}✓ Nginx updated${NC}"

# -----------------------------------------------------------------------------
# Restart
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}Restarting services...${NC}"
sudo systemctl restart sotopia-backend
sudo systemctl restart sotopia-frontend
sleep 2

echo ""
echo -e "${GREEN}Done.${NC}"
echo -e "Backend:  $(systemctl is-active sotopia-backend)"
echo -e "Frontend: $(systemctl is-active sotopia-frontend)"
echo -e "Visit: http://${DEPLOY_HOST}"
echo ""
