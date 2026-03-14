#!/bin/bash
# =============================================================================
# Sotopia EC2 Deployment Script
# =============================================================================
# Deploys Sotopia to a single EC2 instance with:
# - Backend (FastAPI) on port 8800
# - Frontend (Next.js) on port 3000
# - Redis on port 6379
# - Nginx reverse proxy on port 80/443
#
# Prerequisites on EC2:
# - Ubuntu 22.04 LTS (recommended)
# - Security group with ports 22, 80, 443, 8800 open
# - SSH access configured
#
# Usage:
#   1. SSH into your EC2 instance
#   2. Clone the repository
#   3. Run: ./scripts/ec2/deploy_ec2.sh
#
# Or from local machine:
#   scp scripts/ec2/deploy_ec2.sh ubuntu@YOUR_EC2_IP:~
#   ssh ubuntu@YOUR_EC2_IP './deploy_ec2.sh'
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Sotopia EC2 Deployment${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration - UPDATE THESE
DOMAIN="${DOMAIN:-}"  # Set your domain or leave empty for IP access
USE_SSL="${USE_SSL:-false}"  # Set to true if you have a domain and want HTTPS
DEPLOY_HOST="${DEPLOY_HOST:-}"  # e.g. 54.89.222.156 - used when EC2 metadata unavailable

# Auto-detect EC2 public IP if domain not set
if [ -z "$DOMAIN" ]; then
    PUBLIC_IP=$(curl -s --connect-timeout 2 http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || true)
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP=$(curl -s --connect-timeout 2 https://ifconfig.me 2>/dev/null || curl -s --connect-timeout 2 https://icanhazip.com 2>/dev/null || true)
    fi
    if [ -z "$PUBLIC_IP" ]; then
        PUBLIC_IP="${DEPLOY_HOST:-}"
    fi
    echo -e "${YELLOW}No domain set, using IP: ${PUBLIC_IP:-<same-origin>}${NC}"
fi

# =============================================================================
# System Setup
# =============================================================================

install_system_deps() {
    echo -e "\n${YELLOW}Installing system dependencies...${NC}"
    
    sudo apt-get update
    sudo apt-get install -y \
        curl \
        git \
        build-essential \
        nginx \
        certbot \
        python3-certbot-nginx \
        docker.io \
        docker-compose
    
    # Add current user to docker group
    sudo usermod -aG docker $USER
    
    echo -e "${GREEN}✓ System dependencies installed${NC}"
}

install_python() {
    echo -e "\n${YELLOW}Installing Python 3.11...${NC}"
    
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip
    
    # Install uv for faster package management (installs to ~/.local/bin)
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    
    echo -e "${GREEN}✓ Python 3.11 installed${NC}"
}

install_node() {
    echo -e "\n${YELLOW}Installing Node.js 20...${NC}"
    
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
    
    # Install pnpm
    sudo npm install -g pnpm
    
    echo -e "${GREEN}✓ Node.js 20 installed${NC}"
}

# =============================================================================
# Redis Setup
# =============================================================================

start_redis() {
    echo -e "\n${YELLOW}Starting Redis...${NC}"
    
    # Stop any existing container
    sudo docker stop sotopia-redis 2>/dev/null || true
    sudo docker rm sotopia-redis 2>/dev/null || true
    
    # Start Redis with persistence
    sudo docker run -d \
        --name sotopia-redis \
        --restart unless-stopped \
        -p 6379:6379 \
        -v redis-data:/data \
        redis/redis-stack-server:latest
    
    echo -e "${GREEN}✓ Redis started${NC}"
}

# =============================================================================
# Application Setup
# =============================================================================

setup_app() {
    echo -e "\n${YELLOW}Setting up application...${NC}"
    
    # Create app directory
    APP_DIR="/opt/sotopia"
    sudo mkdir -p $APP_DIR
    sudo chown $USER:$USER $APP_DIR
    
    # Clone or update repository
    if [ -d "$APP_DIR/.git" ]; then
        echo "Updating existing repository..."
        cd $APP_DIR
        git pull
    elif [ -f "$APP_DIR/pyproject.toml" ]; then
        # App already present (e.g. rsync'd); skip copy
        echo "Application already at $APP_DIR, skipping clone"
        cd $APP_DIR
    else
        echo "Cloning repository..."
        # If running from within the repo (elsewhere), copy files
        if [ -f "pyproject.toml" ] && [ "$(pwd -P)" != "$(cd $APP_DIR 2>/dev/null && pwd -P)" ]; then
            cp -r . $APP_DIR/
        else
            echo -e "${RED}Error: No app at $APP_DIR. Run from repo root or rsync first.${NC}"
            exit 1
        fi
        cd $APP_DIR
    fi
    
    cd $APP_DIR
    echo -e "${GREEN}✓ Application files ready${NC}"
}

create_env_file() {
    echo -e "\n${YELLOW}Creating environment configuration...${NC}"
    
    APP_DIR="/opt/sotopia"
    
    # Prompt for required values if not set
    if [ -z "$OPENAI_API_KEY" ]; then
        echo -e "${YELLOW}Enter your OpenAI API key:${NC}"
        read -r OPENAI_API_KEY
    fi
    
    if [ -z "$JWT_SECRET" ]; then
        # Generate a secure random secret
        JWT_SECRET=$(openssl rand -base64 32)
        echo -e "${GREEN}Generated JWT secret${NC}"
    fi
    
    cat > $APP_DIR/.env << EOF
# Sotopia Production Environment

# Required
OPENAI_API_KEY=$OPENAI_API_KEY

# Storage
REDIS_OM_URL=redis://localhost:6379
SOTOPIA_STORAGE_BACKEND=redis

# Security
JWT_SECRET=$JWT_SECRET

# OAuth (optional - uncomment and fill if using)
# GOOGLE_CLIENT_ID=
# GOOGLE_CLIENT_SECRET=
# GITHUB_CLIENT_ID=
# GITHUB_CLIENT_SECRET=
# DISCORD_CLIENT_ID=
# DISCORD_CLIENT_SECRET=
EOF

    chmod 600 $APP_DIR/.env
    echo -e "${GREEN}✓ Environment file created${NC}"
}

install_python_deps() {
    echo -e "\n${YELLOW}Installing Python dependencies...${NC}"
    
    cd /opt/sotopia
    
    # Create virtual environment
    python3.11 -m venv .venv
    source .venv/bin/activate
    
    # Install dependencies
    pip install --upgrade pip
    pip install -e ".[test]"
    
    echo -e "${GREEN}✓ Python dependencies installed${NC}"
}

build_frontend() {
    echo -e "\n${YELLOW}Building frontend...${NC}"
    
    cd /opt/sotopia/sotopia-chat/frontend
    
    # Clear stale build cache (prevents old UI from persisting)
    rm -rf .next node_modules/.cache 2>/dev/null || true
    
    # Set production API URL (use port 80 via Nginx, not 8800 - security group typically only allows 80)
    if [ -n "$DOMAIN" ]; then
        API_URL="https://$DOMAIN"
        WS_URL="wss://$DOMAIN"
    elif [ -n "$PUBLIC_IP" ]; then
        API_URL="http://$PUBLIC_IP"
        WS_URL="ws://$PUBLIC_IP"
    else
        # Fallback: use same origin for API (relative URLs); WS needs host - use DEPLOY_HOST if set
        API_URL=""
        if [ -n "${DEPLOY_HOST:-}" ]; then
            WS_URL="ws://${DEPLOY_HOST}"
        else
            WS_URL="ws://localhost"
        fi
    fi
    
    cat > .env.local << EOF
NEXT_PUBLIC_API_BASE_URL=$API_URL
NEXT_PUBLIC_WS_BASE=$WS_URL
EOF
    
    # Install and build
    pnpm install
    pnpm build
    
    # Write deploy info for version checking (served at /deploy-info.json)
    DEPLOY_TS=$(date -Iseconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')
    BUILD_ID="${DEPLOY_TS}-$(openssl rand -hex 4 2>/dev/null || echo "local")"
    cat > public/deploy-info.json << DEPLOYEOF
{"deployed_at":"${DEPLOY_TS}","build_id":"${BUILD_ID}"}
DEPLOYEOF
    echo -e "${GREEN}✓ Frontend built (build_id: ${BUILD_ID})${NC}"
}

# =============================================================================
# Systemd Services
# =============================================================================

create_backend_service() {
    echo -e "\n${YELLOW}Creating backend service...${NC}"
    
    sudo tee /etc/systemd/system/sotopia-backend.service > /dev/null << EOF
[Unit]
Description=Sotopia Backend API
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/sotopia
EnvironmentFile=/opt/sotopia/.env
ExecStart=/opt/sotopia/.venv/bin/python -m sotopia.api.fastapi_server
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable sotopia-backend
    sudo systemctl start sotopia-backend
    
    echo -e "${GREEN}✓ Backend service created and started${NC}"
}

create_frontend_service() {
    echo -e "\n${YELLOW}Creating frontend service...${NC}"
    
    sudo tee /etc/systemd/system/sotopia-frontend.service > /dev/null << EOF
[Unit]
Description=Sotopia Frontend
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/sotopia/sotopia-chat/frontend
ExecStart=/usr/bin/pnpm start
Restart=always
RestartSec=5
Environment=NODE_ENV=production
Environment=PORT=3000

[Install]
WantedBy=multi-user.target
EOF
    
    sudo systemctl daemon-reload
    sudo systemctl enable sotopia-frontend
    sudo systemctl start sotopia-frontend
    
    echo -e "${GREEN}✓ Frontend service created and started${NC}"
}

# =============================================================================
# Nginx Configuration
# =============================================================================

configure_nginx() {
    echo -e "\n${YELLOW}Configuring Nginx...${NC}"
    
    if [ -n "$DOMAIN" ]; then
        SERVER_NAME="$DOMAIN"
    elif [ -n "$PUBLIC_IP" ]; then
        SERVER_NAME="$PUBLIC_IP"
    else
        # Fallback when metadata unavailable: _ accepts any hostname
        SERVER_NAME="_"
    fi
    
    sudo tee /etc/nginx/sites-available/sotopia > /dev/null << EOF
# Sotopia Nginx Configuration

upstream backend {
    server 127.0.0.1:8800;
}

upstream frontend {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name $SERVER_NAME;
    
    # Frontend (no-store to prevent stale UI in browser cache)
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
    
    # Backend API
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
        
        # WebSocket support
        proxy_read_timeout 86400;
    }
    
    # Direct backend access (for development/testing)
    location /auth/ {
        proxy_pass http://backend/auth/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location /oauth/ {
        proxy_pass http://backend/oauth/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location /leaderboard {
        proxy_pass http://backend/leaderboard;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location /profile/ {
        proxy_pass http://backend/profile/;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
    
    location /games/ {
        proxy_pass http://backend/games/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
    
    location /ws/ {
        proxy_pass http://backend/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_read_timeout 86400;
    }
    
    location /health {
        proxy_pass http://backend/health;
    }
}
EOF
    
    # Enable site
    sudo ln -sf /etc/nginx/sites-available/sotopia /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default
    
    # Test and reload
    sudo nginx -t
    sudo systemctl reload nginx
    
    echo -e "${GREEN}✓ Nginx configured${NC}"
}

setup_ssl() {
    if [ "$USE_SSL" != "true" ] || [ -z "$DOMAIN" ]; then
        echo -e "${YELLOW}Skipping SSL setup (no domain or USE_SSL not set)${NC}"
        return
    fi
    
    echo -e "\n${YELLOW}Setting up SSL with Let's Encrypt...${NC}"
    
    sudo certbot --nginx -d $DOMAIN --non-interactive --agree-tos -m admin@$DOMAIN
    
    echo -e "${GREEN}✓ SSL configured${NC}"
}

# =============================================================================
# Health Check
# =============================================================================

health_check() {
    echo -e "\n${YELLOW}Running health checks...${NC}"
    
    sleep 5
    
    # Check Redis
    if docker exec sotopia-redis redis-cli ping | grep -q "PONG"; then
        echo -e "${GREEN}✓ Redis: healthy${NC}"
    else
        echo -e "${RED}✗ Redis: not responding${NC}"
    fi
    
    # Check Backend
    if curl -s http://localhost:8800/health | grep -q "ok"; then
        echo -e "${GREEN}✓ Backend: healthy${NC}"
    else
        echo -e "${RED}✗ Backend: not responding${NC}"
        echo -e "${YELLOW}  Check logs: sudo journalctl -u sotopia-backend -f${NC}"
    fi
    
    # Check Frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Frontend: healthy${NC}"
    else
        echo -e "${RED}✗ Frontend: not responding${NC}"
        echo -e "${YELLOW}  Check logs: sudo journalctl -u sotopia-frontend -f${NC}"
    fi
    
    # Check Nginx
    if curl -s http://localhost > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Nginx: healthy${NC}"
    else
        echo -e "${RED}✗ Nginx: not responding${NC}"
    fi
}

# =============================================================================
# Print Summary
# =============================================================================

print_summary() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${GREEN}  Deployment Complete!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    if [ -n "$DOMAIN" ]; then
        if [ "$USE_SSL" == "true" ]; then
            echo -e "  ${GREEN}Website:${NC}      https://$DOMAIN"
        else
            echo -e "  ${GREEN}Website:${NC}      http://$DOMAIN"
        fi
    else
        echo -e "  ${GREEN}Website:${NC}      http://$PUBLIC_IP"
    fi
    
    echo ""
    echo -e "${YELLOW}Pages:${NC}"
    echo -e "  - Home:        /"
    echo -e "  - Login:       /login"
    echo -e "  - Register:    /register"
    echo -e "  - Profile:     /profile"
    echo -e "  - Leaderboard: /leaderboard"
    echo -e "  - Werewolf:    /games/werewolf"
    echo ""
    echo -e "${YELLOW}Management:${NC}"
    echo -e "  View backend logs:   sudo journalctl -u sotopia-backend -f"
    echo -e "  View frontend logs:  sudo journalctl -u sotopia-frontend -f"
    echo -e "  Restart backend:     sudo systemctl restart sotopia-backend"
    echo -e "  Restart frontend:    sudo systemctl restart sotopia-frontend"
    echo -e "  Restart Redis:       sudo docker restart sotopia-redis"
    echo ""
    echo -e "${YELLOW}Update deployment:${NC}"
    echo -e "  cd /opt/sotopia && git pull && ./scripts/ec2/deploy_ec2.sh"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

main() {
    install_system_deps
    install_python
    install_node
    start_redis
    setup_app
    create_env_file
    install_python_deps
    build_frontend
    create_backend_service
    create_frontend_service
    configure_nginx
    setup_ssl
    health_check
    print_summary
}

main "$@"
