#!/bin/bash
# Fix Nginx so /games/queue and other backend routes go to the backend, not Next.js
# Run on EC2: bash /opt/sotopia/scripts/ec2/fix_nginx_games.sh

set -e
echo "Fixing Nginx routing for /games/..."

# Backup
sudo cp /etc/nginx/sites-available/sotopia /etc/nginx/sites-available/sotopia.bak

# Ensure backend /games/ routes exist BEFORE the frontend catch-all
# The key: /games/queue, /games/history, /games/matchmaking must hit backend
# location /games/ with proxy to backend must come before any /games/ → frontend

sudo tee /etc/nginx/sites-available/sotopia > /dev/null << 'NGINXEOF'
# Sotopia Nginx - backend /games/* must route to API
upstream backend { server 127.0.0.1:8800; }
upstream frontend { server 127.0.0.1:3000; }

server {
    listen 80;
    server_name _;

    # Backend API routes (MUST come before location /)
    location /api/ {
        rewrite ^/api/(.*) /$1 break;
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
    location /auth/ { proxy_pass http://backend/auth/; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /oauth/ { proxy_pass http://backend/oauth/; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /leaderboard { proxy_pass http://backend/leaderboard; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /profile/ { proxy_pass http://backend/profile/; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /health { proxy_pass http://backend/health; proxy_set_header Host $host; }
    location /ws/ {
        proxy_pass http://backend/ws/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    # Backend /games/* API only (NOT /games/werewolf page which is frontend)
    location /games/queue { proxy_pass http://backend/games/queue; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /games/history/ { proxy_pass http://backend/games/history/; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /games/matchmaking/ { proxy_pass http://backend/games/matchmaking/; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /games/werewolf/config { proxy_pass http://backend/games/werewolf/config; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }
    location /games/werewolf/sessions/ {
        proxy_pass http://backend/games/werewolf/sessions/;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
    location /games/leaderboard { proxy_pass http://backend/games/leaderboard; proxy_set_header Host $host; proxy_set_header X-Real-IP $remote_addr; }

    # Frontend (catch-all: /games/werewolf page, /, etc.)
    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        add_header Cache-Control "no-store, no-cache, must-revalidate";
    }
}
NGINXEOF

sudo nginx -t && sudo systemctl reload nginx
echo "Done. Test: curl -s http://localhost/games/queue"