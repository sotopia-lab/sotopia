# Werewolf Game Testing Guide

## Prerequisites

### Required Software
- **Python 3.9+** (for backend)
- **Node.js 18+** (for Next.js frontend)
- **Redis 6+** (for state management)
- **Git** (for version control)

### Required Accounts
- **OpenAI API key** (for AI agents)
- **Vercel account** (for frontend deployment, free tier works)

---

## Local Development Setup

### Step 1: Backend Setup

```bash
# Navigate to backend directory
cd sotopia-backend  # or wherever your backend is

# Install Python dependencies
pip install -r requirements.txt

# Start Redis (choose one method)
# Option A: Docker
docker run -d -p 6379:6379 redis:latest

# Option B: Homebrew (Mac)
brew services start redis

# Option C: WSL/Linux
sudo service redis-server start

# Verify Redis is running
redis-cli ping  # Should return "PONG"

# Create .env file
cat > .env << EOF
REDIS_OM_URL=redis://localhost:6379
OPENAI_API_KEY=sk-your-key-here
FASTAPI_URL=http://localhost:8000
EOF

# Start FastAPI server
uvicorn fastapi_server:app --reload --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process [12345]
INFO:     Started server process [12346]
INFO:     Waiting for application startup.
```

### Step 2: Frontend Setup

```bash
# Navigate to frontend directory
cd sotopia-frontend  # or your Next.js project

# Install dependencies
npm install

# Create .env.local
cat > .env.local << EOF
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
EOF

# Start development server
npm run dev
```

**Expected Output:**
```
ready - started server on 0.0.0.0:3000, url: http://localhost:3000
```

---

## Testing Checklist

### Test 1: Backend Health Check

```bash
# Test FastAPI is running
curl http://localhost:8000/docs

# Should open Swagger UI in browser
```

### Test 2: Create a Game

```bash
# Via curl
curl -X POST http://localhost:8000/games/werewolf/create \
  -H "Content-Type: application/json" \
  -d '{"host_id": "test-player", "num_ai_players": 5}'

# Expected response:
# {
#   "session_id": "abc123...",
#   "status": "starting",
#   "message": "Game server launching..."
# }
```

### Test 3: Check Game State

```bash
# Replace SESSION_ID with actual value from previous step
curl http://localhost:8000/games/werewolf/sessions/SESSION_ID

# Expected response: Full WerewolfSessionState JSON
```

### Test 4: Frontend Flow

1. **Open Browser:** Navigate to `http://localhost:3000/werewolf`
2. **Accept Consent:** Click "I agree and wish to continue"
3. **Enter Player ID:** Type any identifier (e.g., "alice")
4. **Create Game:** Click "Start New Game"
5. **Wait for Load:** Game board should appear within 5-10 seconds
6. **Verify UI:**
   - Left sidebar shows 6 players
   - Center shows game log
   - Right sidebar shows current phase
   - Action panel appears when it's your turn

### Test 5: Submit Action

1. **Wait for your turn** (green "Your Turn" panel appears)
2. **Select action type** (e.g., "speak")
3. **Enter text** in the input field
4. **Click "Submit Action"**
5. **Verify:** Action appears in game log within 2 seconds

### Test 6: Full Game Playthrough

```bash
# Terminal 1: Watch backend logs
tail -f logs/*.log

# Terminal 2: Watch Redis state changes
redis-cli MONITOR | grep werewolf

# Browser: Play through complete game cycle
# - Night phases (werewolves, seer, witch)
# - Day discussion
# - Voting
# - Check game over screen appears
```

---

## Common Issues & Solutions

### Issue 1: "Failed to create game"
**Cause:** werewolf_server.py not found
**Solution:**
```bash
# Verify file exists
ls examples/experimental/werewolves/backend/

# Check WEREWOLF_SERVER_PATH in fastapi_server.py
# Should point to: ./werewolf_server.py or full path
```

### Issue 2: "Connection Error" in frontend
**Cause:** CORS or backend not running
**Solution:**
```python
# In fastapi_server.py, verify CORS middleware:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add this!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Issue 3: "Game state not found"
**Cause:** Game server crashed or Redis cleared
**Solution:**
```bash
# Check if subprocess is running
ps aux | grep werewolf_server

# Check Redis has data
redis-cli KEYS "werewolf:*"

# Restart game server manually:
python werewolf_server.py run-werewolf-game SESSION_ID PLAYER_ID 5
```

### Issue 4: Actions not registering
**Cause:** Action key mismatch or expired
**Solution:**
```bash
# Check Redis for pending actions
redis-cli GET "werewolf:session:SESSION_ID:action:PLAYER_ID"

# Increase expiry time in fastapi_server.py:
await r.set(action_key, ..., ex=120)  # 2 minutes instead of 60s
```

---

## Deployment Guide

### Backend Deployment (Railway.app - Recommended)

1. **Create Railway Account:** https://railway.app
2. **New Project:**
   ```bash
   railway init
   railway add redis  # Adds managed Redis
   ```
3. **Set Environment Variables:**
   ```bash
   railway variables set OPENAI_API_KEY=sk-...
   railway variables set FASTAPI_URL=https://your-app.railway.app
   ```
4. **Deploy:**
   ```bash
   railway up
   ```
5. **Get URL:** Copy from Railway dashboard (e.g., `https://sotopia-backend-production.up.railway.app`)

### Frontend Deployment (Vercel)

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "Add werewolf game"
   git push origin main
   ```

2. **Connect to Vercel:**
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Set environment variable:
     ```
     NEXT_PUBLIC_API_BASE_URL=https://your-railway-app.railway.app
     ```
   - Deploy

3. **Test Production:**
   - Navigate to `https://your-app.vercel.app/werewolf`
   - Create a game
   - Verify it connects to Railway backend

---

## Performance Testing

### Load Test: Multiple Games

```bash
# Create 5 concurrent games
for i in {1..5}; do
  curl -X POST http://localhost:8000/games/werewolf/create \
    -H "Content-Type: application/json" \
    -d "{\"host_id\": \"player-$i\", \"num_ai_players\": 5}" &
done

# Monitor resource usage
htop  # or top on Mac
```

**Expected:** Each game should start within 10 seconds, Redis memory < 100MB.

### Stress Test: Fast Actions

```bash
# Submit 100 actions rapidly
SESSION_ID="your-session-id"
for i in {1..100}; do
  curl -X POST "http://localhost:8000/games/werewolf/sessions/$SESSION_ID/actions?participant_id=test" \
    -H "Content-Type: application/json" \
    -d '{"action_type": "speak", "argument": "Test '$i'"}' &
done
```

**Expected:** All actions should process without errors (may be rate-limited by OpenAI API).

---

## Debugging Tools

### Redis Inspector
```bash
# View all werewolf keys
redis-cli KEYS "werewolf:*"

# Get game state
redis-cli GET "werewolf:session:SESSION_ID:state" | jq

# Watch real-time updates
redis-cli --csv PSUBSCRIBE "werewolf:*"
```

### FastAPI Debug Logs
```python
# Add to fastapi_server.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Next.js Network Inspector
- Open browser DevTools (F12)
- Go to Network tab
- Filter by "werewolf"
- Check request/response payloads

---

## Success Criteria

✅ Backend starts without errors
✅ Frontend loads at localhost:3000/werewolf
✅ Can create a game via UI
✅ Game state updates every 2 seconds
✅ Can submit actions (speak, vote, etc.)
✅ AI agents respond within 5 seconds
✅ Game completes after ~20-30 turns
✅ Game over screen displays correctly
✅ Can start a new game after completion

---

## Next Steps

1. **Add Logging:** Integrate Sentry or LogRocket for error tracking
2. **Optimize Polling:** Replace REST polling with WebSockets for real-time updates
3. **Multiplayer Matchmaking:** Extend `/games/werewolf/create` to support multiple humans
4. **Role Balancing:** Analyze win rates and adjust AI agent prompts
5. **Replay System:** Save `phase_log` to database for post-game analysis

---

## Support

If you encounter issues:
1. Check Redis is running: `redis-cli ping`
2. Verify OpenAI API key: `curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"`
3. Review backend logs: `tail -f logs/*.log`
4. Test endpoints via Swagger UI: `http://localhost:8000/docs`

For deployment-specific issues:
- Railway: Check build logs in dashboard
- Vercel: View function logs in deployment panel
