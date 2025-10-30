# Werewolf Game Deployment Guide

This guide covers deploying the Werewolf game using Option A (Hybrid architecture).

## Architecture

```
Frontend (Next.js) on Vercel
    ↓ HTTPS/WSS
Backend (FastAPI) on Railway/Render
    ↓
Redis (Upstash) - for future state management
```

## Prerequisites

- Node.js 18+ and npm
- Python 3.10+
- Railway or Render account (backend)
- Vercel account (frontend)
- Upstash account (optional, for Redis)

## Local Development

### 1. Backend Setup

```bash
cd backend/

# If using conda (recommended if you already have sotopia-rl):
conda activate sotopia-rl

# OR if using venv:
# python -m venv venv
# source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install additional backend dependencies
pip install -r requirements.txt

# Note: sotopia package should already be installed in your conda env
# If not, install it:
# cd ../../../.. (go to sotopia root)
# pip install -e .
# cd examples/experimental/werewolves/backend/

# Copy environment file
cp .env.example .env

# Start backend
python main.py
# Backend runs on http://localhost:8000
```

### 2. Frontend Setup

```bash
cd frontend/

# Install dependencies
npm install

# Copy environment file
cp .env.example .env.local

# Update .env.local
# NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
# NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Start frontend
npm run dev
# Frontend runs on http://localhost:3000
```

### 3. Test Locally

1. Open http://localhost:3000
2. Enter your name and create a game
3. You'll be redirected to the game page
4. Check browser console and backend logs for WebSocket connection

## Production Deployment

### Backend (Railway)

1. **Create Railway Project**
   ```bash
   cd backend/
   railway login
   railway init
   ```

2. **Set Environment Variables** (in Railway dashboard)
   ```
   PORT=8000
   ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://*.vercel.app
   OPENAI_API_KEY=sk-...
   REDIS_OM_URL=redis://:@localhost:6379
   ```

3. **Deploy**
   ```bash
   railway up
   ```

4. **Note your Railway URL**: `https://your-backend.up.railway.app`

### Backend (Alternative: Render)

1. Create new Web Service on Render
2. Connect your GitHub repo
3. Set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (same as Railway)
5. Deploy

### Frontend (Vercel)

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Deploy**
   ```bash
   cd frontend/
   vercel
   ```

3. **Set Environment Variables** (in Vercel dashboard or CLI)
   ```bash
   vercel env add NEXT_PUBLIC_BACKEND_URL
   # Enter: https://your-backend.railway.app

   vercel env add NEXT_PUBLIC_WS_URL
   # Enter: wss://your-backend.railway.app
   ```

4. **Redeploy** to apply env vars
   ```bash
   vercel --prod
   ```

### Update Backend CORS

After deploying frontend, update backend's `ALLOWED_ORIGINS`:

```
ALLOWED_ORIGINS=https://your-frontend.vercel.app,https://*.vercel.app
```

## Testing Production

1. Open `https://your-frontend.vercel.app`
2. Create a game
3. Check:
   - Network tab shows WebSocket connection (`wss://...`)
   - Backend logs show WebSocket accepts
   - Game events appear in real-time

## Troubleshooting

### WebSocket Connection Fails

**Issue**: Frontend can't connect to WebSocket

**Solutions**:
- Check `NEXT_PUBLIC_WS_URL` uses `wss://` (not `ws://`)
- Verify backend CORS allows your frontend domain
- Check Railway/Render logs for connection errors

### Backend Import Errors

**Issue**: `ModuleNotFoundError: No module named 'sotopia'`

**Solution**: Backend needs access to parent sotopia package

```bash
# If using conda env sotopia-rl:
conda activate sotopia-rl
# Sotopia should already be installed. If not:
cd /path/to/sotopia (root directory)
pip install -e .

# If using venv:
source venv/bin/activate
pip install -e ../../../..  # Install sotopia from parent directory

# The game_manager.py already adds parent directory to sys.path as a fallback
```

### CORS Errors

**Issue**: `Access to fetch at '...' from origin '...' has been blocked by CORS policy`

**Solution**: Update backend `ALLOWED_ORIGINS` to include frontend URL

### Game Doesn't Start

**Issue**: Backend accepts connections but game doesn't progress

**Solution**: Check backend logs for Python errors. Common issues:
- Missing OpenAI API key
- Redis connection issues (if using)
- Missing roster.json or role_actions.json

## Environment Variables Reference

### Backend (.env)

```bash
# Required
PORT=8000
ALLOWED_ORIGINS=https://your-frontend.vercel.app

# Optional
REDIS_URL=redis://...
OPENAI_API_KEY=sk-...
```

### Frontend (.env.local)

```bash
# Required
NEXT_PUBLIC_BACKEND_URL=https://your-backend.railway.app
NEXT_PUBLIC_WS_URL=wss://your-backend.railway.app
```

## Monitoring

### Railway
- View logs: `railway logs`
- Check metrics in Railway dashboard

### Vercel
- View logs in Vercel dashboard
- Check Analytics for frontend performance

### Health Checks

- Backend health: `https://your-backend.railway.app/health`
- Frontend: Visit homepage

## Scaling Considerations

Currently, the backend is a single long-running process. For higher scale:

1. **Add Redis** for state persistence (allows backend restarts)
2. **Implement Option B** (refactor to stateless handlers)
3. **Use managed WebSocket** service (Pusher, Ably)

## Cost Estimates

- **Vercel**: Free tier (hobby projects)
- **Railway**: ~$5-10/month (after free credits)
- **Render**: $7/month (starter tier)
- **Upstash Redis**: Free tier or ~$0.20/100K commands

## Next Steps

- [ ] Add Redis integration for state persistence
- [ ] Implement proper error boundaries in frontend
- [ ] Add game lobby (multiple players can join)
- [ ] Implement spectator mode
- [ ] Add game history/replay
- [ ] Migrate to Option B (all-Vercel stateless)

## Support

For issues:
1. Check backend logs first
2. Check browser console
3. Verify environment variables
4. Test locally before blaming deployment
