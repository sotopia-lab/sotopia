# Quick Start Guide (Conda Users)

For users with the `sotopia-rl` conda environment already set up.

## Prerequisites

- ✅ Conda environment `sotopia-rl` activated
- ✅ Sotopia package installed (`pip install -e .` from repo root)
- ✅ OpenAI API key set (check with `echo $OPENAI_API_KEY`)
- ✅ Node.js 18+ installed

## Step 1: Start Backend

```bash
# Make sure you're in the werewolf directory
cd examples/experimental/werewolves

# Activate conda environment
conda activate sotopia-rl

# Go to backend folder
cd backend/

# Install backend-specific dependencies
pip install fastapi uvicorn[standard] websockets python-dotenv pydantic

# Or install from requirements.txt
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# (Optional) Edit .env if needed - usually defaults are fine
# nano .env

# Start the backend server
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

**Keep this terminal running!**

## Step 2: Start Frontend (New Terminal)

```bash
# Navigate to frontend folder
cd examples/experimental/werewolves/frontend/

# Install Node dependencies (first time only)
npm install

# Copy environment template
cp .env.example .env.local

# The default values should work:
# NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
# NEXT_PUBLIC_WS_URL=ws://localhost:8000

# Start the frontend dev server
npm run dev
```

You should see:
```
  ▲ Next.js 14.1.0
  - Local:        http://localhost:3000
  - Ready in 2.3s
```

## Step 3: Test the Game

1. **Open your browser**: http://localhost:3000

2. **Create a game**:
   - Enter your name (e.g., "Alice")
   - Click "Create Game"

3. **Check connections**:
   - Open browser DevTools (F12) → Console tab
   - You should see: `WebSocket connected`
   - In the backend terminal, you should see:
     ```
     INFO: Client connected to game {game_id}
     ```

4. **Play the game**:
   - Game events will appear in the event feed
   - When it's your turn, the action panel will light up
   - Select an action and submit

## Troubleshooting

### "ModuleNotFoundError: No module named 'sotopia'"

```bash
# Make sure you're in the conda env
conda activate sotopia-rl

# Install sotopia if not already installed
cd /path/to/sotopia/root
pip install -e .
```

### "WebSocket connection failed"

- Check that backend is running on port 8000
- Check browser console for errors
- Make sure `.env.local` has correct URLs

### "CORS error"

- Backend `.env` should have:
  ```
  ALLOWED_ORIGINS=http://localhost:3000,https://*.vercel.app
  ```

### Backend starts but game doesn't progress

- Check that `OPENAI_API_KEY` is set:
  ```bash
  echo $OPENAI_API_KEY
  ```
- If not set, add to `.env`:
  ```bash
  OPENAI_API_KEY=sk-your-key-here
  ```

## What's Happening Under the Hood

1. **Backend (port 8000)**:
   - Wraps your existing werewolf game logic
   - Manages WebSocket connections
   - Runs the game loop in the background
   - Broadcasts events to connected clients

2. **Frontend (port 3000)**:
   - React UI for creating/joining games
   - WebSocket client for real-time updates
   - Action submission form
   - Event feed display

3. **Game Flow**:
   ```
   User creates game
     ↓
   Backend starts game session
     ↓
   Game loop begins (async)
     ↓
   Events broadcast via WebSocket
     ↓
   Frontend receives & displays events
     ↓
   User submits action
     ↓
   Action queued for game loop
     ↓
   Game loop processes action
     ↓
   New events broadcast...
   ```

## Next Steps

- [ ] Test with multiple browser tabs (simulate multiple players)
- [ ] Check backend logs for game progression
- [ ] Try different actions (speak, vote, etc.)
- [ ] Once working locally, deploy to Railway + Vercel (see DEPLOYMENT.md)

## Environment Summary

**Your setup**:
- Python: conda environment `sotopia-rl`
- Node: system Node.js (v18+)
- Backend: FastAPI on port 8000
- Frontend: Next.js on port 3000

**No venv needed** since you're using conda! 🎉
