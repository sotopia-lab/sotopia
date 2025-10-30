# Werewolf Web Game (Option A - Hybrid)

Multi-player werewolf game with AI agents and real-time web interface.

## Quick Start

### Run Locally

**Terminal 1 - Backend:**
```bash
# Activate your conda environment
conda activate sotopia-rl

cd backend/
pip install -r requirements.txt
cp .env.example .env
python main.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend/
npm install
cp .env.example .env.local
npm run dev
```

Open http://localhost:3000

## Project Structure

```
werewolves/
├── backend/              # FastAPI backend
│   ├── main.py          # API routes + WebSocket
│   ├── game_manager.py  # Game loop wrapper
│   ├── ws_manager.py    # WebSocket connections
│   ├── models.py        # Pydantic models
│   └── requirements.txt
├── frontend/            # Next.js frontend
│   ├── app/
│   │   ├── page.tsx          # Home (create game)
│   │   └── game/[id]/page.tsx # Game UI
│   └── package.json
├── main_human.py        # Original game logic
├── roster.json          # Player configuration
├── role_actions.json    # Game rules
└── DEPLOYMENT.md        # Full deployment guide
```

## Key Features

- ✅ Real-time WebSocket updates
- ✅ Beautiful UI with Tailwind CSS
- ✅ Single-player with AI agents
- ✅ Mobile-responsive design
- ✅ Production-ready (Railway + Vercel)

## API Endpoints

### HTTP
- `POST /api/game/create` - Create new game
- `GET /api/game/{id}/state` - Get game state
- `POST /api/game/{id}/action` - Submit player action

### WebSocket
- `WS /ws/{id}` - Real-time game events

## Development

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed setup and deployment instructions.

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **Backend**: FastAPI, Python 3.10+
- **Game Logic**: Sotopia framework
- **LLM**: GPT-4o-mini (OpenAI)
- **Deployment**: Vercel (frontend) + Railway (backend)

## Roadmap

- [ ] Multi-player lobby
- [ ] Spectator mode
- [ ] Game replay/history
- [ ] Voice chat integration
- [ ] Mobile app (React Native)
- [ ] Option B migration (all-Vercel stateless)
