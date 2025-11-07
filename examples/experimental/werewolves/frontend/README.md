# Werewolves Frontend

This Next.js 14 workspace hosts the modular user interface for Sotopia social
games. It reuses the existing `chat_server.py` / `fastapi_server.py`
infrastructure and adapts the components from `sotopia-chatbot` for the
Werewolf experiment.

## Getting Started

```bash
cd examples/experimental/werewolves/frontend
npm install          # or pnpm install
npm run dev          # launches http://localhost:3000
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` to point at the FastAPI service
from `sotopia-chat/fastapi_server.py`.

## Available Scripts

- `npm run dev` – start the Next.js dev server
- `npm run build` / `npm run start` – production build & server
- `npm run lint` – run ESLint checks
- `npm run type-check` – TypeScript project validation

## Local Flow

1. Accept the consent form.
2. Provide your participant identifier (email or assigned token).
3. Click “Enter waiting room” to poll `/enter_waiting_room/{id}` until matched.
4. Once matched, the chat panel connects to `/connect`, `/get`, `/send`, and
   `/get_lock` using the shared API client.

This scaffold is phase-aware: augment `src/lib/types/game.ts` and fetch richer
session state once the Werewolf backend exposes it.
