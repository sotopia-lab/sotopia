# Sotopia Chat Arena Frontend

This Next.js 14 workspace hosts the modular user interface for Sotopia social
games. It reuses the existing `chat_server.py` / `fastapi_server.py`
infrastructure and now exposes a lightweight registry so multiple games can
plug into a shared consent → lobby → experience flow. The Werewolf experience
is implemented in `src/games/werewolf` as the first module.

## Getting Started

```bash
cd sotopia-chat/frontend
pnpm install
pnpm dev            # launches http://localhost:3000
```

Set `NEXT_PUBLIC_API_BASE_URL` in `.env.local` to point at the FastAPI service
from `sotopia-chat/fastapi_server.py`.

## Available Scripts

- `npm run dev` – start the Next.js dev server
- `npm run build` / `npm run start` – production build & server
- `npm run lint` – run ESLint checks
- `npm run type-check` – TypeScript project validation

## Local Flow

1. Visit `/` and select a game from the arena landing page.
2. Accept the consent form rendered via the shared core component.
3. Provide your participant identifier (email or assigned token) in the
   module-specific lobby; for Werewolf this creates a FastAPI session.
4. Once matched, the game board polls `/games/{slug}/sessions/{id}` and submits
   actions via `/games/{slug}/sessions/{id}/actions`.

Additional games can be added by creating a folder under `src/games/<slug>`
with API helpers, hooks, and components, then registering it in
`src/core/config/games.ts`.
