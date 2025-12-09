'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';

export default function Home() {
  const router = useRouter();
  const [playerName, setPlayerName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const createGame = async () => {
    if (!playerName.trim()) {
      setError('Please enter your name');
      return;
    }

    setLoading(true);
    setError('');

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/game/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ player_name: playerName }),
      });

      if (!response.ok) {
        throw new Error('Failed to create game');
      }

      const data = await response.json();
      console.log('Game created:', data);

      // Redirect to game page
      router.push(`/game/${data.game_id}`);
    } catch (err) {
      console.error('Error creating game:', err);
      setError('Failed to create game. Please try again.');
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 flex items-center justify-center p-6">
      <div className="max-w-md w-full">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-5xl font-bold text-white mb-4">🌕 Duskmire Werewolves</h1>
          <p className="text-gray-300 text-lg">
            A social deduction game powered by AI
          </p>
        </div>

        {/* Create Game Card */}
        <div className="bg-slate-800 rounded-lg p-8 shadow-xl">
          <h2 className="text-2xl font-semibold text-white mb-6">Create New Game</h2>

          <div className="space-y-4">
            <div>
              <label htmlFor="playerName" className="block text-sm font-medium text-gray-300 mb-2">
                Your Name
              </label>
              <input
                id="playerName"
                type="text"
                value={playerName}
                onChange={(e) => setPlayerName(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && createGame()}
                placeholder="Enter your name..."
                className="w-full px-4 py-3 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none transition-colors"
                disabled={loading}
              />
            </div>

            {error && (
              <div className="bg-red-900/50 border border-red-600 text-red-200 px-4 py-3 rounded">
                {error}
              </div>
            )}

            <button
              onClick={createGame}
              disabled={loading || !playerName.trim()}
              className="w-full bg-gradient-to-r from-purple-600 to-purple-800 hover:from-purple-700 hover:to-purple-900 disabled:from-gray-600 disabled:to-gray-700 disabled:cursor-not-allowed text-white font-bold py-3 rounded-lg transition-all transform hover:scale-105"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin h-5 w-5 mr-3" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                      fill="none"
                    />
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    />
                  </svg>
                  Creating Game...
                </span>
              ) : (
                'Create Game'
              )}
            </button>
          </div>
        </div>

        {/* Info Section */}
        <div className="mt-8 bg-slate-800/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-3">How to Play</h3>
          <ul className="text-sm text-gray-300 space-y-2">
            <li>🐺 Some players are secretly werewolves</li>
            <li>🏘️ Villagers must find and eliminate werewolves</li>
            <li>🌙 At night, werewolves choose a victim</li>
            <li>☀️ During the day, everyone discusses and votes</li>
            <li>🔮 Special roles have unique powers</li>
          </ul>
        </div>

        {/* Footer */}
        <div className="mt-6 text-center text-sm text-gray-500">
          <p>Powered by Sotopia + Next.js + FastAPI</p>
        </div>
      </div>
    </div>
  );
}
