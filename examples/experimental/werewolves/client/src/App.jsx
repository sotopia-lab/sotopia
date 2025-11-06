import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Moon, Sun, Users, Skull, Eye, Heart, Play, UserCheck } from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:8000';

function App() {
  const [gameId, setGameId] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [ws, setWs] = useState(null);
  
  // Lobby state
  const [showLobby, setShowLobby] = useState(true);
  const [gameMode, setGameMode] = useState('spectate');
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [availablePlayers, setAvailablePlayers] = useState([]);
  
  // Action state
  const [actionArgument, setActionArgument] = useState('');
  const [waitingForInput, setWaitingForInput] = useState(false);

  useEffect(() => {
    loadRoster();
  }, []);

  useEffect(() => {
    if (gameId && !ws) {
      connectWebSocket(gameId);
    }
    return () => {
      if (ws) ws.close();
    };
  }, [gameId]);

  const connectWebSocket = (gid) => {
    //const websocket = new WebSocket(`ws://localhost:8000/ws/${gid}`);
    const wsUrl = API_URL.replace('http', 'ws') + `/ws/${gid}`;
    const websocket = new WebSocket(wsUrl);

    websocket.onopen = () => {
      console.log('✅ WebSocket connected');
    };
    
    websocket.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log('📨 Event received:', data);
      
      if (data.type === 'state_sync') {
        setEvents(data.data.events || []);
      } else if (data.type === 'input_request' && data.player === selectedPlayer) {
        setWaitingForInput(true);
      } else {
        setEvents(prev => [...prev, data]);
      }
    };
    
    websocket.onerror = (error) => {
      console.error('❌ WebSocket error:', error);
    };
    
    websocket.onclose = () => {
      console.log('🔌 WebSocket closed');
    };
    
    setWs(websocket);
  };

  const loadRoster = async () => {
    try {
      const res = await axios.get(`${API_URL}/game/roster`);
      setAvailablePlayers(res.data.players);
      if (res.data.players.length > 0) {
        setSelectedPlayer(res.data.players[0].name);
      }
    } catch (err) {
      console.error('Failed to load roster:', err);
    }
  };

  const startGame = async () => {
    try {
      setLoading(true);
      
      const humanPlayers = gameMode === 'play' ? [selectedPlayer] : [];
      console.log('🎮 Starting game:', { gameMode, humanPlayers });
      
      const res = await axios.post(`${API_URL}/game/start`, {
        human_players: humanPlayers,
        game_mode: gameMode
      });
      
      console.log('✅ Game started:', res.data);
      setGameId(res.data.game_id);
      setShowLobby(false);
    } catch (err) {
      console.error('❌ Failed to start game:', err);
      alert('Failed to start game: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  const submitAction = async () => {
    if (!actionArgument.trim()) return;
    
    try {
      setLoading(true);
      
      await axios.post(`${API_URL}/game/action`, {
        game_id: gameId,
        player_name: selectedPlayer,
        action_type: 'speak',
        argument: actionArgument
      });
      
      setActionArgument('');
      setWaitingForInput(false);
    } catch (err) {
      console.error('❌ Action failed:', err);
      alert('Action failed: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const getAlivePlayers = () => {
    if (!availablePlayers.length) return [];
    const dead = new Set();
    events.forEach(e => {
      if (e.type === 'death' && e.content) {
        availablePlayers.forEach(p => {
          if (e.content.includes(p.name)) {
            dead.add(p.name);
          }
        });
      }
    });
    return availablePlayers.filter(p => !dead.has(p.name));
  };

  const alivePlayers = getAlivePlayers();
  const isGameOver = events.some(e => e.type === 'game_over');

  // Lobby Screen
  if (showLobby) {
    return (
      <div className="lobby-screen">
        <div className="lobby-content">
          <Moon className="w-16 h-16 mb-4" />
          <h1>Werewolf Game</h1>
          <p className="lobby-subtitle">Social Deduction with AI Agents</p>
          
          <div className="lobby-options">
            <div className="game-mode-selector">
              <button
                className={`mode-btn ${gameMode === 'spectate' ? 'active' : ''}`}
                onClick={() => setGameMode('spectate')}
              >
                <Eye className="w-5 h-5" />
                Spectate (Watch AI vs AI)
              </button>
              
              <button
                className={`mode-btn ${gameMode === 'play' ? 'active' : ''}`}
                onClick={() => setGameMode('play')}
              >
                <UserCheck className="w-5 h-5" />
                Play (Join as Human)
              </button>
            </div>
            
            {gameMode === 'play' && (
              <div className="player-selector">
                <label>Choose Your Character:</label>
                <select 
                  value={selectedPlayer} 
                  onChange={(e) => setSelectedPlayer(e.target.value)}
                  className="player-select"
                >
                  {availablePlayers.map((player) => (
                    <option key={player.name} value={player.name}>
                      {player.name} ({player.role})
                    </option>
                  ))}
                </select>
              </div>
            )}
            
            <button
              className="start-btn"
              onClick={startGame}
              disabled={loading}
            >
              <Play className="w-5 h-5" />
              {loading ? 'Starting Game...' : 'Start Game'}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Main Game Screen
  return (
    <div className="app">
      <header className="header">
        <div className="header-content">
          <div className="header-title">
            <Moon className="w-8 h-8" />
            <h1>Werewolf</h1>
            {gameMode === 'spectate' && (
              <span className="spectate-badge">Spectator Mode</span>
            )}
          </div>
          
          <div className="header-stats">
            <Users className="w-4 h-4" />
            <span>{alivePlayers.length} alive</span>
          </div>
        </div>
      </header>

      <div className="main-container">
        <aside className="sidebar">
          {gameMode === 'play' && (
            <div className="player-card">
              <div className="player-header">
                <Users className="w-5 h-5" />
                <div>
                  <h3>{selectedPlayer}</h3>
                  <p className="role-badge">
                    {availablePlayers.find(p => p.name === selectedPlayer)?.role || 'Player'}
                  </p>
                </div>
              </div>
            </div>
          )}

          <div className="players-list">
            <h4>Players ({alivePlayers.length} alive)</h4>
            <ul>
              {alivePlayers.map((player) => (
                <li key={player.name}>
                  <div className="player-item">
                    <span>{player.name}</span>
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {isGameOver && (
            <div className="winner-card">
              <h3>🎉 Game Over!</h3>
              <p className="winner">
                {events.find(e => e.type === 'game_over')?.content || 'Game finished!'}
              </p>
              <button className="restart-btn" onClick={() => window.location.reload()}>
                Play Again
              </button>
            </div>
          )}
        </aside>

        <main className="main-content">
          <div className="events-feed">
            <h3>Game Events ({events.length})</h3>
            <div className="events-list">
              {events.length === 0 ? (
                <p className="no-events">Waiting for game to start...</p>
              ) : (
                events.map((event, idx) => {
                  const content = event.content || '';
                  const type = event.type || '';
                  
                  return (
                    <div 
                      key={idx} 
                      className={`event ${type === 'death' ? 'death' : type === 'phase' ? 'phase' : type === 'speech' ? 'speech' : type === 'action' ? 'action' : type === 'vote' ? 'vote' : 'normal'}`}
                    >
                      {type === 'speech' && event.speaker ? (
                        <>
                          <div className="event-speaker">{event.speaker}</div>
                          <div className="event-message">{content}</div>
                        </>
                      ) : (
                        <div className="event-message">{content}</div>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {gameMode === 'play' && waitingForInput && !isGameOver && (
            <div className="action-panel">
              <h3>✨ Your Turn!</h3>
              
              <textarea
                className="action-input"
                placeholder="Type your message or action (e.g., 'I think Bram is suspicious' or 'vote Celeste')"
                value={actionArgument}
                onChange={(e) => setActionArgument(e.target.value)}
                disabled={loading}
                autoFocus
              />

              <button
                className="submit-btn"
                onClick={submitAction}
                disabled={loading || !actionArgument.trim()}
              >
                {loading ? 'Submitting...' : 'Submit'}
              </button>
            </div>
          )}

          {gameMode === 'play' && !waitingForInput && !isGameOver && (
            <div className="waiting-turn">
              <p>Waiting for your turn...</p>
            </div>
          )}
          
          {gameMode === 'spectate' && !isGameOver && (
            <div className="spectate-info">
              <Eye className="w-12 h-12 mb-2" />
              <p>Spectator Mode</p>
              <span>Watching AI agents play ({events.length} events)</span>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
