import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { Moon, Sun, Users, Skull, Eye, Heart, AlertCircle, Play, UserCheck } from 'lucide-react';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  // Game state
  const [gameState, setGameState] = useState(null);
  const [playerState, setPlayerState] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // Lobby state
  const [showLobby, setShowLobby] = useState(true);
  const [gameMode, setGameMode] = useState('spectate'); // 'spectate' or 'play'
  const [selectedPlayer, setSelectedPlayer] = useState('');
  const [availablePlayers, setAvailablePlayers] = useState([]);
  
  // Action state
  const [actionArgument, setActionArgument] = useState('');

  // Load available players on mount
  useEffect(() => {
    loadRoster();
  }, []);

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
      setError(null);
      
      const humanPlayers = gameMode === 'play' ? [selectedPlayer] : [];
      
      await axios.post(`${API_URL}/game/init`, {
        human_players: humanPlayers,
        auto_play: true
      });
      
      setShowLobby(false);
      pollGameState();
      
      // Start polling
      const interval = setInterval(pollGameState, 2000);
      return () => clearInterval(interval);
    } catch (err) {
      setError('Failed to start game');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const pollGameState = async () => {
    try {
      const gameRes = await axios.get(`${API_URL}/game/state`);
      setGameState(gameRes.data);
      
      // Update events (deduplicate)
      if (gameRes.data.recent_events?.public) {
        setEvents(gameRes.data.recent_events.public);
      }
      
      // Get player state if playing as human
      if (gameMode === 'play' && selectedPlayer) {
        try {
          const playerRes = await axios.get(`${API_URL}/game/state/${selectedPlayer}`);
          setPlayerState(playerRes.data);
        } catch (err) {
          // Player might not exist yet
          console.error('Player state error:', err);
        }
      }
    } catch (err) {
      console.error('Polling error:', err);
    }
  };

  useEffect(() => {
    if (!showLobby && gameState) {
      const interval = setInterval(pollGameState, 2000);
      return () => clearInterval(interval);
    }
  }, [showLobby, gameState, selectedPlayer, gameMode]);

  const submitAction = async (actionType) => {
    try {
      setLoading(true);
      setError(null);
      
      await axios.post(`${API_URL}/game/action`, {
        player_name: selectedPlayer,
        action_type: actionType,
        argument: actionArgument
      });
      
      setActionArgument('');
      await pollGameState();
    } catch (err) {
      setError(err.response?.data?.detail || 'Action failed');
    } finally {
      setLoading(false);
    }
  };

  const renderPhaseIcon = () => {
    const phase = gameState?.phase || '';
    if (phase.includes('night')) return <Moon className="w-5 h-5" />;
    if (phase.includes('day')) return <Sun className="w-5 h-5" />;
    return <Users className="w-5 h-5" />;
  };

  const renderRoleIcon = () => {
    const role = playerState?.role;
    if (role === 'Seer') return <Eye className="w-5 h-5" />;
    if (role === 'Witch') return <Heart className="w-5 h-5" />;
    if (role === 'Werewolf') return <Skull className="w-5 h-5" />;
    return <Users className="w-5 h-5" />;
  };

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

  // Loading screen
  if (loading && !gameState) {
    return (
      <div className="loading-screen">
        <Moon className="w-12 h-12 animate-pulse" />
        <p>Loading Werewolf Game...</p>
      </div>
    );
  }

  // Main Game Screen
  return (
    <div className="app">
      {/* Header */}
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
            {renderPhaseIcon()}
            <span className="phase-name">
              {gameState?.phase_meta?.display_name || gameState?.phase}
            </span>
            
            <div className="alive-count">
              <Users className="w-4 h-4" />
              <span>{gameState?.alive_players?.length || 0} alive</span>
            </div>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* Sidebar */}
        <aside className="sidebar">
          {gameMode === 'play' && playerState && (
            <div className="player-card">
              <div className="player-header">
                {renderRoleIcon()}
                <div>
                  <h3>{playerState?.player_name || 'Player'}</h3>
                  <p className="role-badge">{playerState?.role}</p>
                </div>
              </div>
              
              <div className="player-stats">
                <div className="stat">
                  <span>Team:</span>
                  <span className="stat-value">{playerState?.team}</span>
                </div>
                <div className="stat">
                  <span>Status:</span>
                  <span className={`stat-value ${playerState?.alive ? 'alive' : 'dead'}`}>
                    {playerState?.alive ? 'Alive' : 'Dead'}
                  </span>
                </div>
              </div>
            </div>
          )}

          {/* Players List */}
          <div className="players-list">
            <h4>Players</h4>
            <ul>
              {gameState?.alive_players?.map((player) => (
                <li key={player} className={gameState?.active_players?.includes(player) ? 'active' : ''}>
                  <div className="player-item">
                    <span>{player}</span>
                    {gameState?.active_players?.includes(player) && (
                      <span className="active-indicator">●</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Game Over */}
          {gameState?.winner && (
            <div className="winner-card">
              <h3>🎉 Game Over!</h3>
              <p className="winner">{gameState.winner.winner} Wins!</p>
              <p className="reason">{gameState.winner.message}</p>
              <button 
                className="restart-btn"
                onClick={() => window.location.reload()}
              >
                Play Again
              </button>
            </div>
          )}
        </aside>

        {/* Main Content */}
        <main className="main-content">
          {/* Events Feed */}
          <div className="events-feed">
            <h3>Game Events</h3>
            <div className="events-list">
              {events.length === 0 ? (
                <p className="no-events">Waiting for events...</p>
              ) : (
                events.map((event, idx) => {
                  // Detect event type
                  const isDeath = event.includes('died') || event.includes('dead') || event.includes('executed') || event.includes('was found dead');
                  const isPhase = event.includes('Phase ') || event.includes('begins:') || event.includes('ends');
                  const isSpeech = event.includes(' said: ');
                  const isAction = event.includes('🗡️') || event.includes('🔮') || event.includes('💊') || event.includes('☠️') || event.includes('⚡');
                  const isVote = event.includes('🗳️') || event.includes('voted') || event.includes('Votes are tallied');
                  
                  // Extract speaker name for speech events
                  let speaker = '';
                  let message = event;
                  if (isSpeech) {
                    const parts = event.split(' said: ');
                    if (parts.length === 2) {
                      speaker = parts[0];
                      message = parts[1].replace(/^"/, '').replace(/"$/, '');
                    }
                  }
                  
                  // Add context hints for phases
                  let contextHint = '';
                  if (event.includes('night_werewolves')) {
                    contextHint = '🌙 Werewolves secretly choose their victim';
                  } else if (event.includes('night_seer')) {
                    contextHint = '🔮 Seer investigates one player';
                  } else if (event.includes('night_witch')) {
                    contextHint = '🧪 Witch can save or poison someone';
                  } else if (event.includes('dawn_report')) {
                    contextHint = '☀️ Results of the night are revealed';
                  } else if (event.includes('day_discussion')) {
                    contextHint = '💬 Everyone discusses who might be a werewolf';
                  } else if (event.includes('day_vote')) {
                    contextHint = '🗳️ Time to vote someone out';
                  }
                  
                  return (
                    <div 
                      key={idx} 
                      className={`event ${
                        isDeath ? 'death' : 
                        isPhase ? 'phase' : 
                        isSpeech ? 'speech' :
                        isAction ? 'action' :
                        isVote ? 'vote' :
                        'normal'
                      }`}
                    >
                      {isSpeech && speaker ? (
                        <>
                          <div className="event-speaker">{speaker}</div>
                          <div className="event-message">{message}</div>
                        </>
                      ) : (
                        <>
                          <div className="event-message">{event.replace('[God]', '').trim()}</div>
                          {contextHint && <div className="event-context">{contextHint}</div>}
                        </>
                      )}
                    </div>
                  );
                })
              )}
            </div>
          </div>

          {/* Action Panel (only for human players) */}
          {gameMode === 'play' && playerState?.is_active && playerState?.alive && (
            <div className="action-panel">
              <h3>Your Turn</h3>
              
              {error && (
                <div className="error-message">
                  <AlertCircle className="w-4 h-4" />
                  <span>{error}</span>
                  <button onClick={() => setError(null)}>×</button>
                </div>
              )}

              <div className="action-buttons">
                {playerState?.available_actions?.map((action) => (
                  action !== 'none' && (
                    <button
                      key={action}
                      className={`action-btn ${action}`}
                      onClick={() => submitAction(action)}
                      disabled={loading || !actionArgument}
                    >
                      {action === 'speak' && '💬 Speak'}
                      {action === 'action' && '⚡ Action'}
                      {action === 'vote' && '🗳️ Vote'}
                    </button>
                  )
                ))}
              </div>

              <textarea
                className="action-input"
                placeholder={
                  playerState?.available_actions?.includes('action')
                    ? "Type your action (e.g., 'kill Aurora', 'inspect Bram', 'vote Celeste')"
                    : "Type your message..."
                }
                value={actionArgument}
                onChange={(e) => setActionArgument(e.target.value)}
                disabled={loading}
              />

              <button
                className="submit-btn"
                onClick={() => submitAction(playerState?.available_actions?.[0])}
                disabled={loading || !actionArgument}
              >
                {loading ? 'Submitting...' : 'Submit'}
              </button>
            </div>
          )}

          {gameMode === 'play' && !playerState?.is_active && playerState?.alive && (
            <div className="waiting-turn">
              <p>Waiting for other players...</p>
            </div>
          )}

          {gameMode === 'play' && !playerState?.alive && (
            <div className="dead-overlay">
              <Skull className="w-16 h-16" />
              <p>You are dead</p>
              <span>Watch the remaining players</span>
            </div>
          )}
          
          {gameMode === 'spectate' && (
            <div className="spectate-info">
              <Eye className="w-12 h-12 mb-2" />
              <p>Spectator Mode</p>
              <span>Watching AI agents play</span>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;