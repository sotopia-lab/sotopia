'use client';

import { useEffect, useRef, useState } from 'react';
import { useParams } from 'next/navigation';

interface GameEvent {
  type: string;
  speaker?: string;
  content: string;
  timestamp: string;
  event_class?: string;
  available_actions?: string[];
}

export default function GamePage() {
  const { id } = useParams<{ id: string }>();
  const [events, setEvents] = useState<GameEvent[]>([]);
  const [message, setMessage] = useState('');
  const [selectedAction, setSelectedAction] = useState<string>('');
  const [availableActions, setAvailableActions] = useState<string[]>([]);
  const [yourTurn, setYourTurn] = useState(false);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const eventsEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    eventsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [events]);

  // WebSocket connection
  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000';
    const ws = new WebSocket(`${wsUrl}/ws/${id}`);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('WebSocket connected');
      setConnected(true);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as GameEvent;
      console.log('Received:', data);

      // Update events
      setEvents((prev) => [...prev, data]);

      // Check if it's your turn
      if (data.type === 'input_request') {
        setYourTurn(true);
        setAvailableActions(data.available_actions || []);
        if (data.available_actions && data.available_actions.length > 0) {
          setSelectedAction(data.available_actions[0]);
        }
      }

      // Reset turn when action is received
      if (data.type === 'action_received') {
        setYourTurn(false);
        setAvailableActions([]);
        setSelectedAction('');
        setMessage('');
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setConnected(false);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setConnected(false);
    };

    return () => {
      ws.close();
    };
  }, [id]);

  const submitAction = async () => {
    if (!selectedAction || !yourTurn) return;

    try {
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
      const response = await fetch(`${backendUrl}/api/game/${id}/action`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          action_type: selectedAction,
          argument: message,
          timestamp: new Date().toISOString(),
        }),
      });

      if (response.ok) {
        console.log('Action submitted successfully');
      }
    } catch (error) {
      console.error('Failed to submit action:', error);
    }
  };

  const getEventClassName = (eventClass?: string) => {
    const baseClasses = 'border-l-4 p-4 mb-3 rounded-r';
    switch (eventClass) {
      case 'phase':
        return `${baseClasses} border-orange-500 bg-slate-700`;
      case 'death':
        return `${baseClasses} border-red-500 bg-red-950`;
      case 'speak':
        return `${baseClasses} border-blue-500 bg-slate-800`;
      case 'action':
        return `${baseClasses} border-yellow-500 bg-slate-800`;
      default:
        return `${baseClasses} border-gray-500 bg-slate-800`;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 to-slate-800 text-white p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="bg-gradient-to-r from-purple-600 to-purple-800 p-6 rounded-lg mb-6">
          <h1 className="text-3xl font-bold mb-2">🌕 Duskmire Werewolves</h1>
          <div className="flex items-center gap-4 text-sm">
            <span>Game ID: {id}</span>
            <span className={`px-3 py-1 rounded ${connected ? 'bg-green-600' : 'bg-red-600'}`}>
              {connected ? '● Connected' : '○ Disconnected'}
            </span>
            {yourTurn && (
              <span className="px-3 py-1 rounded bg-yellow-600 animate-pulse">
                🎮 YOUR TURN!
              </span>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Events Feed */}
          <div className="lg:col-span-2">
            <div className="bg-slate-800 rounded-lg p-4">
              <h2 className="text-xl font-semibold mb-4">Game Events</h2>
              <div className="h-[600px] overflow-y-auto pr-2 space-y-2">
                {events.length === 0 && (
                  <p className="text-gray-400 text-center py-8">Waiting for game to start...</p>
                )}
                {events.map((event, i) => (
                  <div key={i} className={getEventClassName(event.event_class)}>
                    <div className="text-xs text-gray-400 mb-1">
                      {new Date(event.timestamp).toLocaleTimeString()}
                    </div>
                    {event.speaker && (
                      <div className="font-bold text-purple-400 mb-1">{event.speaker}</div>
                    )}
                    <div className="text-sm leading-relaxed">{event.content}</div>
                  </div>
                ))}
                <div ref={eventsEndRef} />
              </div>
            </div>
          </div>

          {/* Action Panel */}
          <div className="lg:col-span-1">
            <div className={`bg-slate-800 rounded-lg p-4 ${yourTurn ? 'ring-4 ring-yellow-500' : ''}`}>
              <h2 className="text-xl font-semibold mb-4">Your Action</h2>

              {!yourTurn ? (
                <p className="text-gray-400 text-center py-8">Waiting for your turn...</p>
              ) : (
                <div className="space-y-4">
                  {/* Action Type Selector */}
                  <div>
                    <label className="block text-sm font-medium mb-2">Select Action</label>
                    <div className="flex flex-wrap gap-2">
                      {availableActions.map((action) => (
                        <button
                          key={action}
                          onClick={() => setSelectedAction(action)}
                          className={`px-4 py-2 rounded transition-colors ${
                            selectedAction === action
                              ? 'bg-purple-600 text-white'
                              : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
                          }`}
                        >
                          {action.charAt(0).toUpperCase() + action.slice(1)}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Message Input */}
                  {selectedAction && selectedAction !== 'none' && (
                    <div>
                      <label className="block text-sm font-medium mb-2">Message / Target</label>
                      <textarea
                        value={message}
                        onChange={(e) => setMessage(e.target.value)}
                        placeholder={
                          selectedAction === 'speak'
                            ? 'Enter your message...'
                            : selectedAction === 'action'
                            ? 'Enter action (e.g., vote NAME, kill NAME)...'
                            : 'Enter argument...'
                        }
                        className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded focus:border-purple-500 focus:outline-none resize-none"
                        rows={4}
                      />
                    </div>
                  )}

                  {/* Submit Button */}
                  <button
                    onClick={submitAction}
                    disabled={!selectedAction}
                    className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-3 rounded transition-colors"
                  >
                    Submit Action
                  </button>
                </div>
              )}
            </div>

            {/* Help Section */}
            <div className="bg-slate-800 rounded-lg p-4 mt-4">
              <h3 className="text-lg font-semibold mb-2">Quick Guide</h3>
              <ul className="text-sm text-gray-300 space-y-2">
                <li>• <strong>Speak:</strong> Talk during discussion</li>
                <li>• <strong>Action:</strong> Vote, kill, inspect, etc.</li>
                <li>• <strong>None:</strong> Pass your turn</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
