import { useState, useEffect, useCallback } from 'react';

export const useWebSocket = (sessionId: string | null) => {
  const [socket, setSocket] = useState<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//localhost:8000/ws/${sessionId}`;
    console.log('Attempting to connect to WebSocket:', wsUrl);
    const ws = new WebSocket(wsUrl);
    setSocket(ws);

    ws.onopen = () => {
      console.log('WebSocket connected successfully');
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = (event) => {
      console.log('WebSocket closed:', event);
    };

    return () => {
      ws.close();
    };
  }, [sessionId]);

  const sendMessage = useCallback((message: string) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      console.log('useWebSocket: ', message)
      socket.send(message);
    }
  }, [socket]);

  return { socket, sendMessage };
};