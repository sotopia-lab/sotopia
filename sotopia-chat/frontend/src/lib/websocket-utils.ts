/**
 * WebSocket utilities for connecting to Sotopia backend simulation
 */

export enum WSMessageType {
    START_SIM = "START_SIM",
    SERVER_MSG = "SERVER_MSG",
    CLIENT_MSG = "CLIENT_MSG",
    END_SIM = "END_SIM",
    ERROR = "ERROR",
}

export interface WSMessage {
    type: WSMessageType;
    data: Record<string, unknown>;
}

export interface SimulationStartPayload {
    type: "START_SIM";
    data: {
        session_id?: string;
        participant_id?: string;
        env_id: string;
        agent_ids: string[];
        agent_models: string[];
        evaluator_model: string;
        evaluation_dimension_list_name: string;
        max_turns: number;
        env_profile_dict?: Record<string, unknown>;
        agent_profile_dicts?: Array<Record<string, unknown>>;
        [key: string]: unknown;
    };
}

export class SimulationWebSocket {
    private ws: WebSocket | null = null;
    private messageHandlers: Map<string, (data: unknown) => void> = new Map();
    private url: string;
    private token: string;

    constructor(baseUrl: string, token: string) {
        this.url = baseUrl;
        this.token = token;
    }

    /**
     * Connect to the WebSocket endpoint
     */
    async connect(): Promise<void> {
        return new Promise((resolve, reject) => {
            try {
                const wsUrl = `${this.url}/ws/simulation?token=${encodeURIComponent(this.token)}`;
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => resolve();

                this.ws.onerror = (event) => {
                    console.error("[WS] Connection error:", event);
                    reject(new Error("WebSocket connection failed"));
                };

                this.ws.onmessage = (event) => {
                    try {
                        const message: WSMessage = JSON.parse(event.data);
                        this._handleMessage(message);
                    } catch (err) {
                        console.error("[WS] Failed to parse message:", err, event.data);
                    }
                };

                this.ws.onclose = () => {
                    this.ws = null;
                };
            } catch (err) {
                reject(err);
            }
        });
    }

    /**
     * Start a simulation
     */
    startSimulation(payload: SimulationStartPayload): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket is not connected");
        }
        this.ws.send(JSON.stringify(payload));
    }

    /**
     * Send a client message (player action)
     */
    sendClientMessage(data: Record<string, unknown>): void {
        if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
            throw new Error("WebSocket is not connected");
        }
        const message: WSMessage = {
            type: WSMessageType.CLIENT_MSG,
            data,
        };
        this.ws.send(JSON.stringify(message));
    }

    /**
     * Register a handler for a specific message type
     */
    on(messageType: string, handler: (data: unknown) => void): void {
        this.messageHandlers.set(messageType, handler);
    }

    /**
     * Unregister a handler
     */
    off(messageType: string): void {
        this.messageHandlers.delete(messageType);
    }

    /**
     * Internal: handle incoming messages
     */
    private _handleMessage(message: WSMessage): void {
        const handler = this.messageHandlers.get(message.type);
        if (handler) {
            handler(message.data);
        }
    }

    /**
     * Disconnect cleanly
     */
    disconnect(): void {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.close();
        }
        this.messageHandlers.clear();
    }

    /**
     * Check if connected
     */
    isConnected(): boolean {
        return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
    }
}

/**
 * Generate a unique token for this session
 */
export function generateSessionToken(): string {
    // Use a combination of timestamp and random string
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 15);
    return `${timestamp}-${random}`;
}

/**
 * Get or create session token from localStorage
 */
export function getOrCreateSessionToken(): string {
    const key = "sotopia_session_token";
    let token = localStorage.getItem(key);
    if (!token) {
        token = generateSessionToken();
        localStorage.setItem(key, token);
    }
    return token;
}
