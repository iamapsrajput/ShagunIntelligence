import { io, Socket } from 'socket.io-client';

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  bid: number;
  ask: number;
  timestamp: string;
}

export interface AgentActivity {
  agentId: string;
  agentType: string;
  action: string;
  confidence: number;
  analysis: any;
  timestamp: string;
}

export interface TradeExecution {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  quantity: number;
  price: number;
  status: 'PENDING' | 'EXECUTED' | 'FAILED';
  agentDecisions: AgentActivity[];
  timestamp: string;
  rationale: string;
}

export interface PortfolioUpdate {
  totalValue: number;
  dayPnL: number;
  dayPnLPercent: number;
  positions: Position[];
  cash: number;
  timestamp: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avgPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
}

export interface SystemStatus {
  isActive: boolean;
  activeAgents: string[];
  riskLevel: 'LOW' | 'MEDIUM' | 'HIGH';
  lastUpdate: string;
}

interface WebSocketEvents {
  'market:update': (data: MarketData) => void;
  'agent:activity': (data: AgentActivity) => void;
  'trade:execution': (data: TradeExecution) => void;
  'portfolio:update': (data: PortfolioUpdate) => void;
  'system:status': (data: SystemStatus) => void;
  'error': (error: any) => void;
}

class WebSocketService {
  private socket: Socket | null = null;
  private eventHandlers: Map<string, Set<Function>> = new Map();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  connect(url?: string) {
    if (this.socket?.connected) {
      return;
    }

    // Use environment variable or default to backend URL
    const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';
    const wsUrl = url || `${apiBaseUrl}/ws`;

    this.socket = io(wsUrl, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionAttempts: this.maxReconnectAttempts,
      reconnectionDelay: this.reconnectDelay,
    });

    this.setupEventListeners();
  }

  private setupEventListeners() {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
      this.emit('connected', true);
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
      this.emit('connected', false);
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;

      if (this.reconnectAttempts >= this.maxReconnectAttempts) {
        this.emit('error', { type: 'CONNECTION_FAILED', message: 'Max reconnection attempts reached' });
      }
    });

    // Market data events
    this.socket.on('market:update', (data: MarketData) => {
      this.emit('market:update', data);
    });

    // Agent activity events
    this.socket.on('agent:activity', (data: AgentActivity) => {
      this.emit('agent:activity', data);
    });

    // Trade execution events
    this.socket.on('trade:execution', (data: TradeExecution) => {
      this.emit('trade:execution', data);
    });

    // Portfolio update events
    this.socket.on('portfolio:update', (data: PortfolioUpdate) => {
      this.emit('portfolio:update', data);
    });

    // System status events
    this.socket.on('system:status', (data: SystemStatus) => {
      this.emit('system:status', data);
    });
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  on<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]) {
    if (!this.eventHandlers.has(event)) {
      this.eventHandlers.set(event, new Set());
    }
    this.eventHandlers.get(event)!.add(handler);
  }

  off<K extends keyof WebSocketEvents>(event: K, handler: WebSocketEvents[K]) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  private emit(event: string, data: any) {
    const handlers = this.eventHandlers.get(event);
    if (handlers) {
      handlers.forEach(handler => handler(data));
    }
  }

  // Send commands to the server
  sendCommand(command: string, data?: any) {
    if (!this.socket?.connected) {
      console.error('WebSocket not connected');
      return;
    }

    this.socket.emit(command, data);
  }

  // Specific command methods
  subscribeToSymbol(symbol: string) {
    this.sendCommand('subscribe:symbol', { symbol });
  }

  unsubscribeFromSymbol(symbol: string) {
    this.sendCommand('unsubscribe:symbol', { symbol });
  }

  updateRiskParameters(params: any) {
    this.sendCommand('update:risk', params);
  }

  toggleSystem(active: boolean) {
    this.sendCommand('system:toggle', { active });
  }

  executeTrade(trade: any) {
    this.sendCommand('trade:execute', trade);
  }
}

export const wsService = new WebSocketService();
export default wsService;
