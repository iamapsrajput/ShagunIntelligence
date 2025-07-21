import axios, { AxiosInstance, AxiosError } from 'axios';
import toast from 'react-hot-toast';

export interface ApiError {
  message: string;
  code?: string;
  details?: any;
}

class ApiService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: '/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('authToken');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError<ApiError>) => {
        const message = error.response?.data?.message || 'An error occurred';
        
        if (error.response?.status === 401) {
          // Handle unauthorized
          localStorage.removeItem('authToken');
          window.location.href = '/login';
        } else if (error.response?.status === 500) {
          toast.error('Server error. Please try again later.');
        }

        return Promise.reject(error);
      }
    );
  }

  // Market Data APIs
  async getMarketData(symbol: string) {
    const response = await this.client.get(`/market/quote/${symbol}`);
    return response.data;
  }

  async getHistoricalData(symbol: string, interval: string, days: number = 30) {
    const response = await this.client.get(`/market/historical/${symbol}`, {
      params: { interval, days },
    });
    return response.data;
  }

  async getWatchlist() {
    const response = await this.client.get('/market/watchlist');
    return response.data;
  }

  // Trading APIs
  async getPositions() {
    const response = await this.client.get('/trading/positions');
    return response.data;
  }

  async getOrders() {
    const response = await this.client.get('/trading/orders');
    return response.data;
  }

  async placeOrder(order: any) {
    const response = await this.client.post('/trading/orders', order);
    return response.data;
  }

  async cancelOrder(orderId: string) {
    const response = await this.client.delete(`/trading/orders/${orderId}`);
    return response.data;
  }

  // Portfolio APIs
  async getPortfolio() {
    const response = await this.client.get('/portfolio/summary');
    return response.data;
  }

  async getPortfolioHistory(days: number = 30) {
    const response = await this.client.get('/portfolio/history', {
      params: { days },
    });
    return response.data;
  }

  // Agent APIs
  async getAgentStatus() {
    const response = await this.client.get('/agents/status');
    return response.data;
  }

  async getAgentAnalysis(agentType: string, symbol?: string) {
    const response = await this.client.get(`/agents/${agentType}/analysis`, {
      params: { symbol },
    });
    return response.data;
  }

  async updateAgentConfig(agentType: string, config: any) {
    const response = await this.client.put(`/agents/${agentType}/config`, config);
    return response.data;
  }

  // System APIs
  async getSystemStatus() {
    const response = await this.client.get('/system/status');
    return response.data;
  }

  async updateSystemSettings(settings: any) {
    const response = await this.client.put('/system/settings', settings);
    return response.data;
  }

  async getRiskParameters() {
    const response = await this.client.get('/system/risk-parameters');
    return response.data;
  }

  async updateRiskParameters(params: any) {
    const response = await this.client.put('/system/risk-parameters', params);
    return response.data;
  }

  // Trade History APIs
  async getTradeHistory(params?: {
    startDate?: string;
    endDate?: string;
    symbol?: string;
    limit?: number;
  }) {
    const response = await this.client.get('/trades/history', { params });
    return response.data;
  }

  async getTradeDetails(tradeId: string) {
    const response = await this.client.get(`/trades/${tradeId}`);
    return response.data;
  }

  // Analytics APIs
  async getPerformanceMetrics(period: string = '1M') {
    const response = await this.client.get('/analytics/performance', {
      params: { period },
    });
    return response.data;
  }

  async getRiskMetrics() {
    const response = await this.client.get('/analytics/risk');
    return response.data;
  }
}

export const apiService = new ApiService();
export default apiService;