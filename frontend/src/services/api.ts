import axios, { AxiosInstance, AxiosRequestConfig, AxiosError } from 'axios';

// Create axios instance with default config
const createAPIInstance = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1',
    timeout: 30000,
    headers: {
      'Content-Type': 'application/json',
    },
  });

  // Request interceptor to add auth token
  instance.interceptors.request.use(
    (config) => {
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

  // Response interceptor for error handling
  instance.interceptors.response.use(
    (response) => response,
    async (error: AxiosError) => {
      if (error.response?.status === 401) {
        // Handle unauthorized access
        localStorage.removeItem('authToken');
        window.location.href = '/login';
      } else if (error.response?.status === 403) {
        // Handle forbidden access
        console.error('Access forbidden:', error.response.data);
      } else if (error.response?.status >= 500) {
        // Handle server errors
        console.error('Server error:', error.response.data);
      }
      
      return Promise.reject(error);
    }
  );

  return instance;
};

// Create the API instance
export const api = createAPIInstance();

// API service methods
export const apiService = {
  // Data Quality endpoints
  dataQuality: {
    getMetrics: (symbol: string, lookbackMinutes?: number) =>
      api.get(`/data-quality/metrics/${symbol}`, { 
        params: { lookback_minutes: lookbackMinutes } 
      }),
    
    getSourceHealth: () => 
      api.get('/data-quality/source-health-multi'),
    
    getSentimentFusion: (symbol: string, lookbackHours?: number) =>
      api.get(`/data-quality/sentiment-fusion/${symbol}`, {
        params: { lookback_hours: lookbackHours }
      }),
    
    getFailoverLogs: (params?: any) =>
      api.get('/data-quality/failover-logs', { params }),
    
    getQualityAlerts: (params?: any) =>
      api.get('/data-quality/quality-alerts', { params }),
    
    acknowledgeAlert: (alertId: string) =>
      api.post(`/data-quality/quality-alerts/${alertId}/acknowledge`),
    
    getStreamHealth: () =>
      api.get('/data-quality/stream-health'),
    
    getAPICostSummary: (period: string = 'month') =>
      api.get('/data-quality/api-costs/summary', { 
        params: { period } 
      }),
  },

  // API Management endpoints
  apiManagement: {
    getConfig: (provider?: string) =>
      api.get('/management/config', { 
        params: provider ? { provider } : {} 
      }),
    
    getEnabledAPIs: () =>
      api.get('/management/config/enabled'),
    
    getCosts: () =>
      api.get('/management/config/costs'),
    
    getKeysStatus: () =>
      api.get('/management/keys/status'),
    
    updateKey: (data: any) =>
      api.post('/management/keys/update', data),
    
    getRateLimits: (provider?: string) =>
      api.get('/management/rate-limits', { 
        params: provider ? { provider } : {} 
      }),
    
    getUsage: (provider?: string) =>
      api.get('/management/usage', { 
        params: provider ? { provider } : {} 
      }),
    
    getHealth: (provider?: string) =>
      api.get('/management/health', { 
        params: provider ? { provider } : {} 
      }),
    
    getDashboard: () =>
      api.get('/management/dashboard'),
  },

  // Trading endpoints
  trading: {
    getPositions: () =>
      api.get('/trading/positions'),
    
    placeOrder: (orderData: any) =>
      api.post('/trading/orders', orderData),
    
    getOrders: () =>
      api.get('/trading/orders'),
    
    cancelOrder: (orderId: string) =>
      api.delete(`/trading/orders/${orderId}`),
  },

  // Market Data endpoints
  marketData: {
    getQuote: (symbol: string) =>
      api.get(`/market-data/quote/${symbol}`),
    
    getHistorical: (symbol: string, params?: any) =>
      api.get(`/market-data/historical/${symbol}`, { params }),
    
    getMarketOverview: () =>
      api.get('/market-data/overview'),
  },

  // Authentication endpoints
  auth: {
    login: (credentials: { username: string; password: string }) =>
      api.post('/auth/login', credentials),
    
    logout: () =>
      api.post('/auth/logout'),
    
    getCurrentUser: () =>
      api.get('/auth/me'),
    
    refreshToken: () =>
      api.post('/auth/refresh'),
  },
};

// WebSocket URL helper
export const getWebSocketUrl = (path: string): string => {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const host = process.env.REACT_APP_WS_URL || window.location.host;
  return `${protocol}//${host}${path}`;
};

export default api;