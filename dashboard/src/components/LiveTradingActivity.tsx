import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Clock,
  DollarSign,
  Target,
  AlertTriangle,
  CheckCircle,
  XCircle,
  BarChart3,
  Zap
} from 'lucide-react';
import { format } from 'date-fns';

interface TradingActivity {
  id: string;
  timestamp: string;
  type: 'analysis' | 'decision' | 'order' | 'execution' | 'alert';
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD' | 'ANALYZE';
  price?: number;
  quantity?: number;
  confidence?: number;
  agent: string;
  status: 'pending' | 'success' | 'failed' | 'cancelled';
  message: string;
  details?: any;
}

interface LiveTradingActivityProps {
  className?: string;
}

export const LiveTradingActivity: React.FC<LiveTradingActivityProps> = ({ className = '' }) => {
  const [activities, setActivities] = useState<TradingActivity[]>([]);
  const [filter, setFilter] = useState<'all' | 'analysis' | 'orders' | 'alerts'>('all');
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    // Connect to WebSocket for real-time updates
    const connectWebSocket = () => {
      const ws = new WebSocket(`ws://localhost:8000/ws/trading-activity`);

      ws.onopen = () => {
        setIsConnected(true);
        console.log('Connected to trading activity feed');
      };

      ws.onmessage = (event) => {
        const activity: TradingActivity = JSON.parse(event.data);
        setActivities(prev => [activity, ...prev.slice(0, 99)]); // Keep last 100 activities
      };

      ws.onclose = () => {
        setIsConnected(false);
        console.log('Disconnected from trading activity feed');
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      return ws;
    };

    const ws = connectWebSocket();

    // Load initial activities
    loadInitialActivities();

    return () => {
      ws.close();
    };
  }, []);

  const loadInitialActivities = async () => {
    try {
      const response = await fetch('/api/v1/trading/activity?limit=50');
      if (response.ok) {
        const data = await response.json();
        setActivities(data.activities || []);
      }
    } catch (error) {
      console.error('Failed to load trading activities:', error);
    }
  };

  const getActivityIcon = (activity: TradingActivity) => {
    switch (activity.type) {
      case 'analysis':
        return <BarChart3 className="w-4 h-4" />;
      case 'decision':
        return <Target className="w-4 h-4" />;
      case 'order':
        return activity.action === 'BUY' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />;
      case 'execution':
        return <Zap className="w-4 h-4" />;
      case 'alert':
        return <AlertTriangle className="w-4 h-4" />;
      default:
        return <Activity className="w-4 h-4" />;
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="w-4 h-4 text-green-500" />;
      case 'failed':
        return <XCircle className="w-4 h-4 text-red-500" />;
      case 'cancelled':
        return <XCircle className="w-4 h-4 text-gray-500" />;
      default:
        return <Clock className="w-4 h-4 text-yellow-500" />;
    }
  };

  const getActivityColor = (activity: TradingActivity) => {
    if (activity.status === 'failed') return 'border-red-200 bg-red-50';
    if (activity.status === 'success') return 'border-green-200 bg-green-50';
    if (activity.type === 'alert') return 'border-yellow-200 bg-yellow-50';
    return 'border-gray-200 bg-white';
  };

  const filteredActivities = activities.filter(activity => {
    if (filter === 'all') return true;
    if (filter === 'analysis') return activity.type === 'analysis' || activity.type === 'decision';
    if (filter === 'orders') return activity.type === 'order' || activity.type === 'execution';
    if (filter === 'alerts') return activity.type === 'alert';
    return true;
  });

  return (
    <div className={`bg-white rounded-lg shadow-lg ${className}`}>
      <div className="p-6 border-b">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <Activity className="w-6 h-6 mr-2 text-blue-600" />
            <h2 className="text-xl font-bold">Live Trading Activity</h2>
            <div className={`ml-3 flex items-center ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
              <span className="text-sm">{isConnected ? 'Live' : 'Disconnected'}</span>
            </div>
          </div>

          <div className="flex space-x-2">
            {['all', 'analysis', 'orders', 'alerts'].map((filterType) => (
              <button
                key={filterType}
                onClick={() => setFilter(filterType as any)}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  filter === filterType
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="p-6">
        <div className="space-y-3 max-h-96 overflow-y-auto">
          <AnimatePresence>
            {filteredActivities.map((activity) => (
              <motion.div
                key={activity.id}
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -20 }}
                className={`p-4 rounded-lg border ${getActivityColor(activity)} transition-all hover:shadow-md`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-start space-x-3">
                    <div className="flex-shrink-0 mt-1">
                      {getActivityIcon(activity)}
                    </div>

                    <div className="flex-1 min-w-0">
                      <div className="flex items-center space-x-2">
                        <span className="font-medium text-gray-900">{activity.symbol}</span>
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          activity.action === 'BUY' ? 'bg-green-100 text-green-800' :
                          activity.action === 'SELL' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {activity.action}
                        </span>
                        <span className="text-xs text-gray-500">{activity.agent}</span>
                      </div>

                      <p className="text-sm text-gray-600 mt-1">{activity.message}</p>

                      {(activity.price || activity.quantity) && (
                        <div className="flex items-center space-x-4 mt-2 text-sm">
                          {activity.price && (
                            <div className="flex items-center">
                              <DollarSign className="w-3 h-3 mr-1" />
                              <span>₹{activity.price.toFixed(2)}</span>
                            </div>
                          )}
                          {activity.quantity && (
                            <div className="flex items-center">
                              <span className="text-gray-500">Qty:</span>
                              <span className="ml-1">{activity.quantity}</span>
                            </div>
                          )}
                          {activity.confidence && (
                            <div className="flex items-center">
                              <span className="text-gray-500">Confidence:</span>
                              <span className="ml-1">{(activity.confidence * 100).toFixed(0)}%</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center space-x-2">
                    {getStatusIcon(activity.status)}
                    <span className="text-xs text-gray-500">
                      {format(new Date(activity.timestamp), 'HH:mm:ss')}
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {filteredActivities.length === 0 && (
            <div className="text-center py-8 text-gray-500">
              <Activity className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>No trading activities yet</p>
              <p className="text-sm">Activities will appear here when the system is running</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
