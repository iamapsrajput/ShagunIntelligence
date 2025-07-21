import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowUpCircle, 
  ArrowDownCircle, 
  Clock, 
  CheckCircle, 
  XCircle, 
  AlertCircle,
  ChevronDown,
  ChevronUp,
  Info
} from 'lucide-react';
import { format } from 'date-fns';
import { TradeExecution } from '@/services/websocket';
import wsService from '@/services/websocket';

interface TradeExecutionLogProps {
  limit?: number;
  className?: string;
}

interface ExpandedTrade {
  [key: string]: boolean;
}

export const TradeExecutionLog: React.FC<TradeExecutionLogProps> = ({ 
  limit = 20, 
  className = '' 
}) => {
  const [trades, setTrades] = useState<TradeExecution[]>([]);
  const [expandedTrades, setExpandedTrades] = useState<ExpandedTrade>({});
  const [filter, setFilter] = useState<'all' | 'buy' | 'sell' | 'pending' | 'failed'>('all');

  useEffect(() => {
    // Load initial trade history
    loadTradeHistory();

    // Subscribe to trade executions
    const handleTradeExecution = (trade: TradeExecution) => {
      setTrades(prev => [trade, ...prev].slice(0, limit));
      
      // Auto-expand new trades for 5 seconds
      setExpandedTrades(prev => ({ ...prev, [trade.id]: true }));
      setTimeout(() => {
        setExpandedTrades(prev => ({ ...prev, [trade.id]: false }));
      }, 5000);
    };

    wsService.on('trade:execution', handleTradeExecution);

    return () => {
      wsService.off('trade:execution', handleTradeExecution);
    };
  }, [limit]);

  const loadTradeHistory = async () => {
    try {
      const api = require('@/services/api').default;
      const history = await api.getTradeHistory({ limit });
      setTrades(history);
    } catch (error) {
      console.error('Failed to load trade history:', error);
    }
  };

  const toggleExpanded = (tradeId: string) => {
    setExpandedTrades(prev => ({
      ...prev,
      [tradeId]: !prev[tradeId]
    }));
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'EXECUTED':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'PENDING':
        return <Clock className="w-5 h-5 text-yellow-500 animate-pulse" />;
      case 'FAILED':
        return <XCircle className="w-5 h-5 text-red-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getActionIcon = (action: string) => {
    return action === 'BUY' ? 
      <ArrowUpCircle className="w-5 h-5 text-green-500" /> : 
      <ArrowDownCircle className="w-5 h-5 text-red-500" />;
  };

  const filteredTrades = trades.filter(trade => {
    if (filter === 'all') return true;
    if (filter === 'pending') return trade.status === 'PENDING';
    if (filter === 'failed') return trade.status === 'FAILED';
    return trade.action.toLowerCase() === filter;
  });

  return (
    <div className={`bg-white rounded-lg shadow-lg p-6 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold">Trade Execution Log</h2>
        <div className="flex items-center space-x-2">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as any)}
            className="px-3 py-1 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-primary-500"
          >
            <option value="all">All Trades</option>
            <option value="buy">Buy Orders</option>
            <option value="sell">Sell Orders</option>
            <option value="pending">Pending</option>
            <option value="failed">Failed</option>
          </select>
        </div>
      </div>

      <div className="space-y-3">
        <AnimatePresence>
          {filteredTrades.map((trade) => (
            <motion.div
              key={trade.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="border rounded-lg p-4 hover:shadow-md transition-shadow"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-4">
                  {getActionIcon(trade.action)}
                  <div>
                    <div className="flex items-center space-x-2">
                      <span className="font-semibold">{trade.symbol}</span>
                      <span className="text-sm text-gray-500">
                        {trade.quantity} shares @ ${trade.price.toFixed(2)}
                      </span>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {format(new Date(trade.timestamp), 'MMM d, HH:mm:ss')}
                    </div>
                  </div>
                </div>

                <div className="flex items-center space-x-3">
                  <div className="text-right">
                    <div className="flex items-center space-x-2">
                      {getStatusIcon(trade.status)}
                      <span className={`text-sm font-medium ${
                        trade.status === 'EXECUTED' ? 'text-green-600' :
                        trade.status === 'PENDING' ? 'text-yellow-600' :
                        trade.status === 'FAILED' ? 'text-red-600' :
                        'text-gray-600'
                      }`}>
                        {trade.status}
                      </span>
                    </div>
                    <div className="text-sm text-gray-600 mt-1">
                      ${(trade.quantity * trade.price).toFixed(2)}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => toggleExpanded(trade.id)}
                    className="p-1 hover:bg-gray-100 rounded transition-colors"
                  >
                    {expandedTrades[trade.id] ? 
                      <ChevronUp className="w-4 h-4" /> : 
                      <ChevronDown className="w-4 h-4" />
                    }
                  </button>
                </div>
              </div>

              <AnimatePresence>
                {expandedTrades[trade.id] && (
                  <motion.div
                    initial={{ height: 0, opacity: 0 }}
                    animate={{ height: 'auto', opacity: 1 }}
                    exit={{ height: 0, opacity: 0 }}
                    transition={{ duration: 0.2 }}
                    className="mt-4 pt-4 border-t overflow-hidden"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 mb-2 flex items-center">
                          <Info className="w-4 h-4 mr-1" />
                          Trade Rationale
                        </h4>
                        <p className="text-sm text-gray-600">
                          {trade.rationale || 'No rationale provided'}
                        </p>
                      </div>

                      <div>
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">
                          Agent Decisions
                        </h4>
                        <div className="space-y-2">
                          {trade.agentDecisions.map((decision, idx) => (
                            <div key={idx} className="flex items-center justify-between text-sm">
                              <span className="text-gray-600">
                                {decision.agentType.charAt(0).toUpperCase() + decision.agentType.slice(1)}
                              </span>
                              <div className="flex items-center space-x-2">
                                <span className={`font-medium ${
                                  decision.action === 'buy' ? 'text-green-600' :
                                  decision.action === 'sell' ? 'text-red-600' :
                                  'text-gray-600'
                                }`}>
                                  {decision.action.toUpperCase()}
                                </span>
                                <span className="text-xs text-gray-400">
                                  ({(decision.confidence * 100).toFixed(0)}%)
                                </span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    {trade.agentDecisions.some(d => d.analysis) && (
                      <div className="mt-4">
                        <h4 className="text-sm font-semibold text-gray-700 mb-2">
                          Analysis Details
                        </h4>
                        <div className="bg-gray-50 rounded p-3 text-xs font-mono overflow-x-auto">
                          <pre>{JSON.stringify(
                            trade.agentDecisions.find(d => d.analysis)?.analysis,
                            null,
                            2
                          )}</pre>
                        </div>
                      </div>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          ))}
        </AnimatePresence>

        {filteredTrades.length === 0 && (
          <div className="text-center py-8 text-gray-500">
            <AlertCircle className="w-12 h-12 mx-auto mb-2 text-gray-300" />
            <p>No trades found</p>
          </div>
        )}
      </div>
    </div>
  );
};