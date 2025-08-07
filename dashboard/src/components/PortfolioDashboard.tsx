import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  TrendingUp,
  TrendingDown,
  DollarSign,
  Percent,
  Shield,
  AlertTriangle
} from 'lucide-react';
import { LineChart, AreaChart, Area, PieChart as RePieChart, Pie, Cell,
         BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { format } from 'date-fns';
import { PortfolioUpdate } from '@/services/websocket';
import wsService from '@/services/websocket';

interface PortfolioDashboardProps {
  className?: string;
}

interface PortfolioMetrics {
  totalValue: number;
  dayPnL: number;
  dayPnLPercent: number;
  weekPnL: number;
  weekPnLPercent: number;
  monthPnL: number;
  monthPnLPercent: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  riskRewardRatio: number;
}

interface PortfolioHistory {
  timestamp: string;
  value: number;
  pnl: number;
}

export const PortfolioDashboard: React.FC<PortfolioDashboardProps> = ({ className = '' }) => {
  const [portfolio, setPortfolio] = useState<PortfolioUpdate | null>(null);
  const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null);
  const [history, setHistory] = useState<PortfolioHistory[]>([]);
  const [riskMetrics, setRiskMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Load initial portfolio data
    loadPortfolioData();

    // Subscribe to portfolio updates
    const handlePortfolioUpdate = (update: PortfolioUpdate) => {
      setPortfolio(update);

      // Update history
      setHistory(prev => [...prev, {
        timestamp: update.timestamp,
        value: update.totalValue,
        pnl: update.dayPnL
      }].slice(-100)); // Keep last 100 data points
    };

    wsService.on('portfolio:update', handlePortfolioUpdate);

    return () => {
      wsService.off('portfolio:update', handlePortfolioUpdate);
    };
  }, []);

  const loadPortfolioData = async () => {
    try {
      setLoading(true);
      const api = require('@/services/api').default;

      const [portfolioData, accountBalance, livePositions, metricsData, historyData, riskData] = await Promise.all([
        api.getPortfolio(),
        api.getAccountBalance(),
        api.getLivePositions(),
        api.getPerformanceMetrics('1M'),
        api.getPortfolioHistory(30),
        api.getRiskMetrics()
      ]);

      setPortfolio({
        ...portfolioData,
        accountBalance,
        livePositions
      });
      setMetrics(metricsData);
      setHistory(historyData);
      setRiskMetrics(riskData);
    } catch (error) {
      console.error('Failed to load portfolio data:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!portfolio || !metrics) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-500">
        No portfolio data available
      </div>
    );
  }

  const MetricCard: React.FC<{
    title: string;
    value: string | number;
    change?: number;
    icon: React.ReactNode;
    color?: string;
  }> = ({ title, value, change, icon, color = 'blue' }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow p-6"
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm text-gray-600">{title}</span>
        <div className={`p-2 rounded bg-${color}-100`}>
          {icon}
        </div>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      {change !== undefined && (
        <div className={`flex items-center mt-2 text-sm ${
          change >= 0 ? 'text-green-600' : 'text-red-600'
        }`}>
          {change >= 0 ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
          <span className="ml-1">{change >= 0 ? '+' : ''}{change.toFixed(2)}%</span>
        </div>
      )}
    </motion.div>
  );

  // Prepare data for charts
  const positionData = portfolio.positions.map(pos => ({
    name: pos.symbol,
    value: Math.abs(pos.quantity * pos.currentPrice),
    pnl: pos.unrealizedPnL,
    pnlPercent: pos.unrealizedPnLPercent
  }));

  const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899'];

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Total Portfolio Value"
          value={`$${portfolio.totalValue.toLocaleString()}`}
          change={portfolio.dayPnLPercent}
          icon={<DollarSign className="w-5 h-5 text-blue-600" />}
          color="blue"
        />
        <MetricCard
          title="Day P&L"
          value={`$${portfolio.dayPnL.toLocaleString()}`}
          change={portfolio.dayPnLPercent}
          icon={<TrendingUp className="w-5 h-5 text-green-600" />}
          color="green"
        />
        <MetricCard
          title="Win Rate"
          value={`${(metrics.winRate * 100).toFixed(1)}%`}
          icon={<Percent className="w-5 h-5 text-purple-600" />}
          color="purple"
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpeRatio.toFixed(2)}
          icon={<Shield className="w-5 h-5 text-orange-600" />}
          color="orange"
        />
      </div>

      {/* Portfolio Value Chart */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Portfolio Value</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={history}>
            <defs>
              <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8}/>
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={(value) => format(new Date(value), 'HH:mm')}
            />
            <YAxis />
            <Tooltip
              formatter={(value: any) => `$${value.toLocaleString()}`}
              labelFormatter={(label) => format(new Date(label), 'MMM d, HH:mm')}
            />
            <Area
              type="monotone"
              dataKey="value"
              stroke="#3b82f6"
              fillOpacity={1}
              fill="url(#colorValue)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Position Allocation */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Position Allocation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <RePieChart>
              <Pie
                data={positionData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {positionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip formatter={(value: any) => `$${value.toLocaleString()}`} />
            </RePieChart>
          </ResponsiveContainer>
        </div>

        {/* Position Performance */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">Position Performance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={positionData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip formatter={(value: any) => `$${value.toLocaleString()}`} />
              <Bar dataKey="pnl" fill={(data: any) => data.pnl >= 0 ? '#10b981' : '#ef4444'} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Positions Table */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Open Positions</h3>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead>
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Quantity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Avg Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Current Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  P&L
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  P&L %
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Value
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {portfolio.positions.map((position) => (
                <tr key={position.symbol} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap font-medium">
                    {position.symbol}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    {position.quantity}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${position.avgPrice.toFixed(2)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${position.currentPrice.toFixed(2)}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap font-medium ${
                    position.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    ${position.unrealizedPnL.toFixed(2)}
                  </td>
                  <td className={`px-6 py-4 whitespace-nowrap ${
                    position.unrealizedPnLPercent >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    {position.unrealizedPnLPercent >= 0 ? '+' : ''}
                    {position.unrealizedPnLPercent.toFixed(2)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    ${(position.quantity * position.currentPrice).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk Metrics */}
      {riskMetrics && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Shield className="w-5 h-5 mr-2 text-orange-600" />
            Risk Analysis
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <span className="text-sm text-gray-600">Max Drawdown</span>
              <div className="text-xl font-semibold text-red-600">
                {riskMetrics.maxDrawdown.toFixed(2)}%
              </div>
            </div>
            <div>
              <span className="text-sm text-gray-600">Value at Risk (95%)</span>
              <div className="text-xl font-semibold text-orange-600">
                ${riskMetrics.var95.toLocaleString()}
              </div>
            </div>
            <div>
              <span className="text-sm text-gray-600">Beta</span>
              <div className="text-xl font-semibold">
                {riskMetrics.beta.toFixed(2)}
              </div>
            </div>
          </div>

          {riskMetrics.warnings && riskMetrics.warnings.length > 0 && (
            <div className="mt-4 p-3 bg-yellow-50 rounded-lg">
              <div className="flex items-center text-yellow-800">
                <AlertTriangle className="w-5 h-5 mr-2" />
                <span className="font-medium">Risk Warnings</span>
              </div>
              <ul className="mt-2 text-sm text-yellow-700 space-y-1">
                {riskMetrics.warnings.map((warning: string, idx: number) => (
                  <li key={idx}>• {warning}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
