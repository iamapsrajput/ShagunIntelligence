import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  Brain,
  History,
  Wallet,
  Settings as SettingsIcon,
  Menu,
  X,
  LogOut,
  Activity as ActivityIcon,
  Monitor as MonitorIcon
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { MarketChart } from '@/components/MarketChart';
import { AgentPanel } from '@/components/AgentPanel';
import { TradeExecutionLog } from '@/components/TradeExecutionLog';
import { PortfolioDashboard } from '@/components/PortfolioDashboard';
import { SystemControls } from '@/components/SystemControls';
import { TradingConfiguration } from '@/components/TradingConfiguration';
import { LiveTradingActivity } from '@/components/LiveTradingActivity';
import { SystemMonitoring } from '@/components/SystemMonitoring';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'market' | 'agents' | 'trades' | 'portfolio' | 'controls' | 'config' | 'activity' | 'monitoring'>('market');
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [watchlist] = useState<string[]>(['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']);

  useEffect(() => {
    // Load user preferences
    loadUserPreferences();
    console.log('Dashboard component mounted successfully');
  }, []);

  const loadUserPreferences = async () => {
    try {
      // For now, use default watchlist
      // TODO: Implement API call when watchlist endpoint is available
      console.log('Using default watchlist');
    } catch (error) {
      console.error('Failed to load preferences:', error);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('authToken');
    navigate('/login');
  };

  const Sidebar = () => (
    <div className={`
      fixed inset-y-0 left-0 z-50 w-64 bg-gray-900 transform transition-transform duration-300 ease-in-out
      ${sidebarOpen ? 'translate-x-0' : '-translate-x-full'}
      lg:translate-x-0 lg:static lg:inset-0
    `}>
      <div className="flex items-center justify-between h-16 px-6 bg-gray-800">
        <h1 className="text-xl font-bold text-white">Shagun Intelligence</h1>
        <button
          onClick={() => setSidebarOpen(false)}
          className="lg:hidden text-gray-400 hover:text-white"
        >
          <X size={24} />
        </button>
      </div>

      <nav className="mt-8 px-4">
        <button
          onClick={() => setActiveTab('market')}
          className={`w-full flex items-center px-4 py-3 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'market'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <TrendingUp className="mr-3 h-5 w-5" />
          Market Data
        </button>

        <button
          onClick={() => setActiveTab('agents')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'agents'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <Brain className="mr-3 h-5 w-5" />
          AI Agents
        </button>

        <button
          onClick={() => setActiveTab('trades')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'trades'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <History className="mr-3 h-5 w-5" />
          Trade History
        </button>

        <button
          onClick={() => setActiveTab('portfolio')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'portfolio'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <Wallet className="mr-3 h-5 w-5" />
          Portfolio
        </button>

        <button
          onClick={() => setActiveTab('controls')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'controls'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <SettingsIcon className="mr-3 h-5 w-5" />
          System Controls
        </button>

        <button
          onClick={() => setActiveTab('config')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'config'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <SettingsIcon className="mr-3 h-5 w-5" />
          Trading Config
        </button>

        <button
          onClick={() => setActiveTab('activity')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'activity'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <ActivityIcon className="mr-3 h-5 w-5" />
          Live Activity
        </button>

        <button
          onClick={() => setActiveTab('monitoring')}
          className={`w-full flex items-center px-4 py-3 mt-2 text-sm font-medium rounded-lg transition-colors ${
            activeTab === 'monitoring'
              ? 'bg-primary-600 text-white'
              : 'text-gray-300 hover:bg-gray-800 hover:text-white'
          }`}
        >
          <MonitorIcon className="mr-3 h-5 w-5" />
          System Monitor
        </button>

        <div className="mt-8 pt-8 border-t border-gray-700">
          <button
            onClick={() => navigate('/settings')}
            className="w-full flex items-center px-4 py-3 text-sm font-medium text-gray-300 hover:bg-gray-800 hover:text-white rounded-lg transition-colors"
          >
            <SettingsIcon className="mr-3 h-5 w-5" />
            Settings
          </button>

          <button
            onClick={handleLogout}
            className="w-full flex items-center px-4 py-3 mt-2 text-sm font-medium text-gray-300 hover:bg-gray-800 hover:text-white rounded-lg transition-colors"
          >
            <LogOut className="mr-3 h-5 w-5" />
            Logout
          </button>
        </div>
      </nav>

      {/* Watchlist */}
      <div className="mt-8 px-4">
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider px-4">
          Watchlist
        </h3>
        <div className="mt-3 space-y-1">
          {watchlist.map((symbol) => (
            <button
              key={symbol}
              onClick={() => setSelectedSymbol(symbol)}
              className={`w-full flex items-center justify-between px-4 py-2 text-sm rounded-lg transition-colors ${
                selectedSymbol === symbol
                  ? 'bg-gray-800 text-white'
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <span>{symbol}</span>
              <span className="text-xs text-green-400">+2.5%</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderContent = () => {
    switch (activeTab) {
      case 'market':
        return (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-lg shadow">
              <h2 className="text-xl font-bold mb-4">Market Data - {selectedSymbol}</h2>
              <p className="text-gray-600">Market chart and technical indicators will be displayed here.</p>
              <div className="mt-4 p-4 bg-blue-50 rounded">
                <p className="text-sm text-blue-800">✅ Dashboard is working! Components will be loaded next.</p>
              </div>
            </div>
          </div>
        );

      case 'agents':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4">AI Agents</h2>
            <p className="text-gray-600">Multi-agent system status and coordination will be displayed here.</p>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="p-4 bg-green-50 rounded">
                <h3 className="font-semibold text-green-800">Risk Manager</h3>
                <p className="text-sm text-green-600">Active</p>
              </div>
              <div className="p-4 bg-blue-50 rounded">
                <h3 className="font-semibold text-blue-800">Technical Indicator</h3>
                <p className="text-sm text-blue-600">Active</p>
              </div>
              <div className="p-4 bg-purple-50 rounded">
                <h3 className="font-semibold text-purple-800">Sentiment Analyst</h3>
                <p className="text-sm text-purple-600">Active</p>
              </div>
            </div>
          </div>
        );

      case 'trades':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4">Trade Execution Log</h2>
            <p className="text-gray-600">Real-time trade history and execution details will be displayed here.</p>
          </div>
        );

      case 'portfolio':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4">Portfolio Dashboard</h2>
            <p className="text-gray-600">Portfolio performance, P&L, and positions will be displayed here.</p>
            <div className="mt-4 p-4 bg-yellow-50 rounded">
              <p className="text-sm text-yellow-800">💰 ₹1000 Budget - Live Trading Mode</p>
            </div>
          </div>
        );

      case 'controls':
        return (
          <div className="bg-white p-6 rounded-lg shadow">
            <h2 className="text-xl font-bold mb-4">System Controls</h2>
            <p className="text-gray-600">Automated trading controls and emergency stops will be displayed here.</p>
            <div className="mt-4 space-y-2">
              <button type="button" className="w-full bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700">
                Start Automated Trading
              </button>
              <button type="button" className="w-full bg-red-600 text-white py-2 px-4 rounded hover:bg-red-700">
                Emergency Stop
              </button>
            </div>
          </div>
        );

      case 'config':
        return <TradingConfiguration />;

      case 'activity':
        return <LiveTradingActivity />;

      case 'monitoring':
        return <SystemMonitoring />;

      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen bg-gray-100">
      <Sidebar />

      {/* Overlay for mobile */}
      {sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="flex items-center">
              <button
                onClick={() => setSidebarOpen(true)}
                className="text-gray-500 hover:text-gray-700 lg:hidden"
              >
                <Menu size={24} />
              </button>
              <h2 className="ml-4 text-2xl font-semibold text-gray-800 capitalize">
                {activeTab === 'trades' ? 'Trade History' : activeTab}
              </h2>
            </div>

            <div className="flex items-center space-x-4">
              <div className="text-sm text-gray-600">
                <span className="font-medium">NSE</span>
                <span className={`ml-2 ${new Date().getHours() >= 9 && new Date().getHours() < 16 ? 'text-green-600' : 'text-red-600'}`}>
                  {new Date().getHours() >= 9 && new Date().getHours() < 16 ? '● OPEN' : '● CLOSED'}
                </span>
              </div>
            </div>
          </div>
        </header>

        {/* Main Content Area */}
        <main className="flex-1 overflow-y-auto p-6">
          {renderContent()}
        </main>
      </div>
    </div>
  );
};
