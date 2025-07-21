import React, { useState, useEffect } from 'react';
import { 
  LayoutDashboard, 
  TrendingUp, 
  Brain, 
  History, 
  Wallet, 
  Settings as SettingsIcon,
  Menu,
  X,
  LogOut
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { MarketChart } from '@/components/MarketChart';
import { AgentPanel } from '@/components/AgentPanel';
import { TradeExecutionLog } from '@/components/TradeExecutionLog';
import { PortfolioDashboard } from '@/components/PortfolioDashboard';
import { SystemControls } from '@/components/SystemControls';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'market' | 'agents' | 'trades' | 'portfolio' | 'controls'>('market');
  const [selectedSymbol, setSelectedSymbol] = useState('RELIANCE');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [watchlist, setWatchlist] = useState<string[]>(['RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK']);

  useEffect(() => {
    // Load user preferences
    loadUserPreferences();
  }, []);

  const loadUserPreferences = async () => {
    try {
      const api = require('@/services/api').default;
      const preferences = await api.getWatchlist();
      setWatchlist(preferences.symbols || watchlist);
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
            <MarketChart symbol={selectedSymbol} showIndicators={true} />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <AgentPanel agentType="market" />
              <AgentPanel agentType="technical" />
            </div>
          </div>
        );

      case 'agents':
        return (
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
            <AgentPanel agentType="market" />
            <AgentPanel agentType="technical" />
            <AgentPanel agentType="sentiment" />
            <AgentPanel agentType="risk" />
            <AgentPanel agentType="coordinator" className="lg:col-span-2 xl:col-span-1" />
          </div>
        );

      case 'trades':
        return <TradeExecutionLog />;

      case 'portfolio':
        return <PortfolioDashboard />;

      case 'controls':
        return <SystemControls />;

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