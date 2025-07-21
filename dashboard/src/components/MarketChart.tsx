import React, { useEffect, useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { format } from 'date-fns';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { MarketData } from '@/services/websocket';

interface MarketChartProps {
  symbol: string;
  interval?: '1m' | '5m' | '15m' | '1h' | '1d';
  showIndicators?: boolean;
}

interface ChartData {
  time: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
  sma20?: number[];
  sma50?: number[];
  ema12?: number[];
  ema26?: number[];
  macd?: number[];
  signal?: number[];
  rsi?: number[];
  bollinger_upper?: number[];
  bollinger_middle?: number[];
  bollinger_lower?: number[];
}

export const MarketChart: React.FC<MarketChartProps> = ({ 
  symbol, 
  interval = '5m',
  showIndicators = true 
}) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [currentPrice, setCurrentPrice] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);
  const chartRef = useRef<any>(null);

  useEffect(() => {
    // Load historical data
    loadHistoricalData();
    
    // Subscribe to real-time updates
    const ws = require('@/services/websocket').default;
    ws.subscribeToSymbol(symbol);
    
    const handleMarketUpdate = (data: MarketData) => {
      if (data.symbol === symbol) {
        setCurrentPrice(data);
        updateChart(data);
      }
    };
    
    ws.on('market:update', handleMarketUpdate);
    
    return () => {
      ws.off('market:update', handleMarketUpdate);
      ws.unsubscribeFromSymbol(symbol);
    };
  }, [symbol, interval]);

  const loadHistoricalData = async () => {
    try {
      setLoading(true);
      const api = require('@/services/api').default;
      const data = await api.getHistoricalData(symbol, interval);
      
      // Process and set chart data
      const processed = processChartData(data);
      setChartData(processed);
    } catch (error) {
      console.error('Failed to load chart data:', error);
    } finally {
      setLoading(false);
    }
  };

  const processChartData = (rawData: any): ChartData => {
    // Extract OHLCV data
    const chartData: ChartData = {
      time: rawData.map((d: any) => d.timestamp),
      open: rawData.map((d: any) => d.open),
      high: rawData.map((d: any) => d.high),
      low: rawData.map((d: any) => d.low),
      close: rawData.map((d: any) => d.close),
      volume: rawData.map((d: any) => d.volume),
    };

    if (showIndicators && rawData[0]?.indicators) {
      // Add technical indicators if available
      chartData.sma20 = rawData.map((d: any) => d.indicators?.sma20);
      chartData.sma50 = rawData.map((d: any) => d.indicators?.sma50);
      chartData.ema12 = rawData.map((d: any) => d.indicators?.ema12);
      chartData.ema26 = rawData.map((d: any) => d.indicators?.ema26);
      chartData.macd = rawData.map((d: any) => d.indicators?.macd);
      chartData.signal = rawData.map((d: any) => d.indicators?.signal);
      chartData.rsi = rawData.map((d: any) => d.indicators?.rsi);
      chartData.bollinger_upper = rawData.map((d: any) => d.indicators?.bollinger_upper);
      chartData.bollinger_middle = rawData.map((d: any) => d.indicators?.bollinger_middle);
      chartData.bollinger_lower = rawData.map((d: any) => d.indicators?.bollinger_lower);
    }

    return chartData;
  };

  const updateChart = (data: MarketData) => {
    if (!chartData) return;

    // Update the last candle with new price data
    const newChartData = { ...chartData };
    const lastIndex = newChartData.close.length - 1;
    
    newChartData.close[lastIndex] = data.price;
    newChartData.high[lastIndex] = Math.max(newChartData.high[lastIndex], data.price);
    newChartData.low[lastIndex] = Math.min(newChartData.low[lastIndex], data.price);
    
    setChartData(newChartData);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-96 text-gray-500">
        No data available
      </div>
    );
  }

  // Candlestick trace
  const candlestickTrace = {
    type: 'candlestick',
    x: chartData.time,
    open: chartData.open,
    high: chartData.high,
    low: chartData.low,
    close: chartData.close,
    name: 'OHLC',
    increasing: { line: { color: '#22c55e' } },
    decreasing: { line: { color: '#ef4444' } },
  };

  // Volume trace
  const volumeTrace = {
    type: 'bar',
    x: chartData.time,
    y: chartData.volume,
    name: 'Volume',
    yaxis: 'y2',
    marker: {
      color: chartData.close.map((close, i) => 
        i > 0 && close >= chartData.close[i - 1] ? '#22c55e' : '#ef4444'
      ),
    },
  };

  const traces: any[] = [candlestickTrace, volumeTrace];

  // Add indicator traces if enabled
  if (showIndicators) {
    if (chartData.sma20) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: chartData.time,
        y: chartData.sma20,
        name: 'SMA 20',
        line: { color: '#3b82f6', width: 1 },
      });
    }

    if (chartData.sma50) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: chartData.time,
        y: chartData.sma50,
        name: 'SMA 50',
        line: { color: '#f59e0b', width: 1 },
      });
    }

    if (chartData.bollinger_upper) {
      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: chartData.time,
        y: chartData.bollinger_upper,
        name: 'BB Upper',
        line: { color: '#a78bfa', width: 1, dash: 'dot' },
      });

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: chartData.time,
        y: chartData.bollinger_middle,
        name: 'BB Middle',
        line: { color: '#8b5cf6', width: 1 },
      });

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: chartData.time,
        y: chartData.bollinger_lower,
        name: 'BB Lower',
        line: { color: '#a78bfa', width: 1, dash: 'dot' },
      });
    }
  }

  const layout = {
    title: '',
    xaxis: {
      rangeslider: { visible: false },
      type: 'date',
      gridcolor: '#e5e7eb',
    },
    yaxis: {
      title: 'Price',
      side: 'right',
      gridcolor: '#e5e7eb',
    },
    yaxis2: {
      title: 'Volume',
      overlaying: 'y',
      side: 'left',
      showgrid: false,
      domain: [0, 0.2],
    },
    margin: { l: 50, r: 50, t: 50, b: 50 },
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)',
    showlegend: true,
    legend: {
      x: 0,
      y: 1,
      bgcolor: 'rgba(255,255,255,0.8)',
    },
    height: 500,
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    displaylogo: false,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-4">
          <h2 className="text-2xl font-bold">{symbol}</h2>
          {currentPrice && (
            <div className="flex items-center space-x-2">
              <span className="text-3xl font-semibold">
                ${currentPrice.price.toFixed(2)}
              </span>
              <div className={`flex items-center ${currentPrice.change >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                {currentPrice.change >= 0 ? <TrendingUp size={20} /> : <TrendingDown size={20} />}
                <span className="ml-1">
                  {currentPrice.change >= 0 ? '+' : ''}{currentPrice.change.toFixed(2)} 
                  ({currentPrice.changePercent.toFixed(2)}%)
                </span>
              </div>
            </div>
          )}
        </div>
        <div className="flex items-center space-x-2">
          <Activity className="text-gray-400" size={20} />
          <span className="text-sm text-gray-500">
            {format(new Date(), 'HH:mm:ss')}
          </span>
        </div>
      </div>

      <Plot
        data={traces}
        layout={layout}
        config={config}
        style={{ width: '100%' }}
        useResizeHandler={true}
        ref={chartRef}
      />

      {showIndicators && chartData.rsi && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">RSI</h3>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: chartData.time,
                y: chartData.rsi,
                name: 'RSI',
                line: { color: '#8b5cf6', width: 2 },
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: chartData.time,
                y: new Array(chartData.time.length).fill(70),
                name: 'Overbought',
                line: { color: '#ef4444', width: 1, dash: 'dash' },
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: chartData.time,
                y: new Array(chartData.time.length).fill(30),
                name: 'Oversold',
                line: { color: '#22c55e', width: 1, dash: 'dash' },
              },
            ]}
            layout={{
              height: 200,
              margin: { l: 50, r: 50, t: 20, b: 50 },
              xaxis: { gridcolor: '#e5e7eb' },
              yaxis: { 
                title: 'RSI',
                range: [0, 100],
                gridcolor: '#e5e7eb',
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: false,
            }}
            config={config}
            style={{ width: '100%' }}
            useResizeHandler={true}
          />
        </div>
      )}

      {showIndicators && chartData.macd && (
        <div className="mt-4">
          <h3 className="text-lg font-semibold mb-2">MACD</h3>
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: chartData.time,
                y: chartData.macd,
                name: 'MACD',
                line: { color: '#3b82f6', width: 2 },
              },
              {
                type: 'scatter',
                mode: 'lines',
                x: chartData.time,
                y: chartData.signal,
                name: 'Signal',
                line: { color: '#ef4444', width: 2 },
              },
              {
                type: 'bar',
                x: chartData.time,
                y: chartData.macd?.map((m, i) => m - (chartData.signal?.[i] || 0)),
                name: 'Histogram',
                marker: {
                  color: chartData.macd?.map((m, i) => 
                    m - (chartData.signal?.[i] || 0) >= 0 ? '#22c55e' : '#ef4444'
                  ),
                },
              },
            ]}
            layout={{
              height: 200,
              margin: { l: 50, r: 50, t: 20, b: 50 },
              xaxis: { gridcolor: '#e5e7eb' },
              yaxis: { 
                title: 'MACD',
                gridcolor: '#e5e7eb',
              },
              paper_bgcolor: 'rgba(0,0,0,0)',
              plot_bgcolor: 'rgba(0,0,0,0)',
              showlegend: true,
              legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.8)',
              },
            }}
            config={config}
            style={{ width: '100%' }}
            useResizeHandler={true}
          />
        </div>
      )}
    </div>
  );
};