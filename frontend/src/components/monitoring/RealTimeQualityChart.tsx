import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  ToggleButton,
  ToggleButtonGroup,
  Paper
} from '@mui/material';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts';
import { format } from 'date-fns';

interface RealTimeQualityChartProps {
  symbol: string;
  data: any;
}

const RealTimeQualityChart: React.FC<RealTimeQualityChartProps> = ({
  symbol,
  data
}) => {
  const [chartType, setChartType] = useState<'quality' | 'latency' | 'consistency'>('quality');
  const [timeRange, setTimeRange] = useState<'5m' | '30m' | '1h' | '4h'>('30m');
  const [chartData, setChartData] = useState<any[]>([]);

  useEffect(() => {
    // Process quality history data for charting
    if (data?.quality_history) {
      const processed = data.quality_history.map((item: any) => ({
        timestamp: new Date(item.timestamp).getTime(),
        time: format(new Date(item.timestamp), 'HH:mm:ss'),
        overall: item.overall_score * 100,
        ...Object.entries(item.sources || {}).reduce((acc, [source, sourceData]: [string, any]) => ({
          ...acc,
          [`${source}_score`]: sourceData.quality_score * 100,
          [`${source}_latency`]: sourceData.latency_ms,
          [`${source}_consistency`]: sourceData.consistency_score * 100
        }), {})
      }));
      setChartData(processed);
    }
  }, [data]);

  const handleChartTypeChange = (
    event: React.MouseEvent<HTMLElement>,
    newType: 'quality' | 'latency' | 'consistency' | null
  ) => {
    if (newType !== null) {
      setChartType(newType);
    }
  };

  const handleTimeRangeChange = (
    event: React.MouseEvent<HTMLElement>,
    newRange: '5m' | '30m' | '1h' | '4h' | null
  ) => {
    if (newRange !== null) {
      setTimeRange(newRange);
    }
  };

  const getFilteredData = () => {
    if (!chartData.length) return [];

    const now = Date.now();
    const ranges = {
      '5m': 5 * 60 * 1000,
      '30m': 30 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000
    };

    const cutoff = now - ranges[timeRange];
    return chartData.filter(item => item.timestamp >= cutoff);
  };

  const renderChart = () => {
    const filteredData = getFilteredData();
    const sources = Object.keys(data?.sources || {});

    if (chartType === 'quality') {
      return (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={80} stroke="#4caf50" strokeDasharray="5 5" label="High Quality" />
            <ReferenceLine y={60} stroke="#ff9800" strokeDasharray="5 5" label="Medium Quality" />
            <Line
              type="monotone"
              dataKey="overall"
              stroke="#2196f3"
              strokeWidth={3}
              name="Overall Quality"
              dot={false}
            />
            {sources.map((source, index) => (
              <Line
                key={source}
                type="monotone"
                dataKey={`${source}_score`}
                stroke={`hsl(${index * 360 / sources.length}, 70%, 50%)`}
                strokeWidth={1.5}
                name={source}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    } else if (chartType === 'latency') {
      return (
        <ResponsiveContainer width="100%" height={400}>
          <AreaChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis />
            <Tooltip />
            <Legend />
            <ReferenceLine y={1000} stroke="#f44336" strokeDasharray="5 5" label="High Latency" />
            {sources.map((source, index) => (
              <Area
                key={source}
                type="monotone"
                dataKey={`${source}_latency`}
                stackId="1"
                stroke={`hsl(${index * 360 / sources.length}, 70%, 50%)`}
                fill={`hsl(${index * 360 / sources.length}, 70%, 50%)`}
                name={source}
              />
            ))}
          </AreaChart>
        </ResponsiveContainer>
      );
    } else {
      return (
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="time" 
              angle={-45}
              textAnchor="end"
              height={60}
            />
            <YAxis domain={[0, 100]} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={90} stroke="#4caf50" strokeDasharray="5 5" label="High Consistency" />
            {sources.map((source, index) => (
              <Line
                key={source}
                type="monotone"
                dataKey={`${source}_consistency`}
                stroke={`hsl(${index * 360 / sources.length}, 70%, 50%)`}
                strokeWidth={2}
                name={source}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      );
    }
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Real-Time Quality Monitoring
        </Typography>
        <Box display="flex" gap={2}>
          <ToggleButtonGroup
            value={chartType}
            exclusive
            onChange={handleChartTypeChange}
            size="small"
          >
            <ToggleButton value="quality">Quality Score</ToggleButton>
            <ToggleButton value="latency">Latency</ToggleButton>
            <ToggleButton value="consistency">Consistency</ToggleButton>
          </ToggleButtonGroup>
          <ToggleButtonGroup
            value={timeRange}
            exclusive
            onChange={handleTimeRangeChange}
            size="small"
          >
            <ToggleButton value="5m">5m</ToggleButton>
            <ToggleButton value="30m">30m</ToggleButton>
            <ToggleButton value="1h">1h</ToggleButton>
            <ToggleButton value="4h">4h</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      <Paper elevation={0} sx={{ p: 2, backgroundColor: 'background.default' }}>
        {chartData.length > 0 ? (
          renderChart()
        ) : (
          <Box 
            height={400} 
            display="flex" 
            alignItems="center" 
            justifyContent="center"
          >
            <Typography color="text.secondary">
              No data available for the selected time range
            </Typography>
          </Box>
        )}
      </Paper>

      <Box mt={2} p={2} backgroundColor="background.default" borderRadius={1}>
        <Typography variant="body2" color="text.secondary">
          <strong>Chart Guide:</strong>
        </Typography>
        <Box display="flex" gap={3} mt={1}>
          <Typography variant="body2" color="text.secondary">
            • Quality Score: Higher is better (80%+ = High Quality)
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Latency: Lower is better (&lt;1000ms = Good)
          </Typography>
          <Typography variant="body2" color="text.secondary">
            • Consistency: Higher is better (90%+ = Very Consistent)
          </Typography>
        </Box>
      </Box>
    </Box>
  );
};

export default RealTimeQualityChart;