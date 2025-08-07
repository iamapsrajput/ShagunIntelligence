import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Chip,
  LinearProgress,
  ToggleButton,
  ToggleButtonGroup,
  TextField,
  MenuItem
} from '@mui/material';
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line
} from 'recharts';
import { TrendingUp, TrendingDown, Remove } from '@mui/icons-material';

interface SentimentFusionChartProps {
  symbol: string;
  data: any;
  onSymbolChange: (symbol: string) => void;
}

const SentimentFusionChart: React.FC<SentimentFusionChartProps> = ({
  symbol,
  data,
  onSymbolChange
}) => {
  const [viewType, setViewType] = useState<'fusion' | 'comparison' | 'timeline'>('fusion');
  const [timelineData, setTimelineData] = useState<any[]>([]);

  const sentimentColors = {
    bullish: '#4caf50',
    neutral: '#9e9e9e',
    bearish: '#f44336'
  };

  useEffect(() => {
    // Process sentiment history for timeline view
    if (data?.sentiment_history) {
      const processed = data.sentiment_history.map((item: any) => ({
        time: new Date(item.timestamp).toLocaleTimeString(),
        sentiment: item.fused_sentiment * 100,
        confidence: item.confidence_score * 100,
        ...Object.entries(item.source_sentiments || {}).reduce((acc, [source, value]: [string, any]) => ({
          ...acc,
          [source]: value.sentiment * 100
        }), {})
      }));
      setTimelineData(processed);
    }
  }, [data]);

  const getSentimentLabel = (score: number): string => {
    if (score > 0.3) return 'Bullish';
    if (score < -0.3) return 'Bearish';
    return 'Neutral';
  };

  const getSentimentColor = (score: number): string => {
    if (score > 0.3) return sentimentColors.bullish;
    if (score < -0.3) return sentimentColors.bearish;
    return sentimentColors.neutral;
  };

  const handleViewChange = (
    event: React.MouseEvent<HTMLElement>,
    newView: 'fusion' | 'comparison' | 'timeline' | null
  ) => {
    if (newView !== null) {
      setViewType(newView);
    }
  };

  if (!data) {
    return (
      <Box p={3} textAlign="center">
        <Typography>Loading sentiment data...</Typography>
      </Box>
    );
  }

  const renderFusionView = () => {
    const consensusData = [
      { name: 'Bullish', value: data.consensus?.bullish || 0, color: sentimentColors.bullish },
      { name: 'Neutral', value: data.consensus?.neutral || 0, color: sentimentColors.neutral },
      { name: 'Bearish', value: data.consensus?.bearish || 0, color: sentimentColors.bearish }
    ];

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Fused Sentiment Score
            </Typography>
            <Box textAlign="center" my={3}>
              <Typography variant="h2" component="div" color={getSentimentColor(data.fused_sentiment)}>
                {(data.fused_sentiment * 100).toFixed(1)}
              </Typography>
              <Chip
                label={getSentimentLabel(data.fused_sentiment)}
                color={
                  data.fused_sentiment > 0.3 ? 'success' :
                  data.fused_sentiment < -0.3 ? 'error' : 'default'
                }
                sx={{ mt: 1 }}
              />
            </Box>
            <Box>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Confidence Score: {(data.confidence_score * 100).toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={data.confidence_score * 100}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Source Consensus
            </Typography>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie
                  data={consensusData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {consensusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Metadata
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Sources Used
                </Typography>
                <Typography variant="h6">
                  {data.metadata?.sources_used || 0}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  High Quality Sources
                </Typography>
                <Typography variant="h6">
                  {data.metadata?.high_quality_sources || 0}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Divergence Score
                </Typography>
                <Typography variant="h6">
                  {(data.metadata?.divergence_score || 0).toFixed(2)}
                </Typography>
              </Grid>
              <Grid item xs={6} sm={3}>
                <Typography variant="body2" color="text.secondary">
                  Quality Weighted
                </Typography>
                <Typography variant="h6" color={data.quality_weighted ? 'success.main' : 'text.secondary'}>
                  {data.quality_weighted ? 'Yes' : 'No'}
                </Typography>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  const renderComparisonView = () => {
    const comparisonData = Object.entries(data.source_sentiments || {}).map(([source, sourceData]: [string, any]) => ({
      source,
      sentiment: sourceData.sentiment * 100,
      confidence: sourceData.confidence * 100,
      quality: sourceData.quality_score * 100
    }));

    const radarData = Object.entries(data.source_sentiments || {}).map(([source, sourceData]: [string, any]) => ({
      subject: source,
      sentiment: Math.abs(sourceData.sentiment * 100),
      confidence: sourceData.confidence * 100,
      quality: sourceData.quality_score * 100
    }));

    return (
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Source Sentiment Comparison
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={comparisonData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="source" angle={-45} textAnchor="end" height={80} />
                <YAxis domain={[-100, 100]} />
                <Tooltip />
                <Legend />
                <Bar dataKey="sentiment" fill="#2196f3" name="Sentiment %" />
                <Bar dataKey="confidence" fill="#4caf50" name="Confidence %" />
              </BarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Multi-Dimensional Analysis
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid />
                <PolarAngleAxis dataKey="subject" />
                <PolarRadiusAxis angle={90} domain={[0, 100]} />
                <Radar name="Sentiment" dataKey="sentiment" stroke="#2196f3" fill="#2196f3" fillOpacity={0.6} />
                <Radar name="Confidence" dataKey="confidence" stroke="#4caf50" fill="#4caf50" fillOpacity={0.6} />
                <Radar name="Quality" dataKey="quality" stroke="#ff9800" fill="#ff9800" fillOpacity={0.6} />
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Source Details
            </Typography>
            <Grid container spacing={2}>
              {Object.entries(data.source_sentiments || {}).map(([source, sourceData]: [string, any]) => (
                <Grid item xs={12} sm={6} md={4} key={source}>
                  <Box
                    sx={{
                      p: 2,
                      border: '1px solid',
                      borderColor: 'divider',
                      borderRadius: 1
                    }}
                  >
                    <Typography variant="subtitle1" fontWeight="medium">
                      {source}
                    </Typography>
                    <Box mt={1}>
                      <Box display="flex" justifyContent="space-between" alignItems="center">
                        <Typography variant="body2" color="text.secondary">
                          Sentiment
                        </Typography>
                        <Box display="flex" alignItems="center" gap={1}>
                          <Typography
                            variant="body2"
                            color={getSentimentColor(sourceData.sentiment)}
                            fontWeight="medium"
                          >
                            {(sourceData.sentiment * 100).toFixed(1)}
                          </Typography>
                          {sourceData.sentiment > 0.1 ? <TrendingUp fontSize="small" color="success" /> :
                           sourceData.sentiment < -0.1 ? <TrendingDown fontSize="small" color="error" /> :
                           <Remove fontSize="small" color="disabled" />}
                        </Box>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Confidence
                        </Typography>
                        <Typography variant="body2">
                          {(sourceData.confidence * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                      <Box display="flex" justifyContent="space-between">
                        <Typography variant="body2" color="text.secondary">
                          Quality
                        </Typography>
                        <Typography variant="body2">
                          {(sourceData.quality_score * 100).toFixed(1)}%
                        </Typography>
                      </Box>
                    </Box>
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    );
  };

  const renderTimelineView = () => {
    return (
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Sentiment Timeline
        </Typography>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={timelineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="time" angle={-45} textAnchor="end" height={80} />
            <YAxis domain={[-100, 100]} />
            <Tooltip />
            <Legend />
            <Line
              type="monotone"
              dataKey="sentiment"
              stroke="#2196f3"
              strokeWidth={3}
              name="Fused Sentiment"
              dot={false}
            />
            {Object.keys(data.source_sentiments || {}).map((source, index) => (
              <Line
                key={source}
                type="monotone"
                dataKey={source}
                stroke={`hsl(${index * 360 / Object.keys(data.source_sentiments).length}, 70%, 50%)`}
                strokeWidth={1.5}
                name={source}
                dot={false}
                strokeDasharray="5 5"
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    );
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          Multi-Source Sentiment Analysis
        </Typography>
        <Box display="flex" gap={2}>
          <TextField
            select
            size="small"
            value={symbol}
            onChange={(e) => onSymbolChange(e.target.value)}
            sx={{ minWidth: 150 }}
          >
            <MenuItem value="RELIANCE.NS">RELIANCE.NS</MenuItem>
            <MenuItem value="TCS.NS">TCS.NS</MenuItem>
            <MenuItem value="INFY.NS">INFY.NS</MenuItem>
            <MenuItem value="HDFC.NS">HDFC.NS</MenuItem>
          </TextField>
          <ToggleButtonGroup
            value={viewType}
            exclusive
            onChange={handleViewChange}
            size="small"
          >
            <ToggleButton value="fusion">Fusion</ToggleButton>
            <ToggleButton value="comparison">Comparison</ToggleButton>
            <ToggleButton value="timeline">Timeline</ToggleButton>
          </ToggleButtonGroup>
        </Box>
      </Box>

      {viewType === 'fusion' && renderFusionView()}
      {viewType === 'comparison' && renderComparisonView()}
      {viewType === 'timeline' && renderTimelineView()}
    </Box>
  );
};

export default SentimentFusionChart;
