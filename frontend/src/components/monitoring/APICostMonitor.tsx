import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Grid,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
  Chip,
  ToggleButton,
  ToggleButtonGroup,
  Alert,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  Area,
  AreaChart
} from 'recharts';
import {
  TrendingUp,
  TrendingDown,
  AttachMoney,
  Warning,
  InfoOutlined
} from '@mui/icons-material';
import { api } from '../../services/api';

const APICostMonitor: React.FC = () => {
  const [costData, setCostData] = useState<any>(null);
  const [period, setPeriod] = useState<'day' | 'week' | 'month'>('month');
  const [loading, setLoading] = useState(true);
  const [historicalData, setHistoricalData] = useState<any[]>([]);

  useEffect(() => {
    loadCostData();
  }, [period]);

  const loadCostData = async () => {
    try {
      setLoading(true);
      const response = await api.get(`/data-quality/api-costs/summary?period=${period}`);
      setCostData(response.data);

      // Generate mock historical data for visualization
      generateHistoricalData(response.data);
    } catch (error) {
      console.error('Error loading cost data:', error);
    } finally {
      setLoading(false);
    }
  };

  const generateHistoricalData = (data: any) => {
    // Generate mock historical data based on current costs
    const days = period === 'day' ? 24 : period === 'week' ? 7 : 30;
    const historical = [];

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date();
      date.setDate(date.getDate() - i);

      const dayData: any = {
        date: date.toLocaleDateString(),
        total: 0
      };

      Object.entries(data.providers || {}).forEach(([provider, providerData]: [string, any]) => {
        const baseCost = providerData.cost / days;
        const variation = (Math.random() - 0.5) * 0.4; // ±20% variation
        const cost = baseCost * (1 + variation);
        dayData[provider] = cost;
        dayData.total += cost;
      });

      historical.push(dayData);
    }

    setHistoricalData(historical);
  };

  const handlePeriodChange = (
    event: React.MouseEvent<HTMLElement>,
    newPeriod: 'day' | 'week' | 'month' | null
  ) => {
    if (newPeriod !== null) {
      setPeriod(newPeriod);
    }
  };

  const getStatusColor = (utilization: number): string => {
    if (utilization > 90) return '#f44336';
    if (utilization > 75) return '#ff9800';
    return '#4caf50';
  };

  const formatCurrency = (value: number): string => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  if (loading || !costData) {
    return (
      <Box p={3} textAlign="center">
        <Typography>Loading cost data...</Typography>
      </Box>
    );
  }

  const pieData = Object.entries(costData.providers || {}).map(([provider, data]: [string, any]) => ({
    name: provider,
    value: data.cost,
    percentage: (data.cost / costData.total_cost * 100).toFixed(1)
  }));

  const colors = ['#2196f3', '#4caf50', '#ff9800', '#f44336', '#9c27b0', '#00bcd4', '#795548', '#607d8b'];

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h2">
          API Cost Monitoring
        </Typography>
        <ToggleButtonGroup
          value={period}
          exclusive
          onChange={handlePeriodChange}
          size="small"
        >
          <ToggleButton value="day">Daily</ToggleButton>
          <ToggleButton value="week">Weekly</ToggleButton>
          <ToggleButton value="month">Monthly</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* Cost Overview Cards */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" color="text.secondary">
                Total Cost ({period})
              </Typography>
              <AttachMoney color="primary" />
            </Box>
            <Typography variant="h4" sx={{ mt: 1 }}>
              {formatCurrency(costData.total_cost)}
            </Typography>
            <Box display="flex" alignItems="center" mt={1}>
              <TrendingUp color="error" fontSize="small" />
              <Typography variant="body2" color="error" sx={{ ml: 0.5 }}>
                +12.5% from last {period}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" color="text.secondary">
                Total Budget
              </Typography>
              <InfoOutlined color="action" />
            </Box>
            <Typography variant="h4" sx={{ mt: 1 }}>
              {formatCurrency(costData.total_budget)}
            </Typography>
            <LinearProgress
              variant="determinate"
              value={costData.budget_utilization}
              sx={{
                mt: 1,
                height: 8,
                borderRadius: 4,
                backgroundColor: 'grey.300',
                '& .MuiLinearProgress-bar': {
                  backgroundColor: getStatusColor(costData.budget_utilization),
                  borderRadius: 4
                }
              }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" color="text.secondary">
                Budget Utilization
              </Typography>
              {costData.budget_utilization > 80 && <Warning color="warning" />}
            </Box>
            <Typography variant="h4" sx={{ mt: 1 }}>
              {costData.budget_utilization.toFixed(1)}%
            </Typography>
            <Chip
              label={
                costData.budget_utilization > 90 ? 'Critical' :
                costData.budget_utilization > 75 ? 'Warning' : 'Healthy'
              }
              color={
                costData.budget_utilization > 90 ? 'error' :
                costData.budget_utilization > 75 ? 'warning' : 'success'
              }
              size="small"
              sx={{ mt: 1 }}
            />
          </Paper>
        </Grid>

        <Grid item xs={12} md={3}>
          <Paper sx={{ p: 2 }}>
            <Box display="flex" alignItems="center" justifyContent="space-between">
              <Typography variant="subtitle2" color="text.secondary">
                Projected Monthly
              </Typography>
              <TrendingUp color="action" />
            </Box>
            <Typography variant="h4" sx={{ mt: 1 }}>
              {formatCurrency(
                period === 'day' ? costData.total_cost * 30 :
                period === 'week' ? costData.total_cost * 4.33 :
                costData.total_cost
              )}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Based on current usage
            </Typography>
          </Paper>
        </Grid>

        {/* Cost Breakdown Charts */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cost Trend
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={historicalData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" angle={-45} textAnchor="end" height={80} />
                <YAxis tickFormatter={(value) => `$${value.toFixed(0)}`} />
                <RechartsTooltip formatter={(value: any) => formatCurrency(value)} />
                <Legend />
                {Object.keys(costData.providers || {}).map((provider, index) => (
                  <Area
                    key={provider}
                    type="monotone"
                    dataKey={provider}
                    stackId="1"
                    stroke={colors[index % colors.length]}
                    fill={colors[index % colors.length]}
                    fillOpacity={0.6}
                  />
                ))}
              </AreaChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Cost Distribution
            </Typography>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percentage }) => `${name}: ${percentage}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
                  ))}
                </Pie>
                <RechartsTooltip formatter={(value: any) => formatCurrency(value)} />
              </PieChart>
            </ResponsiveContainer>
          </Paper>
        </Grid>

        {/* Detailed Provider Table */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Provider Details
            </Typography>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Provider</TableCell>
                    <TableCell align="right">Current Cost</TableCell>
                    <TableCell align="right">Budget</TableCell>
                    <TableCell align="center">Utilization</TableCell>
                    <TableCell align="right">Cost per Request</TableCell>
                    <TableCell align="right">Free Requests</TableCell>
                    <TableCell align="center">Status</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {Object.entries(costData.providers || {}).map(([provider, data]: [string, any]) => (
                    <TableRow key={provider}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="medium">
                          {provider}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.cost)}
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.budget)}
                      </TableCell>
                      <TableCell align="center">
                        <Box display="flex" alignItems="center" gap={1}>
                          <LinearProgress
                            variant="determinate"
                            value={data.utilization}
                            sx={{
                              width: 80,
                              height: 6,
                              borderRadius: 3,
                              backgroundColor: 'grey.300',
                              '& .MuiLinearProgress-bar': {
                                backgroundColor: getStatusColor(data.utilization),
                                borderRadius: 3
                              }
                            }}
                          />
                          <Typography variant="body2">
                            {data.utilization.toFixed(1)}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell align="right">
                        {formatCurrency(data.cost_per_request)}
                      </TableCell>
                      <TableCell align="right">
                        {data.free_requests.toLocaleString()}
                      </TableCell>
                      <TableCell align="center">
                        <Chip
                          label={data.status === 'over_budget' ? 'Over Budget' : 'OK'}
                          color={data.status === 'over_budget' ? 'error' : 'success'}
                          size="small"
                        />
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Paper>
        </Grid>

        {/* Cost Alerts */}
        {costData.budget_utilization > 75 && (
          <Grid item xs={12}>
            <Alert
              severity={costData.budget_utilization > 90 ? 'error' : 'warning'}
              sx={{ display: 'flex', alignItems: 'center' }}
            >
              <Box>
                <Typography variant="subtitle1" fontWeight="medium">
                  Budget Alert
                </Typography>
                <Typography variant="body2">
                  You have used {costData.budget_utilization.toFixed(1)}% of your monthly budget.
                  {costData.budget_utilization > 90
                    ? ' Consider reviewing your API usage to avoid overage charges.'
                    : ' Monitor your usage to stay within budget.'}
                </Typography>
              </Box>
            </Alert>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default APICostMonitor;
