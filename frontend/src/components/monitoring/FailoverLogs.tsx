import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Paper,
  Chip,
  IconButton,
  TextField,
  InputAdornment,
  Tooltip
} from '@mui/material';
import {
  SwapHoriz,
  Error as ErrorIcon,
  CheckCircle,
  Search,
  Refresh,
  Timeline
} from '@mui/icons-material';
import { format } from 'date-fns';
import { api } from '../../services/api';

interface FailoverEvent {
  id: string;
  timestamp: string;
  symbol: string;
  from_source: string;
  to_source: string;
  reason: string;
  success: boolean;
  duration_ms: number;
  quality_before: number;
  quality_after: number;
  metadata?: any;
}

interface FailoverLogsProps {
  showFull?: boolean;
}

const FailoverLogs: React.FC<FailoverLogsProps> = ({ showFull = false }) => {
  const [events, setEvents] = useState<FailoverEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(showFull ? 25 : 10);
  const [searchTerm, setSearchTerm] = useState('');

  useEffect(() => {
    loadFailoverLogs();
  }, []);

  const loadFailoverLogs = async () => {
    try {
      setLoading(true);
      const response = await api.get('/data-quality/failover-logs', {
        params: {
          limit: showFull ? 1000 : 100
        }
      });
      setEvents(response.data);
    } catch (error) {
      console.error('Error loading failover logs:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const getReasonChip = (reason: string) => {
    const chipProps: any = {
      label: reason,
      size: 'small' as const
    };

    if (reason.toLowerCase().includes('timeout')) {
      chipProps.color = 'error';
      chipProps.icon = <ErrorIcon />;
    } else if (reason.toLowerCase().includes('quality')) {
      chipProps.color = 'warning';
    } else if (reason.toLowerCase().includes('rate limit')) {
      chipProps.color = 'secondary';
    } else {
      chipProps.color = 'default';
    }

    return <Chip {...chipProps} />;
  };

  const getQualityChange = (before: number, after: number) => {
    const change = ((after - before) / before) * 100;
    const improved = change > 0;

    return (
      <Box display="flex" alignItems="center" gap={0.5}>
        <Typography
          variant="body2"
          color={improved ? 'success.main' : 'error.main'}
        >
          {improved ? '+' : ''}{change.toFixed(1)}%
        </Typography>
        {improved ? (
          <CheckCircle fontSize="small" color="success" />
        ) : (
          <ErrorIcon fontSize="small" color="error" />
        )}
      </Box>
    );
  };

  const filteredEvents = events.filter(event => {
    if (!searchTerm) return true;
    const searchLower = searchTerm.toLowerCase();
    return (
      event.symbol.toLowerCase().includes(searchLower) ||
      event.from_source.toLowerCase().includes(searchLower) ||
      event.to_source.toLowerCase().includes(searchLower) ||
      event.reason.toLowerCase().includes(searchLower)
    );
  });

  const paginatedEvents = filteredEvents.slice(
    page * rowsPerPage,
    page * rowsPerPage + rowsPerPage
  );

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
        <Typography variant={showFull ? "h5" : "h6"} component="h2">
          Failover Event Logs
        </Typography>
        <Box display="flex" gap={1} alignItems="center">
          {showFull && (
            <TextField
              size="small"
              placeholder="Search logs..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              InputProps={{
                startAdornment: (
                  <InputAdornment position="start">
                    <Search />
                  </InputAdornment>
                ),
              }}
            />
          )}
          <IconButton size="small" onClick={loadFailoverLogs}>
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      <TableContainer component={Paper} elevation={0}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Timestamp</TableCell>
              <TableCell>Symbol</TableCell>
              <TableCell>Failover</TableCell>
              <TableCell>Reason</TableCell>
              <TableCell align="center">Success</TableCell>
              <TableCell align="right">Duration</TableCell>
              <TableCell align="center">Quality Impact</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  Loading...
                </TableCell>
              </TableRow>
            ) : paginatedEvents.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} align="center">
                  No failover events found
                </TableCell>
              </TableRow>
            ) : (
              paginatedEvents.map((event) => (
                <TableRow key={event.id}>
                  <TableCell>
                    <Tooltip title={format(new Date(event.timestamp), 'PPpp')}>
                      <Typography variant="body2">
                        {format(new Date(event.timestamp), 'MMM dd, HH:mm:ss')}
                      </Typography>
                    </Tooltip>
                  </TableCell>
                  <TableCell>
                    <Chip
                      label={event.symbol}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </TableCell>
                  <TableCell>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Typography variant="body2">
                        {event.from_source}
                      </Typography>
                      <SwapHoriz fontSize="small" color="action" />
                      <Typography variant="body2" fontWeight="medium">
                        {event.to_source}
                      </Typography>
                    </Box>
                  </TableCell>
                  <TableCell>
                    {getReasonChip(event.reason)}
                  </TableCell>
                  <TableCell align="center">
                    {event.success ? (
                      <CheckCircle color="success" fontSize="small" />
                    ) : (
                      <ErrorIcon color="error" fontSize="small" />
                    )}
                  </TableCell>
                  <TableCell align="right">
                    <Typography variant="body2">
                      {event.duration_ms}ms
                    </Typography>
                  </TableCell>
                  <TableCell align="center">
                    {getQualityChange(event.quality_before, event.quality_after)}
                  </TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </TableContainer>

      {showFull && (
        <TablePagination
          rowsPerPageOptions={[10, 25, 50, 100]}
          component="div"
          count={filteredEvents.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      )}

      {!showFull && events.length > rowsPerPage && (
        <Box mt={2} p={2} bgcolor="background.default" borderRadius={1}>
          <Box display="flex" alignItems="center" gap={1}>
            <Timeline color="action" />
            <Typography variant="body2" color="text.secondary">
              Showing {Math.min(rowsPerPage, events.length)} of {events.length} failover events
            </Typography>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default FailoverLogs;
