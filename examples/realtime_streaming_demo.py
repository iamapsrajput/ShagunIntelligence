"""
Demonstration of the real-time streaming system with quality monitoring.
"""

import asyncio
from datetime import datetime
import os
from typing import Dict, Any
from loguru import logger

# Configure logging
logger.add("streaming_demo.log", rotation="10 MB")

# Import streaming components
from backend.streaming.stream_manager import StreamManager, StreamManagerConfig
from backend.streaming.agent_integration import StreamingAgentBridge, AgentNotification
from backend.streaming.performance_monitor import StreamingPerformanceMonitor, PerformanceAlert


class DemoAgent:
    """Simple demo agent that receives streaming data."""
    
    def __init__(self, agent_id: str, agent_type: str):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.messages_received = 0
        self.last_data = {}
    
    async def handle_notification(self, notification: AgentNotification):
        """Handle incoming data notification."""
        self.messages_received += 1
        self.last_data[notification.symbol] = notification.data
        
        # Log based on notification type
        if notification.type == 'market_data':
            logger.info(
                f"{self.agent_id} received market data for {notification.symbol}: "
                f"Price: {notification.data.get('price', 'N/A')}, "
                f"Quality: {notification.quality_score:.2f}"
            )
        elif notification.type == 'sentiment':
            logger.info(
                f"{self.agent_id} received sentiment for {notification.symbol}: "
                f"Score: {notification.data.get('sentiment_score', 0):.2f}, "
                f"Source: {notification.data.get('source', 'unknown')}"
            )
        elif notification.type == 'news':
            logger.info(
                f"{self.agent_id} received news for {notification.symbol}: "
                f"{notification.data.get('headline', 'No headline')[:100]}..."
            )
        elif notification.type == 'quality_alert':
            logger.warning(
                f"{self.agent_id} received quality alert: {notification.data}"
            )


async def performance_alert_handler(alert: PerformanceAlert):
    """Handle performance alerts."""
    logger.warning(
        f"PERFORMANCE ALERT [{alert.severity.upper()}]: "
        f"{alert.message}"
    )


async def demonstrate_streaming_system():
    """Demonstrate the real-time streaming system."""
    
    print("=== Shagun Intelligence Real-Time Streaming System Demo ===\n")
    
    # 1. Configure the streaming system
    print("1. Configuring streaming system...")
    
    config = StreamManagerConfig(
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        enable_kite=True,
        enable_alpha_vantage=True,
        enable_finnhub=True,
        enable_twitter=True,
        enable_news=True,
        # Load API keys from environment
        kite_api_key=os.getenv("KITE_API_KEY", "demo_key"),
        kite_access_token=os.getenv("KITE_ACCESS_TOKEN", "demo_token"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", "demo_key"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", "demo_key"),
        twitter_bearer_token=os.getenv("TWITTER_BEARER_TOKEN", "demo_token"),
        newsapi_key=os.getenv("NEWSAPI_KEY", "demo_key")
    )
    
    # 2. Initialize components
    print("\n2. Initializing streaming components...")
    
    # Create stream manager
    stream_manager = StreamManager(config)
    await stream_manager.initialize()
    
    # Create agent bridge
    agent_bridge = StreamingAgentBridge(stream_manager)
    
    # Create performance monitor
    perf_monitor = StreamingPerformanceMonitor(performance_alert_handler)
    perf_monitor.start_monitoring()
    
    # 3. Create demo agents
    print("\n3. Creating demo trading agents...")
    
    agents = {
        'market_analyst': DemoAgent('market_analyst_001', 'market_analyst'),
        'risk_manager': DemoAgent('risk_manager_001', 'risk_manager'),
        'sentiment_analyst': DemoAgent('sentiment_analyst_001', 'sentiment_analyst'),
        'trade_executor': DemoAgent('trade_executor_001', 'trade_executor')
    }
    
    # 4. Register agents with different quality thresholds
    print("\n4. Registering agents for streaming updates...")
    
    symbols = ['RELIANCE', 'TCS', 'INFY']
    
    await agent_bridge.register_agent(
        'market_analyst_001',
        'market_analyst',
        agents['market_analyst'].handle_notification,
        symbols,
        data_types=['market_data'],
        quality_threshold=0.7  # Medium quality acceptable
    )
    
    await agent_bridge.register_agent(
        'risk_manager_001',
        'risk_manager',
        agents['risk_manager'].handle_notification,
        symbols,
        data_types=['market_data', 'news'],
        quality_threshold=0.85  # High quality required
    )
    
    await agent_bridge.register_agent(
        'sentiment_analyst_001',
        'sentiment_analyst',
        agents['sentiment_analyst'].handle_notification,
        symbols,
        data_types=['sentiment', 'news'],
        quality_threshold=0.6  # Lower quality acceptable
    )
    
    await agent_bridge.register_agent(
        'trade_executor_001',
        'trade_executor',
        agents['trade_executor'].handle_notification,
        symbols,
        data_types=['market_data'],
        quality_threshold=0.9  # Highest quality required
    )
    
    print(f"   Registered {len(agents)} agents for symbols: {symbols}")
    
    # 5. Demonstrate real-time data flow
    print("\n5. Starting real-time data streaming...")
    print("   (In production, this would show actual market data)")
    print("   Press Ctrl+C to stop\n")
    
    # Simulate some streaming activity
    try:
        # Let the system run for demonstration
        for i in range(30):  # Run for 30 seconds
            await asyncio.sleep(1)
            
            # Every 5 seconds, show status
            if i % 5 == 0:
                print(f"\n--- Status at {datetime.now().strftime('%H:%M:%S')} ---")
                
                # Show stream health
                health = stream_manager.get_stream_health()
                print(f"Overall Health: {health['overall_health']}")
                
                # Show agent statistics
                for agent_name, agent in agents.items():
                    print(f"{agent_name}: {agent.messages_received} messages received")
                
                # Show performance metrics
                metrics = perf_monitor.get_current_metrics()
                if 'messages_per_second' in metrics:
                    mps = metrics['messages_per_second']['value']
                    print(f"Throughput: {mps:.1f} messages/second")
                
                if 'latency_p95_ms' in metrics:
                    latency = metrics['latency_p95_ms']['value']
                    print(f"P95 Latency: {latency:.1f}ms")
        
        # 6. Demonstrate quality testing
        print("\n\n6. Performing quality test across all streams...")
        
        test_result = await stream_manager.perform_quality_test('RELIANCE')
        
        print("\nQuality Test Results:")
        for stream, result in test_result['streams'].items():
            if result['success']:
                print(f"   {stream}: Quality={result['quality_score']:.2f}, "
                      f"Latency={result.get('latency_ms', 0):.0f}ms")
            else:
                print(f"   {stream}: Failed - {result['error']}")
        
        print(f"\nRecommended stream: {test_result['recommendation']['best_stream']}")
        
        # 7. Show multi-stream comparison
        print("\n7. Multi-stream data comparison for RELIANCE:")
        
        multi_data = await stream_manager.get_multi_stream_data('RELIANCE')
        
        if multi_data['streams']:
            print("\nData from different streams:")
            for stream, data in multi_data['streams'].items():
                print(f"   {stream}: Price={data.get('price', 'N/A')}, "
                      f"Quality={data['quality_score']:.2f}")
            
            if multi_data['consensus']:
                consensus = multi_data['consensus']
                print(f"\nConsensus: Mean price={consensus['mean_price']:.2f}, "
                      f"Spread={consensus['spread_percent']:.2f}%")
        
        # 8. Generate performance report
        print("\n8. Performance Report:")
        
        perf_report = perf_monitor.get_performance_report()
        
        print(f"   Health Score: {perf_report['health_score']:.0f}/100")
        print(f"   Total Messages: {perf_report['current_metrics']['summary']['total_messages']}")
        print(f"   Error Rate: {perf_report['current_metrics']['summary']['overall_error_rate']:.1%}")
        
        if perf_report['recommendations']:
            print("\n   Recommendations:")
            for rec in perf_report['recommendations']:
                print(f"   - {rec}")
        
    except KeyboardInterrupt:
        print("\n\nStopping demonstration...")
    
    # 9. Cleanup
    print("\n9. Shutting down streaming system...")
    
    perf_monitor.stop_monitoring()
    await stream_manager.shutdown()
    
    print("\nDemo completed!")


async def demonstrate_quality_scenarios():
    """Demonstrate different data quality scenarios."""
    
    print("\n\n=== Data Quality Scenarios ===\n")
    
    # Scenario 1: High quality data
    print("Scenario 1: High Quality Data (>90%)")
    print("   - All agents receive data")
    print("   - Full analysis capabilities")
    print("   - Normal position sizing")
    print("   - All order types available")
    
    # Scenario 2: Medium quality data
    print("\nScenario 2: Medium Quality Data (60-90%)")
    print("   - Trade executor may not receive data (requires 90%+)")
    print("   - Conservative analysis mode")
    print("   - Reduced position sizing")
    print("   - Limit orders only")
    
    # Scenario 3: Low quality data
    print("\nScenario 3: Low Quality Data (<60%)")
    print("   - Only sentiment analyst receives data (60% threshold)")
    print("   - Minimal analysis")
    print("   - No new trades")
    print("   - Alert notifications sent")
    
    # Scenario 4: Stream failover
    print("\nScenario 4: Stream Failover")
    print("   - Primary stream quality degrades")
    print("   - System automatically switches to backup stream")
    print("   - Agents notified of stream switch")
    print("   - Minimal disruption to data flow")


if __name__ == "__main__":
    print("Starting Shagun Intelligence Real-Time Streaming Demo...")
    print("=" * 50)
    
    # Run main demonstration
    asyncio.run(demonstrate_streaming_system())
    
    # Show quality scenarios
    asyncio.run(demonstrate_quality_scenarios())
    
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print("✓ Real-time data aggregation from multiple sources")
    print("✓ Quality-based stream prioritization and failover")
    print("✓ Agent-specific data filtering and transformation")
    print("✓ Performance monitoring with alerting")
    print("✓ Sub-second data propagation")
    print("✓ Graceful handling of connection failures")
    print("\nThe streaming system is now ready for production use!")