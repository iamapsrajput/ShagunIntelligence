"""Social media monitoring module for Twitter/X and other platforms."""

import aiohttp
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from loguru import logger
import re
from collections import Counter
import hashlib


class SocialMediaMonitor:
    """Monitor social media platforms for stock-related sentiment."""
    
    def __init__(self):
        """Initialize the social media monitor."""
        # API configurations (keys should be set via environment variables)
        self.apis = {
            'twitter': {
                'bearer_token': None,  # Set via environment
                'base_url': 'https://api.twitter.com/2'
            },
            'reddit': {
                'client_id': None,  # Set via environment
                'client_secret': None,
                'user_agent': 'AlgoHive/1.0',
                'base_url': 'https://oauth.reddit.com'
            },
            'stocktwits': {
                'base_url': 'https://api.stocktwits.com/api/2'
            }
        }
        
        # Platform weights for sentiment calculation
        self.platform_weights = {
            'twitter': 1.0,
            'stocktwits': 0.8,
            'reddit': 0.7
        }
        
        # Influence scores for verified/popular accounts
        self.influence_thresholds = {
            'followers': 10000,
            'engagement_rate': 0.05
        }
        
        self.session = None
        self._rate_limits = {}
        logger.info("SocialMediaMonitor initialized")

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get_symbol_mentions(
        self,
        symbol: str,
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Get social media mentions for a specific symbol.
        
        Args:
            symbol: Stock symbol
            lookback_hours: Hours of historical data
            
        Returns:
            List of social media posts with metadata
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            mentions = []
            
            # Gather from multiple platforms
            tasks = []
            
            if self.apis['twitter']['bearer_token']:
                tasks.append(self._get_twitter_mentions(symbol, lookback_hours))
            
            if self.apis['reddit']['client_id']:
                tasks.append(self._get_reddit_mentions(symbol, lookback_hours))
            
            # StockTwits doesn't require auth
            tasks.append(self._get_stocktwits_mentions(symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    mentions.extend(result)
                elif isinstance(result, Exception):
                    logger.error(f"Social media fetch error: {result}")
            
            # Sort by engagement and timestamp
            mentions.sort(
                key=lambda x: (x.get('engagement_score', 0), x['timestamp']),
                reverse=True
            )
            
            logger.info(f"Found {len(mentions)} social mentions for {symbol}")
            return mentions
            
        except Exception as e:
            logger.error(f"Error getting social mentions for {symbol}: {str(e)}")
            return []

    async def get_trending_topics(self) -> List[Dict[str, Any]]:
        """
        Get trending financial topics across social media.
        
        Returns:
            List of trending topics with metadata
        """
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            trending = []
            
            # Get trending from different platforms
            tasks = []
            
            if self.apis['twitter']['bearer_token']:
                tasks.append(self._get_twitter_trending())
            
            tasks.append(self._get_reddit_trending())
            tasks.append(self._get_stocktwits_trending())
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Aggregate trending topics
            all_topics = []
            for result in results:
                if isinstance(result, list):
                    all_topics.extend(result)
            
            # Count occurrences across platforms
            topic_counts = Counter()
            for topic in all_topics:
                topic_counts[topic['topic']] += topic.get('volume', 1)
            
            # Create trending list
            for topic, volume in topic_counts.most_common(20):
                trending.append({
                    'topic': topic,
                    'volume': volume,
                    'platforms': [t['platform'] for t in all_topics if t['topic'] == topic],
                    'timestamp': datetime.now()
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting trending topics: {str(e)}")
            return []

    async def _get_twitter_mentions(
        self,
        symbol: str,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Get mentions from Twitter/X."""
        try:
            if not self.apis['twitter']['bearer_token']:
                return []
            
            # Build search query
            query = f"${symbol} OR #{symbol} -is:retweet lang:en"
            
            # Calculate time range
            start_time = (datetime.now() - timedelta(hours=lookback_hours)).isoformat() + 'Z'
            
            headers = {
                'Authorization': f"Bearer {self.apis['twitter']['bearer_token']}"
            }
            
            params = {
                'query': query,
                'start_time': start_time,
                'max_results': 100,
                'tweet.fields': 'created_at,public_metrics,author_id',
                'user.fields': 'verified,public_metrics'
            }
            
            url = f"{self.apis['twitter']['base_url']}/tweets/search/recent"
            
            async with self.session.get(url, headers=headers, params=params) as response:
                if response.status == 429:  # Rate limited
                    logger.warning("Twitter API rate limited")
                    return []
                
                data = await response.json()
            
            mentions = []
            for tweet in data.get('data', []):
                metrics = tweet.get('public_metrics', {})
                
                mention = {
                    'id': tweet['id'],
                    'text': tweet['text'],
                    'platform': 'twitter',
                    'timestamp': datetime.fromisoformat(
                        tweet['created_at'].replace('Z', '+00:00')
                    ),
                    'author_id': tweet.get('author_id'),
                    'engagement_score': self._calculate_engagement_score(metrics),
                    'metrics': metrics,
                    'url': f"https://twitter.com/i/web/status/{tweet['id']}",
                    'source': 'twitter'
                }
                
                mentions.append(mention)
            
            return mentions
            
        except Exception as e:
            logger.error(f"Error fetching Twitter mentions: {str(e)}")
            return []

    async def _get_reddit_mentions(
        self,
        symbol: str,
        lookback_hours: int
    ) -> List[Dict[str, Any]]:
        """Get mentions from Reddit."""
        try:
            if not self.apis['reddit']['client_id']:
                return []
            
            # Get access token
            auth = aiohttp.BasicAuth(
                self.apis['reddit']['client_id'],
                self.apis['reddit']['client_secret']
            )
            
            token_data = {
                'grant_type': 'client_credentials'
            }
            
            async with self.session.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=token_data
            ) as response:
                token_info = await response.json()
            
            headers = {
                'Authorization': f"Bearer {token_info['access_token']}",
                'User-Agent': self.apis['reddit']['user_agent']
            }
            
            # Search relevant subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing', 'StockMarket']
            mentions = []
            
            for subreddit in subreddits:
                url = f"{self.apis['reddit']['base_url']}/r/{subreddit}/search"
                params = {
                    'q': symbol,
                    'sort': 'new',
                    'time': 'day' if lookback_hours <= 24 else 'week',
                    'limit': 25
                }
                
                async with self.session.get(url, headers=headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post in data.get('data', {}).get('children', []):
                            post_data = post['data']
                            
                            # Check if post is within time range
                            created_time = datetime.fromtimestamp(post_data['created_utc'])
                            if created_time < datetime.now() - timedelta(hours=lookback_hours):
                                continue
                            
                            mention = {
                                'id': post_data['id'],
                                'text': f"{post_data['title']} {post_data.get('selftext', '')}",
                                'platform': 'reddit',
                                'timestamp': created_time,
                                'author': post_data['author'],
                                'engagement_score': self._calculate_reddit_engagement(post_data),
                                'metrics': {
                                    'upvotes': post_data['ups'],
                                    'comments': post_data['num_comments'],
                                    'upvote_ratio': post_data.get('upvote_ratio', 0)
                                },
                                'url': f"https://reddit.com{post_data['permalink']}",
                                'subreddit': subreddit,
                                'source': 'reddit'
                            }
                            
                            mentions.append(mention)
            
            return mentions
            
        except Exception as e:
            logger.error(f"Error fetching Reddit mentions: {str(e)}")
            return []

    async def _get_stocktwits_mentions(self, symbol: str) -> List[Dict[str, Any]]:
        """Get mentions from StockTwits."""
        try:
            url = f"{self.apis['stocktwits']['base_url']}/streams/symbol/{symbol}.json"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
            
            mentions = []
            for message in data.get('messages', []):
                mention = {
                    'id': str(message['id']),
                    'text': message['body'],
                    'platform': 'stocktwits',
                    'timestamp': datetime.fromisoformat(
                        message['created_at'].replace('Z', '+00:00')
                    ),
                    'author': message['user']['username'],
                    'engagement_score': self._calculate_stocktwits_engagement(message),
                    'metrics': {
                        'likes': message.get('likes', {}).get('total', 0),
                        'sentiment': message.get('entities', {}).get('sentiment', {}).get('basic')
                    },
                    'url': message.get('permalink'),
                    'source': 'stocktwits'
                }
                
                mentions.append(mention)
            
            return mentions
            
        except Exception as e:
            logger.error(f"Error fetching StockTwits mentions: {str(e)}")
            return []

    async def _get_twitter_trending(self) -> List[Dict[str, Any]]:
        """Get trending topics from Twitter."""
        # Placeholder for Twitter trending topics
        # Would require Twitter API v2 trends endpoint
        return []

    async def _get_reddit_trending(self) -> List[Dict[str, Any]]:
        """Get trending topics from Reddit finance subreddits."""
        try:
            trending = []
            
            # Get hot posts from finance subreddits
            subreddits = ['wallstreetbets', 'stocks', 'investing']
            
            for subreddit in subreddits:
                url = f"https://www.reddit.com/r/{subreddit}/hot.json"
                params = {'limit': 10}
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        for post in data.get('data', {}).get('children', [])[:5]:
                            post_data = post['data']
                            
                            # Extract stock symbols from title
                            symbols = self._extract_symbols(post_data['title'])
                            
                            for symbol in symbols:
                                trending.append({
                                    'topic': symbol,
                                    'platform': 'reddit',
                                    'volume': post_data['ups'] + post_data['num_comments'],
                                    'source_url': f"https://reddit.com{post_data['permalink']}"
                                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting Reddit trending: {str(e)}")
            return []

    async def _get_stocktwits_trending(self) -> List[Dict[str, Any]]:
        """Get trending stocks from StockTwits."""
        try:
            url = f"{self.apis['stocktwits']['base_url']}/trending/symbols.json"
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    return []
                
                data = await response.json()
            
            trending = []
            for symbol in data.get('symbols', []):
                trending.append({
                    'topic': symbol['symbol'],
                    'platform': 'stocktwits',
                    'volume': 1,  # StockTwits doesn't provide volume
                    'name': symbol.get('title', '')
                })
            
            return trending
            
        except Exception as e:
            logger.error(f"Error getting StockTwits trending: {str(e)}")
            return []

    def _calculate_engagement_score(self, metrics: Dict[str, int]) -> float:
        """Calculate engagement score for Twitter posts."""
        likes = metrics.get('like_count', 0)
        retweets = metrics.get('retweet_count', 0)
        replies = metrics.get('reply_count', 0)
        quotes = metrics.get('quote_count', 0)
        
        # Weighted engagement score
        score = (likes * 1.0 + retweets * 2.0 + replies * 1.5 + quotes * 2.5)
        
        # Normalize to 0-1000 scale
        return min(score, 1000)

    def _calculate_reddit_engagement(self, post_data: Dict[str, Any]) -> float:
        """Calculate engagement score for Reddit posts."""
        upvotes = post_data.get('ups', 0)
        comments = post_data.get('num_comments', 0)
        ratio = post_data.get('upvote_ratio', 0.5)
        
        # Weighted score considering upvote ratio
        score = (upvotes * ratio) + (comments * 2.0)
        
        # Normalize
        return min(score, 1000)

    def _calculate_stocktwits_engagement(self, message: Dict[str, Any]) -> float:
        """Calculate engagement score for StockTwits messages."""
        likes = message.get('likes', {}).get('total', 0)
        
        # Check if from verified user
        if message.get('user', {}).get('official', False):
            likes *= 2
        
        return min(likes * 10, 1000)

    def _extract_symbols(self, text: str) -> List[str]:
        """Extract stock symbols from text."""
        # Look for $SYMBOL or common stock symbols
        dollar_symbols = re.findall(r'\$([A-Z]{1,5})\b', text)
        
        # Also look for common symbols without $
        common_symbols = ['GME', 'AMC', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
        text_upper = text.upper()
        found_symbols = [s for s in common_symbols if f' {s} ' in f' {text_upper} ']
        
        return list(set(dollar_symbols + found_symbols))

    def get_status(self) -> Dict[str, Any]:
        """Get the status of the social media monitor."""
        return {
            'status': 'active',
            'configured_platforms': sum(
                1 for platform, config in self.apis.items()
                if any(v for k, v in config.items() if k != 'base_url' and v)
            ),
            'session_active': self.session is not None and not self.session.closed,
            'platforms': list(self.apis.keys())
        }