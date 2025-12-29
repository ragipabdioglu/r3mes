"""
Load Testing with Locust

Tests backend API under various load conditions.
"""

from locust import HttpUser, task, between, events
import json
import random
import string


class APIUser(HttpUser):
    """Simulated API user for load testing."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Setup user session."""
        # Generate random wallet address for testing
        self.wallet_address = self._generate_wallet_address()
        self.api_key = None  # Can be set for authenticated requests
    
    def _generate_wallet_address(self) -> str:
        """Generate random wallet address."""
        prefix = "remes1"
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=39))
        return f"{prefix}{suffix}"
    
    @task(3)
    def get_network_stats(self):
        """GET /network/stats - High frequency."""
        self.client.get("/network/stats", name="network_stats")
    
    @task(2)
    def get_user_info(self):
        """GET /user/info/{address} - Medium frequency."""
        self.client.get(f"/user/info/{self.wallet_address}", name="user_info")
    
    @task(2)
    def get_blocks(self):
        """GET /blocks - Medium frequency."""
        params = {"limit": 10, "offset": 0}
        self.client.get("/blocks", params=params, name="blocks")
    
    @task(1)
    def post_chat(self):
        """POST /chat - Low frequency (expensive operation)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        data = {
            "message": "Test message for load testing",
            "wallet_address": self.wallet_address
        }
        self.client.post("/chat", json=data, headers=headers, name="chat")
    
    @task(1)
    def get_miner_stats(self):
        """GET /miner/stats/{address} - Low frequency."""
        self.client.get(f"/miner/stats/{self.wallet_address}", name="miner_stats")
    
    @task(1)
    def get_leaderboard(self):
        """GET /leaderboard/miners - Low frequency."""
        params = {"limit": 100}
        self.client.get("/leaderboard/miners", params=params, name="leaderboard")


class HighLoadUser(APIUser):
    """High load user - more aggressive."""
    
    wait_time = between(0.5, 1.5)
    
    @task(5)
    def get_network_stats(self):
        """High frequency network stats."""
        self.client.get("/network/stats", name="network_stats_high_load")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when test starts."""
    print("Load test started")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when test stops."""
    print("Load test completed")


# Load test scenarios
class NormalLoad(HttpUser):
    """Normal load scenario: 100 concurrent users."""
    tasks = [APIUser]
    wait_time = between(1, 3)


class HighLoad(HttpUser):
    """High load scenario: 1000 concurrent users."""
    tasks = [HighLoadUser]
    wait_time = between(0.5, 1.5)


class StressTest(HttpUser):
    """Stress test scenario: 2000+ concurrent users."""
    tasks = [HighLoadUser]
    wait_time = between(0.1, 0.5)
