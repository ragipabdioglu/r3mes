/**
 * k6 Load Testing for R3MES Backend
 * 
 * Run with: k6 run tests/performance/k6_load_test.js
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const chatResponseTime = new Trend('chat_response_time');
const apiResponseTime = new Trend('api_response_time');

// Test configuration
export const options = {
  stages: [
    { duration: '2m', target: 100 },  // Ramp up to 100 users
    { duration: '5m', target: 100 }, // Stay at 100 users
    { duration: '2m', target: 200 }, // Ramp up to 200 users
    { duration: '5m', target: 200 }, // Stay at 200 users
    { duration: '2m', target: 0 },   // Ramp down to 0 users
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests should be below 500ms
    http_req_failed: ['rate<0.01'],   // Error rate should be less than 1%
    errors: ['rate<0.01'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';

// Generate random wallet address
function generateWalletAddress() {
  const chars = 'abcdefghijklmnopqrstuvwxyz0123456789';
  const randomPart = Array.from({ length: 38 }, () => chars[Math.floor(Math.random() * chars.length)]).join('');
  return `remes1${randomPart}`;
}

// Test scenarios
export default function () {
  const walletAddress = generateWalletAddress();
  
  // Test 1: Network Stats
  const statsResponse = http.get(`${BASE_URL}/network/stats`);
  check(statsResponse, {
    'network stats status is 200': (r) => r.status === 200,
    'network stats has data': (r) => JSON.parse(r.body).hasOwnProperty('total_miners'),
  });
  errorRate.add(statsResponse.status !== 200);
  apiResponseTime.add(statsResponse.timings.duration);
  
  sleep(1);
  
  // Test 2: User Info
  const userInfoResponse = http.get(`${BASE_URL}/user/info/${walletAddress}`);
  check(userInfoResponse, {
    'user info status is 200': (r) => r.status === 200,
  });
  errorRate.add(userInfoResponse.status !== 200);
  apiResponseTime.add(userInfoResponse.timings.duration);
  
  sleep(1);
  
  // Test 3: Chat Message
  const chatMessages = [
    'What is R3MES?',
    'How does mining work?',
    'Explain the protocol',
    'What is BitNet?',
    'How do I start mining?',
  ];
  
  const chatPayload = JSON.stringify({
    message: chatMessages[Math.floor(Math.random() * chatMessages.length)],
    wallet_address: walletAddress,
    adapter: 'general',
  });
  
  const chatResponse = http.post(`${BASE_URL}/chat`, chatPayload, {
    headers: { 'Content-Type': 'application/json' },
  });
  
  check(chatResponse, {
    'chat status is 200': (r) => r.status === 200,
    'chat has response': (r) => JSON.parse(r.body).hasOwnProperty('response'),
  });
  errorRate.add(chatResponse.status !== 200);
  chatResponseTime.add(chatResponse.timings.duration);
  
  sleep(2);
}

// Setup function (runs once)
export function setup() {
  console.log('🚀 Starting k6 load test for R3MES...');
  return { baseUrl: BASE_URL };
}

// Teardown function (runs once)
export function teardown(data) {
  console.log('✅ Load test completed!');
}

