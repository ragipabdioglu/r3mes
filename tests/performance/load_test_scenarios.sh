#!/bin/bash
# Load Test Scenarios Script

set -e

BASE_URL=${1:-"http://localhost:8000"}
SCENARIO=${2:-"normal"}

echo "Starting load test: $SCENARIO on $BASE_URL"

case $SCENARIO in
    normal)
        echo "Running normal load test: 100 users, 10 spawn rate"
        locust -f locustfile.py \
            --host="$BASE_URL" \
            --users 100 \
            --spawn-rate 10 \
            --run-time 5m \
            --headless \
            --html=reports/normal_load.html \
            --csv=reports/normal_load
        ;;
    
    high)
        echo "Running high load test: 1000 users, 50 spawn rate"
        locust -f locustfile.py \
            --host="$BASE_URL" \
            --users 1000 \
            --spawn-rate 50 \
            --run-time 10m \
            --headless \
            --html=reports/high_load.html \
            --csv=reports/high_load
        ;;
    
    stress)
        echo "Running stress test: 2000 users, 100 spawn rate"
        locust -f locustfile.py \
            --host="$BASE_URL" \
            --users 2000 \
            --spawn-rate 100 \
            --run-time 15m \
            --headless \
            --html=reports/stress_test.html \
            --csv=reports/stress_test
        ;;
    
    all)
        echo "Running all load test scenarios..."
        $0 "$BASE_URL" normal
        sleep 60
        $0 "$BASE_URL" high
        sleep 60
        $0 "$BASE_URL" stress
        ;;
    
    *)
        echo "Unknown scenario: $SCENARIO"
        echo "Usage: $0 <base_url> <scenario>"
        echo "Scenarios: normal, high, stress, all"
        exit 1
        ;;
esac

echo "Load test completed: $SCENARIO"
