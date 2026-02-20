#!/usr/bin/env bash

HOST=http://localhost:8880

locust -f ./psr/openai_load_test.py -H "$HOST" --web-host 127.0.0.1 --users 1 --spawn-rate 0.1 --run-time 5m MatchaTTSClient
