## Introduction
This is a simple PSR test written in Locust. 
It simulates users that execute the following scenario in a loop:
- call the OpenAI API to convert text to speech
- play the generated audio file


## Get Started
Install the packages required by the PSR test: 
```
uv pip install locust atomicx
```

## Run the test
Start Matcha-TTS server first, then start Locust:
```
export HOST=http://localhost:8880
locust -f ./psr/openai_load_test.py -H "$HOST" --web-host 127.0.0.1 --users 1 --spawn-rate 0.1 --run-time 5m MatchaTTSClient
```

1. Use the Locust UI on http://localhost:8089 to start the test.
2. Set the number of users to a value supported by your machine (e.g.: Intel i9 + nvidia 3050 could support 10 users).
3. Set the user ramp up to a desired value (e.g.: 0.1 means one user every 10 seconds)
4. Click Advanced and set the desired test duration (Locust will run user simulations for that duration).
5. Click Start.
