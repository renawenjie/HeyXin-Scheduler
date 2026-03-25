# HeyXin Scheduler — Deployment Guide

This guide covers deploying the HeyXin daily check-in scheduler to a free cloud service. The scheduler runs 24/7, sends personalized check-in messages to users at their preferred times, and automatically extracts user preferences from Dify conversation history.

## Architecture Overview

The scheduler is a Python FastAPI application that provides an HTTP API for user registration and runs a background thread that sends Telegram check-in messages at each user's scheduled time. It reads conversation history from the Dify API and uses a lightweight LLM call (GPT-4.1-nano) to extract user preferences such as name, check-in time, timezone, language, and core values.

## Environment Variables

The following environment variables must be set in your cloud deployment:

| Variable | Description | Example |
|---|---|---|
| `DIFY_API_KEY` | Your Dify Chatflow API key | `app-H8ycAHrUlDFu6YfqHpCSrBYO` |
| `TELEGRAM_BOT_TOKEN` | Your Telegram bot token | `8734162668:AAF4Fc_M-PLvBj1...` |
| `OPENAI_API_KEY` | OpenAI API key for preference extraction | `sk-...` |
| `PORT` | Port the server listens on (auto-set by most platforms) | `8080` |
| `DB_PATH` | Path to SQLite database file | `heyxin_users.db` |
| `SCAN_INTERVAL_MINUTES` | How often to re-scan for new users (not currently used in HTTP mode) | `10` |

## Option A: Deploy to Render (Recommended for Trial)

Render offers a free tier for web services. The service may sleep after 15 minutes of inactivity on the free tier, but since the Dify Workflow will call the `/register` endpoint on every message, it stays awake during active usage. For the paid tier ($7/month), the service runs continuously without sleeping.

### Steps

1. Push your code to a GitHub repository (public or private).

2. Go to [render.com](https://render.com) and sign up or log in.

3. Click **New** and select **Web Service**.

4. Connect your GitHub repository.

5. Configure the service with the following settings:
   - **Name:** `heyxin-scheduler`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python scheduler.py`

6. Add the environment variables listed above in the **Environment** section.

7. Click **Create Web Service** and wait for the deployment to complete.

8. Note the service URL (e.g., `https://heyxin-scheduler.onrender.com`). You will need this for the Dify Workflow.

### Important Note on Render Free Tier

The free tier spins down after 15 minutes of inactivity. This means the background check-in thread will stop running. To keep it alive, you can use a free uptime monitoring service like [UptimeRobot](https://uptimerobot.com) to ping the health endpoint (`GET /`) every 5 minutes. Alternatively, upgrade to the $7/month Starter plan for always-on service.

## Option B: Deploy to Railway

Railway offers a $5 trial credit (no credit card required) and then pay-as-you-go pricing. Services run continuously without sleeping.

### Steps

1. Push your code to a GitHub repository.

2. Go to [railway.app](https://railway.app) and sign up with GitHub.

3. Click **New Project** and select **Deploy from GitHub repo**.

4. Select your repository.

5. Railway will auto-detect the Dockerfile. If not, it will use the Procfile.

6. Go to the **Variables** tab and add the environment variables listed above.

7. Go to **Settings** and under **Networking**, click **Generate Domain** to get a public URL.

8. Note the service URL (e.g., `https://heyxin-scheduler-production.up.railway.app`).

## Option C: Deploy to Koyeb

Koyeb offers a free tier with one nano instance that runs continuously (no sleeping).

### Steps

1. Push your code to a GitHub repository.

2. Go to [koyeb.com](https://www.koyeb.com) and sign up.

3. Click **Create App** and select **GitHub** as the deployment method.

4. Select your repository and configure:
   - **Builder:** Dockerfile
   - **Port:** 8080

5. Add environment variables in the **Environment variables** section.

6. Deploy and note the public URL.

## After Deployment: Update the Dify Telegram Workflow

Once the scheduler is deployed and you have the public URL, you need to add one more node to your Dify Telegram Workflow so that every user message triggers a registration check.

### Add an HTTP Request node (Register User)

Add a new HTTP Request node **after CODE 2** (the node that parses the Dify response) and **in parallel with HTTP REQUEST 2** (the node that sends the Telegram reply). This way registration happens alongside the Telegram reply without slowing it down.

Configure the new node as follows:

- **Method:** POST
- **URL:** `https://YOUR-SCHEDULER-URL/register`
- **Headers:** `Content-Type` = `application/json`
- **Body (JSON):**

```json
{
  "chat_id": "{{CODE.chat_id}}"
}
```

Use the variable picker to select **CODE → chat_id** for the `chat_id` field.

This node does not need to connect to anything after it — it is a fire-and-forget call. The scheduler will read the Dify conversation and register the user if onboarding is complete.

## API Endpoints

The scheduler exposes the following endpoints for administration and integration:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check — returns service status and active user count |
| POST | `/register` | Register a user by chat_id (called by Dify Workflow) |
| POST | `/register/manual` | Manually register a user with all preferences |
| GET | `/users` | List all registered users |
| GET | `/users/{chat_id}` | Get info for a specific user |
| POST | `/users/{chat_id}/deactivate` | Stop check-ins for a user |
| POST | `/users/{chat_id}/activate` | Resume check-ins for a user |
| POST | `/checkin/test/{chat_id}` | Send a test check-in message |

## Monitoring

Check the health endpoint periodically to ensure the service is running. The response includes the number of active users and the current timestamp. Set up UptimeRobot or a similar service to alert you if the health check fails.
