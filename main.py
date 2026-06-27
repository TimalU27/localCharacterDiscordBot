import discord
from discord.ext import commands
import logging
import aiohttp
from dotenv import load_dotenv
import os

load_dotenv()

discord_token = os.getenv('DISCORD_TOKEN')
lm_studio_url = os.getenv('LM_STUDIO_URL')
lm_studio_token = os.getenv('LM_STUDIO_TOKEN')         
lm_model = os.getenv('LM_MODEL') 

# ─────────────────────────────────────────────
# LOAD SYSTEM PROMPT FROM FILE
# ─────────────────────────────────────────────
system_prompt_path = os.getenv('SYSTEM_PROMPT_FILE', 'system_prompt.txt')

try:
    with open(system_prompt_path, 'r', encoding='utf-8') as f:
        SYSTEM_PROMPT = f.read().strip()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Could not find system prompt file at '{system_prompt_path}'. "
        f"Make sure it exists or check SYSTEM_PROMPT_FILE in your .env."
    )

if not SYSTEM_PROMPT:
    raise ValueError(f"System prompt file '{system_prompt_path}' is empty.")
# ─────────────────────────────────────────────
# How many past messages to fetch for context
MESSAGE_HISTORY_LIMIT = 10

# Alternatively, fetch messages within this many minutes (commented out — swap with the block above to use)
# MESSAGE_HISTORY_MINUTES = 15

handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

intents = discord.Intents.default()
intents.message_content = True
intents.members = True

bot = commands.Bot(command_prefix='!', intents=intents)


async def fetch_recent_messages(message: discord.Message) -> list[dict]:
    """
    Strategy A (active): fetch the last MESSAGE_HISTORY_LIMIT messages before the triggering one.
    Strategy B (commented): fetch messages sent within the last MESSAGE_HISTORY_MINUTES minutes.
    Returns a list of {role, content} dicts ready for the LLM.
    """
    history = []

    # ── Strategy A: last N messages ──────────────────────────────────────────
    async for msg in message.channel.history(limit=MESSAGE_HISTORY_LIMIT, before=message):
        history.append(msg)

    # ── Strategy B: messages within a time window ─────────────────────────────
    # from datetime import datetime, timezone, timedelta
    # cutoff = datetime.now(timezone.utc) - timedelta(minutes=MESSAGE_HISTORY_MINUTES)
    # async for msg in message.channel.history(limit=100, before=message, after=cutoff):
    #     history.append(msg)
    # ─────────────────────────────────────────────────────────────────────────

    # Reverse so oldest messages come first (natural reading order)
    history.reverse()

    messages_for_llm = []
    for msg in history:
        # Map the bot's own past messages to "assistant" role, everyone else to "user"
        role = "assistant" if msg.author == bot.user else "user"
        if role == "assistant":
            content = msg.clean_content  # no name prefix on its own past turns
        else:
            content = f"{msg.author.display_name}: {msg.clean_content}"
        messages_for_llm.append({"role": role, "content": content})

    return messages_for_llm

async def query_lm_studio(chat_history: list[dict], user_message: str) -> str:
    """Send the conversation to LM Studio's OpenAI-compatible endpoint and return the reply."""
    payload = {
        "model": lm_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *chat_history,
            {"role": "user", "content": user_message},
        ],
        "temperature": 0.8,
        "max_tokens": 512,
        "stream": False,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {lm_studio_token}",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{lm_studio_url}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                raise RuntimeError(f"LM Studio returned {resp.status}: {error_text}")
            data = await resp.json()
            return data["choices"][0]["message"]["content"].strip()


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user.name} ({bot.user.id})")
    print("Bot is ready!")


@bot.event
async def on_message(message: discord.Message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Only respond when the bot is @mentioned
    if bot.user not in message.mentions:
        await bot.process_commands(message)
        return

    # Strip the @mention from the triggering message so the LLM sees clean text
    clean_trigger = message.clean_content.replace(f'@{bot.user.display_name}', '').strip()
    if not clean_trigger:
        clean_trigger = "(no message — just a mention)"

    async with message.channel.typing():
        try:
            # 1. Gather channel history for context
            history = await fetch_recent_messages(message)

            # 2. Query the local LLM
            reply = await query_lm_studio(
                chat_history=history,
                user_message=f"{message.author.display_name}: {clean_trigger}",
            )

            # 3. Send the reply, mentioning the user who triggered it
            await message.reply(reply)

        except Exception as e:
            logging.error(f"Error querying LM Studio: {e}")
            await message.reply("Mou~ something went wrong on my end... please try again! 💙")

    # Still process any other bot commands if present
    await bot.process_commands(message)


bot.run(discord_token, log_handler=handler, log_level=logging.DEBUG)