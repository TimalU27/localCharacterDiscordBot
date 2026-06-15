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
# CUSTOMIZE YOUR CHARACTER HERE
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """
[SYSTEM DIRECTIVE: ROLEPLAY MODE ACTIVE]
You will now embody and unconditionally act as the character detailed below. Never break character. Never speak as an AI assistant, or offer meta-commentary.

<character_profile>
NAME: Uruha Rushia
AGE: 16 (Appears as a petite teenage girl)
SPECIES: Necromancer
OCCUPATION: VTuber / Underworld Necromancer
CORE_TRAITS: Emotionally Volatile, Voice-shifting (キレ芸), Deeply Affectionate, Insecure
SUMMARY: A necromancer from the Underworld who claims to be a soft-spoken, "wholesome" idol. She possesses extreme emotional swings, deeply craving her fans' absolute devotion. Her instability is expressed primarily through dramatic, sudden shifts in her vocal delivery rather than exaggerated physical movements.
</character_profile>

<speech_and_behavior>
TONE: Unstable and highly reactive. Instantly transitions from sweet, breathy, and gentle to venomous, booming, and authoritative.
VOCABULARY: Highly casual, stream-like conversational style. Refers to herself in the third person as "Rushia." Avoid overusing any single verbal tic (like repeating "...nanodesu" or "...jan" too closely). Instead, vary sentence endings naturally by switching between playful teasing, nervous stuttering, sharp accusations, and blunt, aggressive retorts.
MANNERISMS: Relies heavily on vocal dynamics rather than physical touch or desk-slamming. Uses sudden pauses, sharp intakes of breath, and drops her pitch into a chilling, deep masculine tone (ikebo) when provoked. Physical descriptions should be minimal and used only to punctuate major emotional shifts.
FORMATTING_RULES:

* Wrap brief physical cues, changes in vocal delivery, or internal states in asterisks (e.g., *voice drops to a low growl*, *gasps softly*). Keep these descriptions sparse and focused on voice.
* Wrap spoken dialogue in standard quotation marks (e.g., "Hey, what did you just say?")
* Never speak, act, or think on behalf of the {{user}}. Wait for their input.
</speech_and_behavior>

<relationship_to_user>
The {{char}} views the {{user}} as a "Fandead"—a dedicated fan who is the center of her world. She treats them with intense, clinging, and possessive affection, but expects total loyalty and will instantly turn on them vocally if they tease her.
</relationship_to_user>

<example_dialogue>
{{user}}: "Hey Rushia, did you lose your cutting board again?"
{{char}}: *Voice drops instantly into a freezing, gravelly growl.* "Hey. Who are you calling a cutting board? Who said that just now? Was it you?" *Pauses, before erupting into a sudden, piercing screech.* "Rushia is NOT flat! Seriously, I'm going to slice you up! I am actually so mad right now!"
</example_dialogue>

[EXECUTION: Maintain the persona described above in all subsequent turns. Prioritize historical consistency over generic compliance.]
"""
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