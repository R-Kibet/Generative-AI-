import logging
import asyncio
from aiogram import Bot, Dispatcher, html, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv
import os

load_dotenv()
telt = os.getenv("TEL_BOT")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize bot and dispatcher
bot = Bot(token=telt)
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message):
    """
    This handler receives messages with `/start` command
    """
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        f"Welcome to the REDI! Send me any message and I'll echo it back.\n"
        f"Use /help to see available commands."
    )

@dp.message(Command(commands=['help']))
async def command_help_handler(message: Message):
    """
    This handler receives messages with `/help` command
    """
    help_text = """
Available commands:
/start - Start the bot
/help - Show this help message
    
Just send me any text message and I'll echo it back to you!
    """
    await message.answer(help_text)

@dp.message(F.text & ~F.text.startswith('/'))
async def echo_handler(message: Message):
    """
    Handler for all text messages (except commands)
    """
    await message.answer(f"You said: {html.quote(message.text)}")

@dp.message(F.photo)
async def photo_handler(message: Message):
    """
    Handler for photo messages
    """
    await message.answer("Nice photo! 📸")

@dp.message(F.document)
async def document_handler(message: Message):
    """
    Handler for document messages
    """
    await message.answer("Thanks for the document! 📄")

@dp.message()
async def other_handler(message: Message):
    """
    Handler for all other types of messages
    """
    await message.answer("I received your message, but I'm not sure how to handle this type of content yet.")

async def main():
    """
    Main function to start the bot
    """
    print("Starting bot...")
    try:
        # Start polling
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        print("Bot stopped by user")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())