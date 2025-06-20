# Backend connecting to the LLM
import logging
import asyncio
import openai
from aiogram import Bot, Dispatcher, html, F
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPEN_AI_KEY")  # Fixed variable name
telt = os.getenv("TEL_BOT")

# Initialize bot and dispatcher
bot = Bot(token=telt)
dp = Dispatcher()
openai_client = AsyncOpenAI(api_key=openai_api_key)

class Reference:
    """
    Storage of previous chats
    """
    
    def __init__(self) -> None:
        self.reference = ""
        
ref = Reference()
model_name = "gpt-3.5-turbo"

def clear_timeline():
    """Function to clear the history"""
    ref.reference = ""  # Fixed attribute name

@dp.message(CommandStart())
async def command_start_handler(message: Message):
    """
    This handler receives messages with `/start` command
    """
    await message.answer(
        f"Hello, {html.bold(message.from_user.full_name)}!\n"
        f"Welcome to the REDI! Send me any message and I'll respond with AI.\n"
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
/ask [question] - Ask a specific question
/clear - Clear conversation history
    
Just send me any text message and I'll respond using AI!
    """
    await message.answer(help_text)

@dp.message(Command(commands=['clear']))
async def clear_handler(message: Message):
    """
    This handler clears the conversation history
    """
    clear_timeline()
    await message.answer("The conversation history has been cleared! 🧹")

@dp.message(Command(commands=['ask']))
async def chatgpt_command_handler(message: Message):
    """
    This handler processes /ask command and generates a response using ChatGPT
    """
    # Extract the question from the command
    question = message.text[5:].strip()  # Remove '/ask ' from the beginning
    
    if not question:
        await message.answer("Please provide a question after /ask command.\nExample: /ask What is Python?")
        return
    
    # Send "typing" action to show bot is processing
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant responding to questions in a Telegram bot. Keep responses concise but informative."}
        ]
        
        # Add previous context if exists
        if ref.reference:
            messages.append({"role": "assistant", "content": ref.reference})
        
        messages.append({"role": "user", "content": question})
        
        # Call OpenAI API
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1000,
            temperature=0.7
        )
        
        # Extract the response
        chatgpt_response = response.choices[0].message.content
        
        # Update reference for context
        ref.reference = chatgpt_response
        
        # Send the response back to user
        await message.answer(f"🤖 AI says:\n\n{chatgpt_response}")
        
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        await message.answer("Sorry, I encountered an error while processing your request. Please try again later.")

@dp.message(F.text & ~F.text.startswith('/'))
async def chatgpt_text_handler(message: Message):
    """
    This handler processes user input and generates a response using ChatGPT
    """
    # Send "typing" action to show bot is processing
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    
    try:
        # Build conversation history
        messages = [
            {"role": "system", "content": "You are a helpful assistant in a Telegram bot. Respond naturally and helpfully to user messages. Keep responses conversational and not too long."}
        ]
        
        # Add previous context if exists
        if ref.reference:
            messages.append({"role": "assistant", "content": ref.reference})
        
        messages.append({"role": "user", "content": message.text})
        
        # Call OpenAI API
        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=800,
            temperature=0.7
        )
        
        # Extract the response
        chatgpt_response = response.choices[0].message.content
        
        # Update reference for context
        ref.reference = chatgpt_response
        
        # Send the response back to user
        await message.answer(chatgpt_response)
        
    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}")
        await message.answer("Sorry, I encountered an error while processing your request. Please try again later.")

@dp.message()
async def other_handler(message: Message):
    """
    Handler for all other types of messages
    """
    await message.answer("I can only process text messages right now. Send me a text message and I'll respond using AI!")

async def main():
    """
    Main function to start the bot
    """
    print("Starting AI Telegram Bot...")
    
    # Check if API keys are set
    if not openai_api_key:
        print("Error: OPEN_AI_KEY not found in environment variables")
        return
    
    if not telt:
        print("Error: TEL_BOT not found in environment variables")
        return
    
    try:
        # Start polling
        await dp.start_polling(bot)
    except KeyboardInterrupt:
        print("Bot stopped by user")
    finally:
        await bot.session.close()

if __name__ == "__main__":
    asyncio.run(main())