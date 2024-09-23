from dotenv import load_dotenv
load_dotenv()
from langchain import hub
from langchain.agents import (AgentExecutor,create_structured_chat_agent)
from langchain_core.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
import pygame
import whisper
import sounddevice as sd
import wavio
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def play_audio(*args,**kwargs):
    """play audio from audio folder"""
    audio_file_path = os.path.join(BASE_DIR, "play", "dingdong.wav")
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    response = {
        "action": "Final Answer",
        "action_input": "Audio played"
    }
    return json.dumps(response)

def audio_to_text(*args,**kwargs):
    """convert audio to text"""
    model = whisper.load_model("base")
    audio_file_path = os.path.join(BASE_DIR, "audio", "male.wav")
    result = model.transcribe(audio_file_path)
    print("Here is the transcription: \n")
    print(result["text"])
    text_file_path = os.path.join(BASE_DIR, "audio_to_text", "audio-to-text.txt")
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.write(result["text"])
    response = {
        "action": "Final Answer",
        "action_input": "Audio to text conversion finished"
    }
    return json.dumps(response)

def record_audio(*args,**kwargs):    
    """record audio"""
    duration = 5 
    fs = 44100    
    print("\nRecording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait() 
    print("Recording finished")
    audio_output_path = os.path.join(BASE_DIR, "audio", "output.wav")
    wavio.write(audio_output_path, audio, fs, sampwidth=2)
    response = {
        "action": "Final Answer",
        "action_input": "Recording finished"
    }
    return json.dumps(response)
#tools list
tools = [
    Tool(
        name="play_audio", 
        func=play_audio, 
        description="Play an audio file from the audio folder. Use this tool for any action related to playing audio.\nHere is the keyword you can refer to play audio: play audio , play speaker"
    ),
    Tool(
        name="audio_to_text", 
        func=audio_to_text, 
        description="Convert an audio file to text and save the result in a text file. Use this tool only for audio-to-text conversion.\nHere is the keyword you can refer to audio to text conversion: audio to text , audio to speech , STT"
    ),
    Tool(
        name="record_audio", 
        func=record_audio, 
        description="Record audio using the system's microphone and save it as an audio file. Use this tool for recording audio.\nHere is the keyword you can refer to record audio: record audio , store the speaker,Voice to audio"
    )
]

#stores conversation 
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

#pull  prompt template from hun
prompt = hub.pull("hwchase17/structured-chat-agent")

#initialize model , set temperature to 0 for deterministic output
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#create agent, set stop sequence to True to avoid hallucination
agent = create_structured_chat_agent(llm=model, tools=tools, prompt=prompt, stop_sequence=True)

#create agent executor
# max_iterations=3 => 暫時的workaround: 有時候LLM 會持續Observation and Thoughts並且認為還沒有獲得答案,日後可以針對狀態做調整或是使用Langgraph來實現agent
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True,memory=memory,handle_parsing_errors=True,max_iterations=3,early_stopping_method="force")
   

initial_message = "You are a helpful AI assistant that can provide helpful answers using available tools\nIf you are unable to answer the question, please say so.\nDon't repeat the tool again and again.\nNote which you must follow:\nIf you see the output like this {'action': 'Final Answer', 'action_input': 'Audio played/Recording finished/Audio to text conversion finished'} it means you have successfully implement the tool.Then you can stop Observation and Thought, and response with your final answer:\n"
memory.chat_memory.add_message(SystemMessage(content=initial_message))

#chat loope
while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break
    memory.chat_memory.add_user_message(HumanMessage(content=user_input))
    response = agent_executor.invoke({"input": user_input})
    print("Bot: ", response["output"])
    memory.chat_memory.add_ai_message(AIMessage(content=response["output"]))
