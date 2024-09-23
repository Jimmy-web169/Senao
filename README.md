## Requirements

- Python 3.7+
- `dotenv`
- `langchain`
- `pygame`
- `whisper`
- `sounddevice`
- `wavio`
- `json`

## Installation

1. Clone the repository:

   ```sh
   git clone https://github.com/Jimmy-web169/SENAO.git
   ```

2. Create a `.env` file in the root directory and add your environment variables OPENAI_API_KEY.

3. Install the required packages:
   ```sh
   sudo apt-get install portaudio19-dev
   pip install -r requirements.txt
   ```

## Usage

1. Run the main script:

   ```sh
   python main.py
   ```

2. Interact with the AI assistant through the command line. You can use the following commands:

   - `play audio`: To play dingdong.wav.
   - `audio to text`: To convert male.wav to text.
   - `record audio`: To record audio using the system's microphone.

3. Type `exit` to stop the assistant.

## File Structure

- `main.py`: The main script that initializes the AI assistant and handles user interactions.
- `play/`: Directory containing play files.
- `audio/`: Directory containing audio files.
- `audio-to-text/`: Directory where the transcribed text files are saved.
- `play/dingdong.wav`: Sample files.
- `audio/male.wav`: Sample audio files.

## Model

OpenAI(gpt-3.5-turbo)

## Note

有時候 LLM 會持續 Observation and Thoughts 並且認為還沒有獲得答案,日後可以針對狀態做調整或是使用 Langgraph 的 State 來實現 agenty，或是調整 prompt
