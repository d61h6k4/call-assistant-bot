# AI Call Assistant Bot

## Overview

The AI Call Assistant Bot is an advanced system designed to assist with various tasks through perception, evaluation, and articulation. It comprises three main subsystems, each with a distinct role:

1. **Perceiver**: Responsible for reading and interpreting data from sensors.
2. **Evaluator**: Processes and analyzes data, making decisions.
3. **Articulator**: Expresses and communicates information through writing or speaking.

## Subsystems

### Perceiver

The Perceiver subsystem is the sensory component of the AI Call Assistant Bot. It captures and interprets data from various sources, such as:

- **Screen Capturing**: Reads and interprets information displayed on screens.
- **Speaker Capturing**: Captures and processes audio input from speakers.

Once the data is interpreted, the Perceiver sends it to the Evaluator for further processing.

### Evaluator

The Evaluator subsystem is the decision-making core of the AI Call Assistant Bot. Its functions include:

- **Data Processing**: Analyzes the data received from the Perceiver.
- **Decision Making**: Makes informed decisions based on the analyzed data.

The Evaluator can send information to the Articulator for expression and communication.

### Articulator

The Articulator subsystem is responsible for expressing and communicating information. It can operate through:

- **Chat**: Communicates information via text in chat interfaces.
- **Microphone**: Uses speech synthesis to express information audibly.

## Installation

To install and set up the AI Call Assistant Bot, follow these steps:


### Prerequisite
Install git and bazel.
#### MacOS
Install XCode

#### Linux
```
sudo apt install -y yasm automake libtool m4 g++
```

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI-Call-Assistant-Bot.git
   cd AI-Call-Assistant-Bot
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Run the initial setup:
   ```sh
   python setup.py
   ```

## Usage

To start the AI Call Assistant Bot, execute the following command:

```sh
python main.py
```

You can configure the system by modifying the configuration file `config.json`.

## Contributing

We welcome contributions to improve the AI Call Assistant Bot. To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix:
   ```sh
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```sh
   git commit -m "Description of your changes"
   ```
4. Push to the branch:
   ```sh
   git push origin feature-name
   ```
5. Open a pull request.
