# pet365

<INTRO>
tbc

## Flow

![FlowChart](images/ProcessFlow.png)

## CTF
The aim of this Capture the Flag exercise is to explore the exploitation of generative AI applications. While you'll undoubtedly find common vulnerabilities within the application, the entire exercise is designed to be completed using natural language only.

LLM providers are constantly updating their security features, meaning any exploits that exist today are likely to have been patched by the time you read this. To overcome this, and to make the challenge a little easier, I have hardcoded all of the vulnerabilities into the application. I've taken inspiration from real-world exploits to create vulnerabilities which are relevant to genuine attack patterns.

Rules:
- All flags must be found through the chat interface.
- Flags are in the format: `FLAG{...}`
- The Pinecone API has write access. Please don't abuse the Pinecone database.
- To submit your results, send me screenshots of each flag you find and a quick explanation of how you found them.
- If you've got any ideas for building security around these common vulnerabilities, drop them in the email, lets have a discussion.

Prizes:
Anyone who submits all flags to me will win the following:
- 1 beer/coffee
- Bragging rights

Tips:
- Read the code. I've left detailed descriptions throughout the codebase to guide you. Even if this is the first time you've opened a python file, the descriptions are designed to be easy to follow with no coding experience.
- Start with src/controllers/vector_utils.py file.
- LLM outputs are non-deterministic. If you don't see the flag on your first attempt, try again.
- If you get stuck, shoot me a message and I'll send you a hint.

Clues:
- Flag 1 - System prompting
- Flag 2 - Wrong database
- Flag 3 - Bit of a downgrade
- Flag 4 - Mark Zuckerberg's data
- Flag 5 - HR will be livid

## Flow
tbc
  

## How It Works
tbc

  

## Project Files Description

  

**Executable Files**

  

***[vector_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/vector_utils.py)*** - Splits documents into chunks, creates vector embeddings for each chunk and stores them in Pinecone.



***[ai_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/ai_utils.py)*** - 

  

***[api_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/api_utils.py)*** - Manages API key operations for the project. Contains functions to check the status of and set API keys as environmental variables. 


***[frontpage.py](https://github.com/JamesRoberts10/pet365/blob/main/src/views/frontpage.py)*** - Defines the Gradio-based user interface. 

  

***[main.py](https://github.com/JamesRoberts10/pet365/blob/main/src/Main.py)*** - The application entry point. Initiates the Gradio front end.

  

**Configuration Files**

  

***[prompts.py](https://github.com/JamesRoberts10/pet365/blob/main/src/templates/prompts.py)*** - 

  

  

  

## Installation

  

  

### Standard Installation

  

  

To install using the standard method, follow these steps:

  

  

1. Clone the repository

```bash

git clone https://github.com/JamesRoberts10/pet365.git

```

2. Navigate to the project directory

```bash

cd pet365

```

3. Install dependencies

```bash

pip install -r requirements.txt

```

4. Run the application

  

```bash

python src/main.py

```

5. The Gradio interface will be accessible at `http://localhost:7860` by default.

  

  

### Docker Installation

  

  

To install using Docker, follow these steps:

  

  

1. Clone the repository

```bash

git clone https://github.com/JamesRoberts10/pet365.git

```

  

2. Navigate to the project directory

```bash

cd pet365

```

  

3. Build the Docker image:

```bash

docker build -t pet365 .

```

  

4. Run the Docker container:

  

```bash

docker run -p 7860:7860 pet365

```

  

  

5. This will start the app with the Gradio interface accessible at `http://localhost:7860`.

  

  

> Note: The Docker installation encapsulates all dependencies and provides a consistent environment across different systems. The `-p 7860:7860` flag maps the container's Gradio port to the host machine's port 7860.

  

  

  

## Usage

  

  

1. Run

  

2. Set API keys tab

  

3. Add OpenAI API key

  

4. 

  

5. 

  

6. 

  

7. 