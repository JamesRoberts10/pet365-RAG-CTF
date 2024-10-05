# pet365

  

A content analysis tool that extracts key insights from various sources including YouTube videos, web pages, and documents.
  

> ***It's Blinkest, but for YouTube, blogs and PDFs***

  

## Flow

![FlowChart](images/FlowChart.png)


  

## How It Works

 1. Content Extraction: The system use various methods to extract raw
    text from the provided source (YouTube video, web page, or uploaded
    file).   
 2. Preprocessing: Raw content is cleaned and formatted in preparation for analysis.
    
 3. Analysis: The selected LLM processes the content using a set of optimised prompts.
 4. Insight Generation: The system generates a structured output including a summary, key ideas, insights, and recommendations.
 5. User Interaction: Results are streamed in real-time through the chat interface.

  

  

## Project Files Description

  

**Executable Files**

  

***[vector_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/vector_utils.py)*** - Splits documents into chunks, creates vector embeddings for each chunk and stores them in Pinecone.



***[ai_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/ai_utils.py)*** - Provides core functionality for content analysis. Handles YouTube transcript extraction, web content scraping, file parsing, and LLM summarisation. Includes functions for processing content, making API calls, and streaming responses to the frontend.

  

***[api_utils.py](https://github.com/JamesRoberts10/pet365/blob/main/src/controllers/api_utils.py)*** - Manages API key operations for the project. Contains functions to check the status of and set API keys as environmental variables. 


***[frontpage.py](https://github.com/JamesRoberts10/pet365/blob/main/src/views/frontpage.py)*** - Defines the Gradio-based user interface. 

  

***[main.py](https://github.com/JamesRoberts10/MEGAai/blob/main/src/Main.py)*** - The application entry point. Initiates the Gradio front end.

  

**Configuration Files**

  

***[prompts.py](https://github.com/JamesRoberts10/MEGAai/blob/main/src/templates/prompts.py)*** - Predefined prompt templates for various content analysis tasks.

  

  

  

## Installation

  

  

### Standard Installation

  

  

To install using the standard method, follow these steps:

  

  

1. Clone the repository

```bash

git clone https://github.com/JamesRoberts10/MEGAai.git

```

2. Navigate to the project directory

```bash

cd MEGAai

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

git clone https://github.com/JamesRoberts10/MEGAai.git

```

  

2. Navigate to the project directory

```bash

cd MEGAai

```

  

3. Build the Docker image:

```bash

docker build -t megaai .

```

  

4. Run the Docker container:

  

```bash

docker run -p 7860:7860 megaai

```

  

  

5. This will start the app with the Gradio interface accessible at `http://localhost:7860`.

  

  

> Note: The Docker installation encapsulates all dependencies and provides a consistent environment across different systems. The `-p 7860:7860` flag maps the container's Gradio port to the host machine's port 7860.

  

  

  

## Usage

  

  

1. Run

  

2. Set API keys tab

  

3. Add at least one LLM API key

  

4. Extract Ideas tab

  

5. Select your LLM of choice

  

6. Enter Youtube/Web URL or upload a file

  

7. Submit