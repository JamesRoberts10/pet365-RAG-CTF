# pet365

<INTRO>
  

## Flow

<DIAGRAM>


  

## How It Works

 

  

  

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

git clone https://github.com/JamesRoberts10/MEGAai.git

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

  

3. Add at least one LLM API key

  

4. 

  

5. 

  

6. 

  

7. 