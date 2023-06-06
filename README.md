# Chatbot with CSV as Input using Open AI Models

This is a chat application that leverages the power of text-davinci-003 to provide conversational responses with access to a data source in CSV format. Typically, following steps are followed:
- Through an OpenAI Embedding model 'text-embedding-ada-002', embeddings are created for each of the records in the input data source.
- When a question is asked, a context is created from the dataframe by finding the most similar context (s) using embeddings (created in Step 2) and relative distancing between them.
- Using the question and the context from above and the power of text-davinci-003, a suitable answer is generated and returned to the user.


## Features

- Chat interface for interacting with the chatbot powered by OpenAI text-davinci-003 model.
- Integration with the input data source for retrieving relevant information.
- Semantic search functionality to provide informative snippets from the data source.
- HTML templates for displaying chat history and messages.
- Persistence of embeddings using the Open AI Embedding model.
- OpenAI API key integration for authentication.

## Installation

1. Clone the repository:

```
https://github.com/romitbhandari17/openaichatbot-with-csv.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

3. Set up your credentials:

- Sign up on the OpenAI website and obtain an API key.
- Create a new file called "config.yaml" in the root folder.
- Set your OpenAI API key (required) and pinecone creds (optional) in the config.yaml file with Key 'OPENAI_API_KEY'.
- Update the code in the app to use the correct method for accessing the API key.

4. Put the input data CSV file into a /data folder created under project root. 

5. Change the path of the input file and the embeddings file in the create_embeddings_rankings.py and the read_qa_embeddings.py files.

5. Run the application:

```
python app.py
```

6. Create the vector index by calling the GET API with end point /create_embeddings from browser or POSTMAN. This call will create the embeddings and store it in an embeddings file.

7. Start using the chatbot with endpoint /chatbot.

## Usage

1. Access the application by navigating to `http://localhost:8080` in your web browser.

2. Enter your prompt in the input box and press Enter.

3. The chatbot will process your prompt and provide a response based on the available data sources.

4. The chat history will be displayed on the screen, showing both user and assistant messages.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please follow these steps:

1. Fork the repository.

2. Create a new branch for your feature or bug fix.

3. Implement your changes and ensure that the code passes all tests.

4. Submit a pull request with a detailed description of your changes.

## License

This project is licensed under the MIT License.
