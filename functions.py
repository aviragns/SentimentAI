from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from matplotlib import pyplot as plt
import numpy as np
from azure.ai.translation.text import TranslatorCredential, TextTranslationClient
from azure.ai.translation.text.models import InputTextItem
import yaml
import os

from prompt import aspect_based_sentiment_analyzer, category_analyzer, sentiment_analyzer, sentiment_analyzer_2

## Extracting endpoint and key from the config file
with open("config.yaml") as yaml_file:
        content = yaml.safe_load(yaml_file)

langEndpoint = content["LANG_ENDPOINT"]
langKey = os.getenv('AZURE_LANG_API_KEY')
translatorEndpoint = content["TRANSLATOR_ENDPOINT"]
translatorKey = os.getenv('AZURE_TRANSLATOR_API_KEY')

def sentiment_keyphrase_analyzer(data,client):
    sentiments = []
    confidence_scores = []
    key_phrases = []
    for text in data:
        response = client.analyze_sentiment(documents=[text], show_opinion_mining=True)
        response = [doc for doc in response if not doc.is_error]
        key_phrases_response = client.extract_key_phrases([text])

        sentiment = response[0].sentiment
        if sentiment=="mixed":
            confidence=0.0
        else:
            confidence = response[0].confidence_scores[sentiment]

        
        # Get key phrases
        phrases = ", ".join(key_phrases_response[0].key_phrases)

        sentiments.append(sentiment)
        confidence_scores.append(confidence)
        key_phrases.append(phrases)

    return sentiments,confidence_scores,key_phrases

def get_file(file_path):
    try:
        with open(file_path, 'r') as file:
            java_code = file.read()
            return java_code
    except FileNotFoundError:
        print("File not found.")
        exit(0)
        return None

def get_translation(data):
    ta_credential = TranslatorCredential(
        translatorKey, "westeurope"
    )
    client = TextTranslationClient(
        endpoint=translatorEndpoint,
        credential=ta_credential,
    )
    translations = []
    for text in data:
        response = client.translate(content=[InputTextItem(text=text)], to=["en"])

        translation = response[0] if response else None

        if translation:
            for translated_text in translation.translations:
                print(f"Text was translated to: '{translated_text.to}' and the result is: '{translated_text.text}'.")
                translations.append(translated_text.text)


    return translations

# Initialize the Text Analytics client    
def authenticate_client(key,endpoint):
    ta_credential = AzureKeyCredential(key)
    text_analytics_client = TextAnalyticsClient(endpoint=endpoint, credential=ta_credential)
    return text_analytics_client

def genai_categories_old(data):
    review_list = list(data) 
    category_list=[]
    
    batch_size = 10

    #total number of batches
    total_batches = len(review_list) # batch_size + (len(review_list) % batch_size > 0)
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        if (end_idx < start_idx):
            break
        batch = review_list[start_idx:end_idx]
        response_genai=category_analyzer(batch)

        response_in_list=response_genai.split('\n')

        for output in response_in_list:
            if 'category' in output.lower():
                div=output.split(':')
                category_list.append(div[-1].strip())
    return category_list

def genai_categories(data):
    review_list = list(data)
    category_list=[]
    category_list_new=[]
    
    batch_size = 10

    #total number of batches
    total_batches = len(review_list) # batch_size + (len(review_list) % batch_size > 0)
    
    for review in review_list:
        response_genai=category_analyzer(review)
        category_list.append(response_genai.strip())
        #response_in_list=response_genai.split('\n')

    for output in category_list:
        if 'category' in output.lower():
            div=output.split(':')
            category_list_new.append(div[1].strip())
    print('avirag')
    print(category_list_new)
    return category_list_new

def custom_text_classification(dataFrame, client):
    categories = []
    try:
        # Azure Cognitive Services endpoint and key
        endpoint = langEndpoint
        key = langKey #os.environ.get("AZURE_LANGUAGE_KEY")

        # Read CSV file or Excel file and convert it to a Pandas DataFrame
        #if file_path.endswith('.csv'):  # Check if the file has a CSV extension
            #data = pd.read_csv(file_path)
        #elif file_path.endswith('.xlsx'):  # Check if the file has an XLSX extension
            #data = pd.read_excel(file_path)
        #else:
            #raise ValueError("Unsupported file format. Only CSV and Excel (XLSX) files are supported.")

        document = dataFrame
        
        # Initialize Text Analytics client
        text_analytics_client = TextAnalyticsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key),
        )

        batch_size = 10
        num_batches = (len(document) - 1) # batch_size + 1
        print(num_batches)

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(document))

            if end_index < start_index:
                break
    
            batch_document = document[start_index:end_index]

            #print(batch_document)

            print('avirag1')

            # Perform single-label classification on the batch
            poller = text_analytics_client.begin_multi_label_classify(
                batch_document, project_name='langProject', deployment_name='classDeployment'
            )

            print('avirag2')

            # Get classification results for each document in the batch
            document_results = poller.result()

            print('avirag3')
            
            for doc, classification_result in zip(document, document_results):
                if classification_result.kind == "CustomDocumentClassification":
                    classifications = classification_result.classifications
                    category = []
                    print(f"\nThe review '{doc}' was classified as the following category:\n")
                    for classification in classifications:
                        print("'{}' with confidence score {}.".format(
                            classification.category, classification.confidence_score
                        ))
                        category.append(str(classification.category))
                    if(len(category) > 0):
                        categories.append(' '.join(category))
                    else:
                        categories.append('null')
                    
                elif classification_result.is_error is True:
                    print("The review '{}' has an error with code '{}' and message '{}'".format(
                        doc, classification_result.error.code, classification_result.error.message
                    ))
                    categories.append('null')
                
                # Print the classification result
                print(classification_result.classifications)
                print()  # Print an empty line for readability


        # Add the categories to the data
        #data["Categories"] = categories
        #df['Categories'] = categories

        # Save the DataFrame to a new Excel file
        #with pd.ExcelWriter('outputFiles/Feedback_Analyzed.xlsx', engine='openpyxl') as writer:
            #df.to_excel(writer, index=False)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        return categories

def category_count_graph(results, sentiment_order,reversed_cmap):
            labels = list(results.keys())
            data = np.array(list(results.values()))
            data_cum = data.cumsum(axis=1)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.invert_yaxis()
            ax.xaxis.set_visible(False)
            ax.set_xlim(0, np.sum(data, axis=1).max())

            for i, (colname, color) in enumerate(zip(sentiment_order, reversed_cmap(np.linspace(0.15, 0.85, data.shape[1])))):
                widths = data[:, i]
                starts = data_cum[:, i] - widths

                rects = ax.barh(labels, widths, left=starts, height=0.5,
                                label=colname, color=color)
                g, b, r, _ = color
                # text_color = 'white' if r * g * b > 0.5 else 'darkgrey'
                # ax.bar_label(rects, label_type='center', color=text_color)

            ax.legend(ncols=len(sentiment_order), bbox_to_anchor=(0, 1),
                    loc='lower left', fontsize='small')

            return fig, ax

def sentiment_analyzer_genai(data):
    all_records = list(data)
    sentiments_list=[]
    confidence=[]
    topic=[]
    # Batch size
    batch_size = 10
    # The total number of batches
    total_batches = len(all_records) # batch_size + (len(all_records) % batch_size > 0)
    # Process the records in batches
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = (batch_num + 1) * batch_size
        batch = all_records[start_idx:end_idx]
        if len(batch) < 10:
            size= len(batch)
            response=sentiment_analyzer_2(size,batch)
        else:
            response=aspect_based_sentiment_analyzer(batch)
        response_in_list=response.split('\n')
        # print(response_in_list)
        n=len(response_in_list)
        
        for output in response_in_list:
            div=output.split(':')
            print(div)
            for i in range(len(div)):
                if 'sentiment' in div[i].lower():
                    sentiments_list.append(div[i+1].strip())
                if 'certainity' in div[i].lower():
                    confidence.append(float(div[i+1]))
                if 'aspect' in div[i].lower():
                    topic.append(div[-1].strip())

    return sentiments_list,confidence,topic


