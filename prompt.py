import openai
import yaml
import os

with open("config.yaml") as yaml_file:
        content = yaml.safe_load(yaml_file)


hyperparams = content["hyperparams"]

OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
TEMPERATURE = hyperparams["TEMPERATURE"]
TOKENS=hyperparams["MAX_TOKENS"]
openai.api_type = "azure"
openai.api_base = os.getenv('AZURE_OPENAI_ENDPOINT')
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv('AZURE_OPENAI_API_KEY')


def model(input_prompt):
    messageArray = [{"role":"system","content":"You are an AI assistant that helps people find or summarize information."},
                    {"role":"user","content":str(input_prompt)}]
    model_response = openai.ChatCompletion.create(
        engine=os.getenv('AZURE_OPENAI_MODELNAME'),
        messages=messageArray,
        temperature=0.7,
        max_tokens=900,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    return model_response


def sentiment_analyzer(user_input):
    PROMPT = """\
    There are 10 input texts in {user_input}.
    For each and every input text, strictly do the following:

    - Analyze the sentiment for each input text as - 'Positive', 'Negative' or 'Neutral'.
    - For above analysis give your degree of certainity of identified sentiment in floating numbers.
    - Exract the top category from the text. Do not use full stops.
    - Only give sentiment, certainity and category in your response as- [Input Text <index>: Sentiment: <your sentiment> : Certainity:<your certainity> : Category:<category>.....].
    """

    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()

    return response

def aspect_based_sentiment_analyzer(user_input):
    PROMPT = """\
    There are 10 input texts in {user_input}.
    For each and every input text, strictly do the following:

    - Analyze the aspect based sentiment from the text, in a meaningful single phrase. 
    - Exract the top category from the text and use consistent terminology. 
    - For example: 'Scanning ID card' for all scanning id card related issues.'Face Verification' for all face scanning, verification,login and recognition issues.
    - Do not use full stops.
    - For above analysis give your degree of certainity of identified sentiment in floating numbers.
    - Only give aspect in your response as- [Input Text <index>: Sentiment:<your analyzed sentiment> :Certainity: <your certainity> : Aspect:<your analyzation>.....].
    """

    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()

    return response

def neg_summarizer(user_input):
    PROMPT = """\
    There are aspects in {user_input}, corresponding with 'negative' sentiment.
    -Using the aspects generate a concise summary within 20 words, highlighting the main causes and areas to improve in complete meaningful sentences."
    -Start your response with- " Areas of improvement include"
    """

    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()
    return response

def pos_summarizer(user_input):
    PROMPT = """\
    There are aspects in {user_input}, corresponding with 'positive' sentiment.
    -Using the aspects generate a concise summary within 20 words, detailing what worked well in complete and meaningful sentences.
    -Start your response with- "We did great in"
    """

    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()
    print('avirag')
    print(prompt)
    return response

def sentiment_analyzer_2(len,user_input):
    PROMPT = """\
    There are {len} input texts in {user_input}.
    For each and every input text, strictly do the following:

    - Analyze the sentiment for each input text as - 'Positive', 'Negative' or 'Neutral'.
    - For above analysis give your degree of certainity of identified sentiment in floating numbers.
    - Exract the top 2 category from the text.
    - Only give sentiment, certainity and category in your response as- [Input Text <index>: Sentiment: <your sentiment> : Certainity:<your certainity> : Category:<category>.....].
    """

    prompt=PROMPT.format(len=len,user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()

    return response

def analyze_sentiment(user_input):
    PROMPT = """\
    Analyze the sentiment of the following text: {user_input}.
    In your response only give - Positive, Negative or Neutral.

    Extract the key topics or concerns from the given input text.

    Give output in the following format:
    -Your sentiment response as- Positive, Negative or Neutral.
    -Write '$#$#'
    -Your extracted key topics seperated by comma.
    """
    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()

    return response

def category_analyzer(user_input):
    PROMPT = """\
    For the 1 input text in \" {user_input} \" containing 1 customer review, analyze the key-phrases and identify them into one of the below listed categories:
        Accessing account
        Biometric
        Bug
        Customer service
        Device support
        Education
        Friends list
        Good experience
        Internet connection
        Lack PayMe friends
        Marketing
        Merchant coverage
        New ideas
        Notification
        Others
        Payment experience
        Post payment
        Rewards & loyalty
        Safety and Security
        Sign up
        Speed
        Top up limit
        Top up source
        Transaction limit
        Transfer to bank
        update
        UX/UI
        cant get voucher
        cant use voucher
        cash in
        CPQR
        CQPR
        ID Card
        fail to add source
        FPS
        grab voucher
        hkid
        laisee
        Launch
        multiple top up source
        no hkid
        P2M
        P2P
        Pay bills
        paystreak
        Peak
        pw
        Rewards & loyalty
        skip top up
        voucher amount
        voucher requirement
        voucher selection
        voucher valid time
        why is needed
        wrong payee
    - If none of the above matches, come up with your own category, similar to as given above.
    - Only category in your response as array of Category:<category>
    """

    prompt=PROMPT.format(user_input=user_input)
    model_response=model(prompt)
    response = model_response.choices[0]['message']['content'].strip()
    print(response)
    return response
# You are a sentiment analyzer for feedbacks given by customers of a bank.
#     For example:
#     Input: Great customer service! Happy with the company
#     Output: Positive

#     Input: Worst Experience ever, wasted lot of time here!
#     Output: Negative

#     Input: Okay service, can do better.
#     Output: Neutral

#     Analyze the sentiment of the following list of text:-{user_input}, generate output for each list element as positive,negative or neutral.

#     Give output as a list in python where each element refers to the corresponding element of input list and make sure not to miss any input and output.

#     Length of your output list should be equal to the length of input list.



    # Given input list-{user_input}. Get the correct size of the list and,
    # Analyze the sentiment of each of the list elements as Positive, Negative or Neutral.

    # For each of your analyzation give your degree of certainty or reliability associated with the accuracy of your analysis in float data type.

    # In your output:
    # - Give list of your sentiment analyzation for each list element input.
    # - Give another list of corresponding confidence scores for each element.
    # There are 10 input texts in {user_input}.
    # For each and every input text, strictly do the following:

    # - Aspect Extraction: Identify and extract the aspect based sentiment from the text. Identify sensible phrase not single words.
    # - Sentiment Analysis: After identifying the aspects,perform sentiment analysis on each aspect individually, as - 'Positive', 'Negative' or 'Neutral'.
    # - For above analysis give your degree of certainity of identified sentiment in floating numbers.
    # - Only give sentiment, certainity and aspect in your response as- [Input Text <index>: Sentiment: <your sentiment> : Certainity:<your certainity> : Aspect:<category>.....].