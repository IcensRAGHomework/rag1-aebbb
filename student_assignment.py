from model_configurations import get_model_configuration

from langchain_openai import AzureChatOpenAI

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import (
    RunnableLambda,
    ConfigurableFieldSpec,
    RunnablePassthrough,
)

from pydantic import BaseModel, Field
from typing import List

import traceback
import requests
import json
import re

gpt_chat_version = 'gpt-4o'
gpt_config = get_model_configuration(gpt_chat_version)

class holiday(BaseModel):
    date: str = Field(..., description="Date of holiday")
    name: str = Field(..., description="Name of holiday.")

class holiday_list(BaseModel):
    Result: List[holiday]

# Custom parser
def extract_json(message: AIMessage) -> List[dict]:
    """Extracts JSON content from a string where JSON is embedded between \`\`\`json and \`\`\` tags.

    Parameters:
        text (str): The text containing the JSON content.

    Returns:
        list: A list of extracted JSON strings.
    """
    text = message.content
    # Define the regular expression pattern to match JSON blocks
    pattern = r"\`\`\`json(.*?)\`\`\`"

    # Find all non-overlapping matches of the pattern in the string
    matches = re.findall(pattern, text, re.DOTALL)

    # Return the list of matched JSON strings, stripping any leading or trailing whitespace
    try:
        return [json.loads(match.strip()) for match in matches]
    except Exception:
        raise ValueError(f"Failed to parse: {message}")

def generate_hw01(question):
    llm = AzureChatOpenAI(
            model=gpt_config['model_name'],
            deployment_name=gpt_config['deployment_name'],
            openai_api_key=gpt_config['api_key'],
            openai_api_version=gpt_config['api_version'],
            azure_endpoint=gpt_config['api_base'],
            temperature=gpt_config['temperature']
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "當使用者對特定年份的台灣行事曆做查詢時, 請列出該月份所有的國定紀念日"
                "Output your answer as JSON that  "
                "matches the given schema: \`\`\`json\n{schema}\n\`\`\`. "
                "Make sure to wrap the answer in \`\`\`json and \`\`\` tags",
            ),
            ("human", "{query}"),
        ]
    ).partial(schema=holiday_list.model_json_schema())

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=holiday_list)

    chain = prompt | llm | parser
    response = chain.invoke({"query": question})

    # convert to json format
    return json.dumps(response, ensure_ascii = False)


def generate_hw02(question):

    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )
    query_year = llm.invoke([
        SystemMessage(content="只回答問題中的年份並用數字表示"),
        HumanMessage(content=question),
    ]).content
    query_month = llm.invoke([
        SystemMessage(content="只回答問題中的月份並用數字表示"),
        HumanMessage(content=question),
    ]).content

    api_url = "https://calendarific.com/api/v2/holidays?&api_key=JQeWnmY3xqc6y2jRtEhdL58tQY3lKdA5&country=TW&language=zh&year="+str(query_year)+"&month="+str(query_month)
    webapi_response = requests.get(api_url)
    cal_json = webapi_response.json()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Output your answer as JSON that  "
                "matches the given schema: \`\`\`json\n{schema}\n\`\`\`. "
                "Make sure to wrap the answer in \`\`\`json and \`\`\` tags",
            ),
            ("human", "{inputdata}"),
        ]
    ).partial(schema=holiday_list.model_json_schema())

    parser = JsonOutputParser(pydantic_object=holiday_list)

    chain = prompt | llm
    response = chain.invoke({"inputdata": cal_json})

    return response.content[7:-3]



class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []

store = {}

def get_by_session_id(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryHistory()
    return store[session_id]

def generate_hw03(question2, question3):
    cal_list = generate_hw02(question2)

    llm = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    prompt = ChatPromptTemplate.from_messages([
        ("ai", "{holiday_list}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])
    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_by_session_id,
        input_messages_key="question",
        history_messages_key="history",
    )

    result1 = chain_with_history.invoke(
        {"holiday_list": cal_list,
         "question": question3},
        config={"configurable": {"session_id": "cal_chat_history"}}
    )

    result_add_or_not = chain_with_history.invoke(
        {"holiday_list": cal_list,
         "question": "這節日如果不在之前的清單, 並且需要被加入, 請回答 true, 反之則回答 false"},
        config={"configurable": {"session_id": "cal_chat_history"}}
    )

    result_reason = chain_with_history.invoke(
        {"holiday_list": cal_list,
         "question": "只用一行來解釋一下需要加入或不加入的原因, 並列出目前已存在的所有節日名稱"},
        config={"configurable": {"session_id": "cal_chat_history"}}
    )
    reason = result_reason.content.replace("\"", "") # remove " inside string

    final_info = " \"add\": {0},   \"reason\": \"{1}\" "
    final_response = final_info.format(result_add_or_not.content.lower(), reason)
    final_response = " { \"Result\": {  " + final_response + " }  }"

    return final_response
