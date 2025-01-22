from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import json
import requests
 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

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
