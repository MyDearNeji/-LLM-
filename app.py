import streamlit as st
from operator import itemgetter
from urllib.parse import quote_plus
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDataBaseTool
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
import os

# 你的数据库连接信息
username = "root"
password = ""  
database = "sky_take_out"
host = "localhost"
port = "3306"

# 对密码进行 URL 编码
encoded_password = quote_plus(password)

# 构建 MySQL 的连接 URI，使用编码后的密码
uri = f"mysql+pymysql://{username}:{encoded_password}@{host}:{port}/{database}"

# 创建数据库连接
db = SQLDatabase.from_uri(uri)

os.environ["DASHSCOPE_API_KEY"] = ''#实际使用的LLM的API KEY
from langchain_community.chat_models.tongyi import ChatTongyi


chatLLM = ChatTongyi(
    streaming=True,
)


def clean_sql_query(query):
    if query.startswith("SQLQuery: "):
        query = query[10:]
    if query.startswith("```sql"):
        query = query[6:]
        query = query[:-3]
    return query

answer_prompt = PromptTemplate.from_template(
    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

    Question: {question}
    SQL Query: {query}
    SQL Result: {result}
    Answer: 
    note:You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question. Your response should include the SQL statement, a table of the query results, and a natural language answer. Unless the user specifies in the question a specific number of examples to obtain, query for all results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database. Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers. Pay attention to use only the column names you can see in the tables below. Be careful not to query for columns that do not exist. Also, pay attention to which column is in which table. Pay attention to use the CURDATE() function to get the current date, if the question involves 'today'. Unless otherwise specified by the user, all responses should be in Chinese.
    """
)

generate_query = create_sql_query_chain(chatLLM, db)

execute_query = QuerySQLDataBaseTool(db=db)

answer = answer_prompt | chatLLM | StrOutputParser()

chain = (
    RunnablePassthrough.assign(query=generate_query)
    .assign(cleaned_query=lambda context: clean_sql_query(context["query"]))
    .assign(
        result=itemgetter("cleaned_query") | execute_query
    )
    | answer
)

# Streamlit界面
st.title('SQL Query Generator')

# 聊天界面
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 输入对话框
if prompt := st.chat_input("输入你的问题"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        response = generate_query.invoke({"question": prompt})
        response=clean_sql_query(response)
        print(response)
        # 调用链
        prompt=prompt+"只输出sql语句"
        response = chain.invoke({"question": prompt})
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"发生错误：{str(e)}")
